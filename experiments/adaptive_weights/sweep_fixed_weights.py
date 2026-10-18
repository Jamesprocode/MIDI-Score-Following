"""How much could adaptive cue weighting gain? A fixed-weight sweep of the B7 follower.

The B7 follower (yamaha_follower/yamaha_filter.py) already combines its two cues
as a weighted sum of log-likelihoods:

    log phi(s) = elapsed * (w_h * log H(s) + w_d * log D(s))

with w_h = emission_power_per_second = 1 and w_d = downbeat_power_per_second = 1
(Maezawa's p_h * p_d). This script re-runs B7 on the 52 live JazzSAMBA takes for
a grid of fixed (w_h, w_d) and keeps, per take, the predicted bar at every 200 ms
evaluation instant. analyze_sweep.py then asks:

  - which single fixed setting is best over all takes (what a tuned system gets);
  - which setting is best for each take (a per-take oracle: an upper bound on
    what choosing weights per song could gain; optimistic, since it picks the
    best of many runs);
  - which setting is best in each kind of passage (head, piano solo, comping
    behind another soloist, piano laid out).

Nothing else changes from B7: per-beat tempo walk with the 0.02 pull toward the
played tempo, tempo feedback, 50 ms updates, 32 positions per bar, the Stage A
harmonic observation. The follower code is imported read-only.

  ../yamaha_timing/.venv/bin/python sweep_fixed_weights.py \
      --samba "../../Dataset/SAMBA dataset" --checkpoint ../yamaha_timing/runs/lakh_r3b/best_model.pt \
      --workers 48 --chunk 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
import follower_data  # noqa: E402  (also puts the benchmark modules on sys.path)
import run_study  # noqa: E402
from evidence import NetworkPhase  # noqa: E402
from run_follow import position_metrics  # noqa: E402
from run_follow_b import STEP, played_bpm  # noqa: E402
from yamaha_filter import YamahaConfig, follow  # noqa: E402

B7 = YamahaConfig(beat_tempo_reversion=0.02)
LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
GRID = [(h, d) for h in LEVELS for d in LEVELS]
_MODEL = None


def _init(checkpoint):
    global _MODEL
    import torch
    from yamaha_timing.training import load_checkpoint
    torch.set_num_threads(1)
    _MODEL = load_checkpoint(checkpoint, "cpu")[0]


def piano_roles(samba_root, performance_id, times):
    """Per evaluation instant: what the piano is doing, from sections.csv and
    section_musicians.csv (head / piano_solo / comping / laid_out / other)."""
    protocol, song, take = performance_id.split("/")
    folder = Path(samba_root) / f"{protocol}_audio/songs/{song}/{take}/annotations"
    with open(folder / "sections.csv") as handle:
        sections = list(csv.DictReader(handle))
    with open(folder / "section_musicians.csv") as handle:
        players = {}
        for row in csv.DictReader(handle):
            players.setdefault(row["section_index"], set()).add(row["instrument"])
    roles = np.full(len(times), "other", dtype=object)
    for row in sections:
        if "piano" not in players.get(row["section_index"], set()):
            role = "laid_out"
        elif row["is_solo"] == "True":
            role = "piano_solo" if row["soloist"] == "piano" else "comping"
        elif row["is_head"] == "True":
            role = "head"
        else:
            role = "other"
        inside = (times >= float(row["start_time"])) & (times < float(row["end_time"]))
        roles[inside] = role
    return roles


def run_chunk(data, take, grid):
    """Run B7 with each (w_h, w_d) in ``grid`` on one take; returns metrics rows and per-instant predictions."""
    started = time.perf_counter()
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    at_eval = np.round((eval_times - times[0]) / STEP).astype(int)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(_MODEL, take)
    bpm = played_bpm(data)
    meta = {"performance_id": data["performance_id"], "song": take.info["song"], "take": take.info["take"],
            "frames": len(eval_times), "played_bpm": bpm, "written_bpm": data["initial_bpm"]}
    rows, predicted_bars, predicted_positions = [], [], []
    for w_h, w_d in grid:
        config = replace(B7, emission_power_per_second=w_h, downbeat_power_per_second=w_d)
        prediction = follow(times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                            case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], bpm,
                            start_time=data["start_time"], network=network if w_d else None, config=config)
        at = {"position_beats": prediction["position_beats"][at_eval], "bar_index": prediction["bar_index"][at_eval]}
        rows.append({**meta, "w_h": w_h, "w_d": w_d, **position_metrics(at, data, sample, eval_times)})
        predicted_bars.append(at["bar_index"])
        predicted_positions.append(at["position_beats"])
    return {"performance_id": data["performance_id"], "grid": list(grid), "rows": rows, "times": eval_times,
            "truth_bar": data["truth_bar_index"][sample], "truth_position": data["truth_position_beats"][sample],
            "bars": predicted_bars, "positions": predicted_positions,
            "seconds": time.perf_counter() - started}


def write_take(parts, samba_root, out_dir):
    """Merge one take's chunks (in grid order) into frames/<take>.npz."""
    parts = sorted(parts, key=lambda p: p["order"])
    first = parts[0]
    name = first["performance_id"].replace("/", "__")
    np.savez_compressed(out_dir / "frames" / f"{name}.npz", times=first["times"],
                        grid=np.array([g for p in parts for g in p["grid"]]),
                        truth_bar=first["truth_bar"], truth_position=first["truth_position"],
                        predicted_bar=np.array([b for p in parts for b in p["bars"]]),
                        predicted_position=np.array([x for p in parts for x in p["positions"]]),
                        role=piano_roles(samba_root, first["performance_id"], first["times"]).astype(str))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "sweep_b7")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunk", type=int, default=0,
                        help="weight settings per task (0 = all of a take in one task); smaller spreads long takes "
                             "over more workers")
    parser.add_argument("--only", nargs="*", default=[], help="performance ids (smoke tests)")
    parser.add_argument("--levels", type=float, nargs="*", default=list(LEVELS), help="weight levels for both cues")
    args = parser.parse_args()
    grid = [(h, d) for h in args.levels for d in args.levels]
    chunk = args.chunk or len(grid)
    pieces = [tuple(grid[i:i + chunk]) for i in range(0, len(grid), chunk)]
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba, protocols=("sync",))
    if args.only:
        takes = [t for t in takes if t[0]["performance_id"] in args.only]
    takes.sort(key=lambda t: -len(t[0]["times"]))  # longest first keeps the pool busy
    (args.out / "config.json").write_text(json.dumps(
        {"base": "B7 (YamahaConfig(beat_tempo_reversion=0.02), played tempo)", "levels": args.levels,
         "checkpoint": str(args.checkpoint), "takes": len(takes), "chunk": chunk}, indent=1))
    print(json.dumps({"takes": len(takes), "configs": len(grid), "tasks": len(takes) * len(pieces),
                      "workers": args.workers}), flush=True)
    rows, pending, done_takes = [], {}, 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = {}
        for data, take in takes:
            for order, piece in enumerate(pieces):
                jobs[pool.submit(run_chunk, data, take, piece)] = order
        for job in as_completed(jobs):
            result = job.result()
            result["order"] = jobs[job]
            parts = pending.setdefault(result["performance_id"], [])
            parts.append(result)
            if len(parts) == len(pieces):
                write_take(pending.pop(result["performance_id"]), args.samba, args.out)
                rows += [r for p in sorted(parts, key=lambda p: p["order"]) for r in p["rows"]]
                done_takes += 1
                print(json.dumps({"done": done_takes, "of": len(takes), "take": result["performance_id"],
                                  "task_seconds": round(sum(p["seconds"] for p in parts))}), flush=True)
    with open(args.out / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
