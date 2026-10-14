"""Stage B: the paper's follower (yamaha_filter.py) on JazzSAMBA piano.

Variants, all on 32 positions per 4/4 bar with 50 ms updates; the harmonic
observation is Stage A's (g00) computed every 50 ms:

  B1 continuous            Stage A's tempo dynamics on a 0.5x-2x grid, written
                           tempo as the fixed BPM condition
  B2 continuous+feedback   B1 with the tracked tempo fed back as the condition
  B3 paper                 tempo random walk at each beat + feedback
  B4 paper, harmony only   B3 without the timing network
  B5 paper, played tempo   B3 started from the take's median played tempo (the
                           "target tempo" input assumes the player states it;
                           here it is read from the annotations, an oracle)

Scored at Stage A's 200 ms evaluation instants with the same metrics
(run_follow.position_metrics). Writes metrics.csv, summary.md and
summary.json to --out (default $RUN_DIR).

  python run_follow_b.py --samba DIR --checkpoint best_model.pt --workers 32
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

import follower_data  # noqa: F401  (sys.path)
import run_study  # noqa: E402
from evidence import NetworkPhase  # noqa: E402
from run_follow import position_metrics  # noqa: E402
from yamaha_filter import YamahaConfig, follow  # noqa: E402

VARIANTS = {
    "B1 continuous": (YamahaConfig(tempo_model="continuous", feedback=False), True, "written"),
    "B2 continuous+feedback": (YamahaConfig(tempo_model="continuous"), True, "written"),
    "B3 paper": (YamahaConfig(), True, "written"),
    "B4 paper, harmony only": (YamahaConfig(feedback=False), False, "written"),
    "B5 paper, played tempo": (YamahaConfig(), True, "played"),
    "B6 paper + reversion": (YamahaConfig(beat_tempo_reversion=0.02), True, "written"),
    "B7 paper + reversion, played tempo": (YamahaConfig(beat_tempo_reversion=0.02), True, "played"),
}
STEP = 0.05
_MODEL = None


def _init(checkpoint):
    global _MODEL
    import torch
    from yamaha_timing.training import load_checkpoint
    torch.set_num_threads(1)
    _MODEL = load_checkpoint(checkpoint, "cpu")[0]


def played_bpm(data):
    """Median annotated tempo in quarter notes per minute (oracle "announced" tempo)."""
    durations = [float(b["end_time"]) - float(b["start_time"]) for b in data["bars"]]
    return 60 * data["beats_per_bar"] / float(np.median(durations))


def run_take(data, take, variants):
    started = time.perf_counter()
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    at_eval = np.round((eval_times - times[0]) / STEP).astype(int)
    assert np.allclose(times[at_eval], eval_times)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(_MODEL, take)
    meta = {"performance_id": data["performance_id"], "protocol": take.info["protocol"], "take": take.info["take"],
            "song": take.info["song"], "frames": len(eval_times), "written_bpm": data["initial_bpm"],
            "played_bpm": played_bpm(data)}
    rows = []
    for name in variants:
        config, use_network, tempo = VARIANTS[name]
        before = time.perf_counter()
        prediction = follow(times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                            case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"],
                            meta["played_bpm"] if tempo == "played" else data["initial_bpm"],
                            start_time=data["start_time"], network=network if use_network else None, config=config)
        at = {"position_beats": prediction["position_beats"][at_eval], "bar_index": prediction["bar_index"][at_eval]}
        rows.append({**meta, "method": name, **position_metrics(at, data, sample, eval_times),
                     "ms_per_update": (time.perf_counter() - before) / len(times) * 1000,
                     "final_bpm_condition": float(prediction["bpm_condition"][-1])})
    return rows, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/follow_b")))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--only", nargs="*", default=[], help="performance ids (smoke tests)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba)
    if args.only:
        takes = [t for t in takes if t[0]["performance_id"] in args.only]
    takes.sort(key=lambda t: -len(t[0]["times"]))  # longest first keeps the pool busy
    print(json.dumps({"takes": len(takes), "variants": args.variants, "workers": args.workers}), flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_take, data, take, args.variants) for data, take in takes]
        for k, job in enumerate(as_completed(jobs), 1):
            result, seconds = job.result()
            rows += result
            print(json.dumps({"done": k, "of": len(jobs), "take": result[0]["performance_id"],
                              "seconds": round(seconds)}), flush=True)
    with open(args.out / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ["bar_accuracy", "beat_accuracy", "within2beat_accuracy", "median_abs_beats", "ms_per_update"]
    summary = {}
    for subset, keep in (("async", lambda r: r["protocol"] == "async"), ("sync", lambda r: r["protocol"] == "sync")):
        for method in args.variants:
            members = [r for r in rows if keep(r) and r["method"] == method]
            if not members:
                continue
            lost = [r for r in members if r["median_abs_beats"] > 4]
            kept = [r for r in members if r["median_abs_beats"] <= 4]
            summary[f"{subset} | {method}"] = {
                "takes": len(members), **{k: float(np.mean([r[k] for r in members])) for k in keys},
                "median_take_bar_accuracy": float(np.median([r["bar_accuracy"] for r in members])),
                "takes_bar_acc_ge_0.9": sum(r["bar_accuracy"] >= .9 for r in members), "lost_takes": len(lost),
                "median_abs_lag_ms_not_lost": float(np.median([r["median_abs_lag_ms"] for r in kept])) if kept else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    cols = ["takes", "bar_accuracy", "median_take_bar_accuracy", "beat_accuracy", "takes_bar_acc_ge_0.9",
            "lost_takes", "median_abs_lag_ms_not_lost", "ms_per_update"]
    lines = ["| subset | method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, v in summary.items():
        subset, method = name.split(" | ")
        lines.append(f"| {subset} | {method} | " + " | ".join(
            "-" if v[c] is None else (f"{v[c]:.3f}" if isinstance(v[c], float) and v[c] < 10 else f"{v[c]:.0f}")
            for c in cols) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
