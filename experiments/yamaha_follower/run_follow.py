"""Stage A: swap the timing network into the existing harmonic score follower.

Same filter, harmonic observation and metrics as the ensemble benchmark's
Maezawa-style control (g00: improvement_study/adaptive_grid.follow_positions,
binary 88-key activity against binary triad templates, z-score scaling, 200 ms
updates, known start and catalog tempo), with the take's piano MIDI as input.
Only the rhythm evidence changes:

  harmony            no rhythm evidence (g00)
  harmony+onset      the rule-based onset-phase cue (improvement_study/rhythm.py,
                     strength 3 as in the r-series); beat phase only, no meter
  harmony+downbeat   the network's bar-phase posterior p_d(s) (evidence.py),
                     tempered like the harmony term (strength 1), so the update
                     is p_h(x|s) p_d(s) as in Maezawa (2023)
  onset / downbeat   that rhythm evidence alone (harmony switched off)
  clock              constant catalog tempo from the known start

each on the benchmark's half-beat grid (8 positions per 4/4 bar) and on an
eighth-beat grid (32 per bar, the network's bar-phase resolution). The BPM
condition is the catalog (written) tempo at every frame. Everything is causal:
the filter sees evidence up to the current update only.

Metrics per take at the 200 ms updates (truth interpolated within annotated
bars): bar accuracy, beat accuracy (same quarter-note beat of the unfolded
score), within-two-beat rate, absolute error in beats, the signed lag in ms
(negative = behind the player, converted with the local annotated tempo), and
the filter's compute time per update. Writes metrics.csv, summary.md and
summary.json to --out (default $RUN_DIR).

  python run_follow.py --samba DIR --checkpoint best_model.pt --workers 16
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

import follower_data  # sets sys.path for the other experiment folders
import adaptive_grid  # noqa: E402
import run_study  # noqa: E402
from evidence import DownbeatEvidence, timing_log_probs  # noqa: E402

VARIANTS = {}
for grid, bins in (("8/bar", 2), ("32/bar", 8)):
    VARIANTS[f"harmony {grid}"] = ("harmony", {"bins_per_beat": bins})
    VARIANTS[f"harmony+onset {grid}"] = ("onset", {"bins_per_beat": bins, "rhythm_strength": 3.})
    VARIANTS[f"harmony+downbeat {grid}"] = ("downbeat", {"bins_per_beat": bins, "rhythm_strength": 1.})
VARIANTS["onset only 32/bar"] = ("onset", {"bins_per_beat": 8, "rhythm_strength": 3., "emission_power_per_second": 0.})
VARIANTS["downbeat only 32/bar"] = ("downbeat", {"bins_per_beat": 8, "rhythm_strength": 1.,
                                                 "emission_power_per_second": 0.})
_MODEL = None


def _init(checkpoint):
    global _MODEL
    import torch
    from yamaha_timing.training import load_checkpoint
    torch.set_num_threads(1)
    _MODEL = load_checkpoint(checkpoint, "cpu")[0]


def position_metrics(prediction, data, sample, times):
    truth = data["truth_position_beats"][sample]
    error = prediction["position_beats"] - truth
    beats_per_second = np.gradient(truth, times)  # local annotated tempo (beats/s)
    lag_ms = error / np.maximum(beats_per_second, 1e-6) * 1000
    return {"bar_accuracy": float((prediction["bar_index"] == data["truth_bar_index"][sample]).mean()),
            "beat_accuracy": float((np.floor(prediction["position_beats"]) == np.floor(truth)).mean()),
            "within2beat_accuracy": float((np.abs(error) <= 2).mean()),
            "mean_abs_beats": float(np.abs(error).mean()), "median_abs_beats": float(np.median(np.abs(error))),
            "median_lag_ms": float(np.median(lag_ms)), "median_abs_lag_ms": float(np.median(np.abs(lag_ms))),
            "out_of_score_rate": float((prediction["position_beats"] >= data["score_end_beats"]).mean())}


def run_take(data, take):
    from rhythm import prepare_rhythm
    started = time.perf_counter()
    sample = np.arange(0, len(data["times"]), 2)  # 200 ms updates, as in the benchmark
    times = data["times"][sample]
    case = run_study.get_features(data, {"triad"})["triad"]
    scores, available, mass = [a[sample] for a in case["features"]["active_pooled"]]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    bpm = data["initial_bpm"]
    tempi = bpm * np.asarray(adaptive_grid.FilterConfig().tempo_ratios)
    t0 = time.perf_counter()
    _, bar_lp = timing_log_probs(_MODEL, take, bpm)
    network_seconds = time.perf_counter() - t0
    evidence = {"harmony": None,
                "onset": prepare_rhythm(data["notes_by_instrument"], times, tempi, bpm),
                "downbeat": DownbeatEvidence(bar_lp, _MODEL.config.frame_rate, times, data["beats_per_bar"],
                                             len(tempi))}
    meta = {"performance_id": data["performance_id"], "protocol": take.info["protocol"],
            "take": take.info["take"], "song": take.info["song"], "frames": len(times),
            "network_seconds_whole_take": network_seconds}
    clock = adaptive_grid.clock_positions(times, data["bar_starts_beats"], bpm, start_time=data["start_time"],
                                          score_end_beats=data["score_end_beats"])
    rows = [{**meta, "method": "clock", **position_metrics(clock, data, sample, times), "ms_per_update": 0.0}]
    for name, (kind, config) in VARIANTS.items():
        before = time.perf_counter()
        prediction = adaptive_grid.follow_positions(
            times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
            case["targets"]["ids"], data["bar_starts_beats"], bpm, start_time=data["start_time"],
            config=config, rhythm=evidence[kind])
        rows.append({**meta, "method": name, **position_metrics(prediction, data, sample, times),
                     "ms_per_update": (time.perf_counter() - before) / len(times) * 1000})
    return rows, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/follow")))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba)
    if args.limit:
        takes = takes[:args.limit]
    print(json.dumps({"takes": len(takes), "variants": ["clock", *VARIANTS], "workers": args.workers}), flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_take, data, take) for data, take in takes]
        for k, job in enumerate(as_completed(jobs), 1):
            result, seconds = job.result()
            rows += result
            print(json.dumps({"done": k, "of": len(jobs), "take": result[0]["performance_id"],
                              "seconds": round(seconds, 1)}), flush=True)
    with open(args.out / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ["bar_accuracy", "beat_accuracy", "within2beat_accuracy", "median_abs_beats", "median_lag_ms",
            "median_abs_lag_ms", "ms_per_update"]
    summary = {}
    for subset, keep in (("async", lambda r: r["protocol"] == "async"),
                         ("sync (repaired)", lambda r: r["protocol"] == "sync"),
                         ("async preferred", lambda r: r["protocol"] == "async" and r["take"] == "preferred")):
        for method in ["clock", *VARIANTS]:
            members = [r for r in rows if keep(r) and r["method"] == method]
            if members:
                summary[f"{subset} | {method}"] = {"takes": len(members),
                                                   **{k: float(np.mean([r[k] for r in members])) for k in keys}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| subset | method | takes | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3)]
    for name, values in summary.items():
        subset, method = name.split(" | ")
        lines.append(f"| {subset} | {method} | {values['takes']} | " +
                     " | ".join(f"{values[k]:.3f}" if abs(values[k]) < 10 else f"{values[k]:.0f}" for k in keys) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
