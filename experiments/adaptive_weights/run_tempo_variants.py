"""B3, B5, B6 and B7 on the 52 live takes, keeping per-check predictions (same frame files as the sweep).

The 10-15 runs (yamaha_follower/run_follow_b.py) kept only per-take summaries.
This re-runs the four tempo variants of the paper's follower so every metric
(including the mean timing error) is computed the same way for all of them:

  B3  per-beat tempo walk, written tempo        (the paper's description)
  B5  per-beat tempo walk, played tempo
  B6  B3 + pull 0.02 toward the target tempo, written tempo
  B7  B3 + pull 0.02 toward the target tempo, played tempo   (the baseline)

  capXX_played / capXX_written: no pull; the tempo grid itself is limited to
      about +-XX% of the target (0.8-1.25x or 0.87-1.15x), the walk is free inside

All use harmony x network downbeat with both weights 1.

  ../yamaha_timing/.venv/bin/python run_tempo_variants.py --samba "../../Dataset/SAMBA dataset" \
      --checkpoint ../yamaha_timing/runs/lakh_r3b/best_model.pt --workers 9
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
import follower_data  # noqa: E402
import run_study  # noqa: E402
from evidence import NetworkPhase  # noqa: E402
from run_follow_b import STEP, played_bpm  # noqa: E402
from yamaha_filter import YamahaConfig, follow  # noqa: E402
from sweep_fixed_weights import _init, piano_roles  # noqa: E402
import sweep_fixed_weights  # noqa: E402

VARIANTS = {
    "B3": (YamahaConfig(), "written"),
    "B5": (YamahaConfig(), "played"),
    "B6": (YamahaConfig(beat_tempo_reversion=0.02), "written"),
    "B7": (YamahaConfig(beat_tempo_reversion=0.02), "played"),
    # B5 with one cue only (the cue analysis on the reported baseline)
    "B5_rhythm_only": (YamahaConfig(emission_power_per_second=0.0), "played"),
    "B5_harmony_only": (YamahaConfig(downbeat_power_per_second=0.0), "played"),
    # hard tempo caps instead of the pull: pure per-beat walk on a narrow tempo grid around the target
    "cap25_played": (YamahaConfig(tempo_min_ratio=0.8, tempo_max_ratio=1.25), "played"),
    "cap15_played": (YamahaConfig(tempo_min_ratio=0.87, tempo_max_ratio=1.15), "played"),
    "cap25_written": (YamahaConfig(tempo_min_ratio=0.8, tempo_max_ratio=1.25), "written"),
}


def run_variant(data, take, name):
    config, tempo = VARIANTS[name]
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    at_eval = np.round((eval_times - times[0]) / STEP).astype(int)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(sweep_fixed_weights._MODEL, take)
    bpm = played_bpm(data) if tempo == "played" else data["initial_bpm"]
    out = follow(times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                 case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], bpm,
                 start_time=data["start_time"], network=network, config=config)
    return {"performance_id": data["performance_id"], "variant": name, "times": eval_times,
            "truth_bar": data["truth_bar_index"][sample], "truth_position": data["truth_position_beats"][sample],
            "bar": out["bar_index"][at_eval], "position": out["position_beats"][at_eval]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "tempo_variants")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--variants", nargs="*", default=["B3", "B5", "B6", "B7"])
    args = parser.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba, protocols=("sync",))
    takes.sort(key=lambda t: -len(t[0]["times"]))
    print(json.dumps({"takes": len(takes), "variants": args.variants, "workers": args.workers}), flush=True)
    pending = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_variant, data, take, name) for data, take in takes for name in args.variants]
        for k, job in enumerate(as_completed(jobs), 1):
            r = job.result()
            parts = pending.setdefault(r["performance_id"], {})
            parts[r["variant"]] = r
            if len(parts) == len(args.variants):
                first = next(iter(parts.values()))
                names = list(args.variants)
                np.savez_compressed(
                    args.out / "frames" / f"{first['performance_id'].replace('/', '__')}.npz",
                    times=first["times"], variants=np.array(names), truth_bar=first["truth_bar"],
                    truth_position=first["truth_position"],
                    predicted_bar=np.array([parts[n]["bar"] for n in names]),
                    predicted_position=np.array([parts[n]["position"] for n in names]),
                    role=piano_roles(args.samba, first["performance_id"], first["times"]).astype(str))
                pending.pop(first["performance_id"])
                print(json.dumps({"done_take": first["performance_id"], "tasks_done": k, "of": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
