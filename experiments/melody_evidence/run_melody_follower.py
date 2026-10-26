"""B5 plus the melody cue on the 52 live takes, keeping per-check predictions (same frame files as
adaptive_weights/run_tempo_variants.py, so metrics_from_frames.py reads them).

B5 is the paper's follower started from the played tempo (harmony x network downbeat, both weights 1).
Each variant adds the melody cue (melody_evidence.py; the tune is the other take's horn head) with weight w_m:

  log phi_t(s) = elapsed * (log H_t(s) + log D_t(s) + w_m log M_t(s))

  B5          no melody (control; must equal runs/tempo_variants B5)
  B5+M0.5 ... w_m = 0.5, 1, 2
  B5+M1same   w_m = 1 with the take's own head as the tune (a ceiling for the heads, not a result)

  ../yamaha_timing/.venv/bin/python run_melody_follower.py --samba "../../Dataset/SAMBA dataset" \
      --checkpoint ../yamaha_timing/runs/lakh_r3b/best_model.pt --workers 10 --shard 0 5
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adaptive_weights"))
from melody_evidence import MelodyConfig, MelodyEvidence, section_kinds  # noqa: E402
import follower_data  # noqa: E402
import run_study  # noqa: E402
from evidence import NetworkPhase  # noqa: E402
from run_follow_b import STEP, played_bpm  # noqa: E402
from yamaha_filter import YamahaConfig  # noqa: E402
from sweep_fixed_weights import _init, piano_roles  # noqa: E402
import sweep_fixed_weights  # noqa: E402
from weighted_filter import follow_weighted  # noqa: E402

VARIANTS = {  # name: (w_m, melody config or None)
    "B5": (0., None),
    "B5+M0.5": (.5, MelodyConfig()),
    "B5+M1": (1., MelodyConfig()),
    "B5+M2": (2., MelodyConfig()),
    "B5+M1same": (1., replace(MelodyConfig(), reference="same")),
}


def run_variant(data, take, name, samba):
    w_m, melody_config = VARIANTS[name]
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    at_eval = np.round((eval_times - times[0]) / STEP).astype(int)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(sweep_fixed_weights._MODEL, take)
    melody = None if melody_config is None else MelodyEvidence(data, samba, melody_config)
    weights = (1., 1.) if melody is None else (1., 1., w_m)
    out = follow_weighted(times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                          case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], played_bpm(data),
                          weights, start_time=data["start_time"], network=network, config=YamahaConfig(),
                          melody=melody)
    return {"performance_id": data["performance_id"], "variant": name, "times": eval_times,
            "truth_bar": data["truth_bar_index"][sample], "truth_position": data["truth_position_beats"][sample],
            "bar": out["bar_index"][at_eval], "position": out["position_beats"][at_eval]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "melody_follower")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"),
                        help="run only shard K of N; takes are spread so each shard gets a similar amount of work")
    args = parser.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba, protocols=("sync",))
    k, n = args.shard
    if n > 1:  # greedy by cost (grows with the square of take length), deterministic
        load, owner = [0.0] * n, {}
        for data, _ in sorted(takes, key=lambda t: (-len(t[0]["times"]), t[0]["performance_id"])):
            i = load.index(min(load))
            owner[data["performance_id"]] = i
            load[i] += len(data["times"]) ** 2
        takes = [t for t in takes if owner[t[0]["performance_id"]] == k]
    done = {f.stem.replace("__", "/") for f in (args.out / "frames").glob("*.npz")}
    takes = [t for t in takes if t[0]["performance_id"] not in done]  # resume: skip takes already saved
    takes.sort(key=lambda t: -len(t[0]["times"]))
    print(json.dumps({"takes": len(takes), "skipped_done": len(done), "variants": args.variants,
                      "workers": args.workers, "shard": [k, n]}), flush=True)
    pending = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_variant, data, take, name, args.samba) for data, take in takes
                for name in args.variants]
        for count, job in enumerate(as_completed(jobs), 1):
            r = job.result()
            parts = pending.setdefault(r["performance_id"], {})
            parts[r["variant"]] = r
            if len(parts) == len(args.variants):
                first, names = next(iter(parts.values())), list(args.variants)
                np.savez_compressed(
                    args.out / "frames" / f"{first['performance_id'].replace('/', '__')}.npz",
                    times=first["times"], variants=np.array(names), truth_bar=first["truth_bar"],
                    truth_position=first["truth_position"],
                    predicted_bar=np.array([parts[v]["bar"] for v in names]),
                    predicted_position=np.array([parts[v]["position"] for v in names]),
                    role=piano_roles(args.samba, first["performance_id"], first["times"]).astype(str),
                    kind=section_kinds(args.samba, first["performance_id"], first["times"]).astype(str))
                pending.pop(first["performance_id"])
                print(json.dumps({"done_take": first["performance_id"], "tasks_done": count, "of": len(jobs)}),
                      flush=True)


if __name__ == "__main__":
    main()
