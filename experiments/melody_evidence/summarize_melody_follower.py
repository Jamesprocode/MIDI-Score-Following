"""Results of run_melody_follower.py: the four metrics per follower, and bar accuracy by section kind.

Overall metrics are as in adaptive_weights/metrics_from_frames.py (means over takes; lost is a count).
By section kind (head, horn solo, other solo), bar accuracy is the share of that take's checks in the right bar,
averaged over the takes that have that kind. Also: per take, how many takes the melody variant improves or
worsens by more than 2 points of bar accuracy against B5.

  ../yamaha_timing/.venv/bin/python summarize_melody_follower.py runs/melody_follower

With several run folders, each take's variants are put together, over the takes found in all of them (e.g. the saved
B5 and v1 frames with the lead-sheet run: runs/melody_follower runs/melody_follower_written).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adaptive_weights"))
from metrics_from_frames import summary, take_metrics  # noqa: E402

KINDS = ("head", "horn_solo", "other_solo")


def frames(runs):
    """Per take, the variants of every run folder together (the first run's copy of a variant wins). Only takes found
    in all of them; their check times (to rounding) and true bars must agree."""
    found = [{f.name: f for f in (run / "frames").glob("*.npz")} for run in runs]
    for name in sorted(set.intersection(*(set(f) for f in found))):
        parts = [np.load(f[name]) for f in found]
        z = parts[0]
        assert all(p["times"].shape == z["times"].shape and np.allclose(p["times"], z["times"], rtol=0, atol=1e-6)
                   and np.array_equal(p["truth_bar"], z["truth_bar"]) for p in parts), name
        take = {key: z[key] for key in ("times", "truth_bar", "truth_position", "kind")}
        take["variants"], take["predicted_bar"], take["predicted_position"] = [], [], []
        for p in parts:
            for k, variant in enumerate(str(v) for v in p["variants"]):
                if variant not in take["variants"]:
                    take["variants"].append(variant)
                    take["predicted_bar"].append(p["predicted_bar"][k])
                    take["predicted_position"].append(p["predicted_position"][k])
        yield take


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    files = list(frames(runs))
    overall, by_kind, per_take = {}, {}, {}
    for z in files:
        for k, name in enumerate(z["variants"]):
            bar, position = z["predicted_bar"][k], z["predicted_position"][k]
            overall.setdefault(name, []).append(take_metrics(z["times"], z["truth_bar"], z["truth_position"],
                                                             bar, position))
            right = bar == z["truth_bar"]
            per_take.setdefault(name, []).append(right.mean())
            for kind in KINDS:
                here = z["kind"] == kind
                if here.any():
                    by_kind.setdefault((name, kind), []).append(right[here].mean())
    print(f"{' + '.join(map(str, runs))}: {len(files)} takes\n")
    print(f"{'follower':12s} {'bar':>6s} {'beat':>6s} {'timing':>8s} {'lost':>5s}   " +
          "   ".join(f"{k:>10s}" for k in KINDS) + "   better/worse than B5")
    base = np.array(per_take.get("B5", []))
    for name, rows in overall.items():
        s = summary(rows)
        kinds = "   ".join(f"{np.mean(by_kind[(name, k)]):10.3f}" if (name, k) in by_kind else f"{'-':>10s}"
                           for k in KINDS)
        change = ""
        if name != "B5" and len(base):
            d = np.array(per_take[name]) - base
            change = f"{(d > .02).sum()} / {(d < -.02).sum()}"
        print(f"{name:12s} {s['bar_accuracy']:6.3f} {s['beat_accuracy']:6.3f} {s['timing_error_ms']:6.0f}ms "
              f"{s['lost_takes']:5d}   {kinds}   {change}")
    print("\nsection columns: bar accuracy inside that kind of section, mean over takes that have it")


if __name__ == "__main__":
    main()
