"""Results of run_melody_follower.py: the four metrics per follower, and bar accuracy by section kind.

Overall metrics are as in adaptive_weights/metrics_from_frames.py (means over takes; lost is a count).
By section kind (head, horn solo, other solo), bar accuracy is the share of that take's checks in the right bar,
averaged over the takes that have that kind. Also: per take, how many takes the melody variant improves or
worsens by more than 2 points of bar accuracy against B5.

  ../yamaha_timing/.venv/bin/python summarize_melody_follower.py runs/melody_follower
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adaptive_weights"))
from metrics_from_frames import summary, take_metrics  # noqa: E402

KINDS = ("head", "horn_solo", "other_solo")


def main():
    run = Path(sys.argv[1])
    files = sorted((run / "frames").glob("*.npz"))
    overall, by_kind, per_take = {}, {}, {}
    for f in files:
        z = np.load(f)
        for k, name in enumerate(str(v) for v in z["variants"]):
            bar, position = z["predicted_bar"][k], z["predicted_position"][k]
            overall.setdefault(name, []).append(take_metrics(z["times"], z["truth_bar"], z["truth_position"],
                                                             bar, position))
            right = bar == z["truth_bar"]
            per_take.setdefault(name, []).append(right.mean())
            for kind in KINDS:
                here = z["kind"] == kind
                if here.any():
                    by_kind.setdefault((name, kind), []).append(right[here].mean())
    print(f"{run}: {len(files)} takes\n")
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
