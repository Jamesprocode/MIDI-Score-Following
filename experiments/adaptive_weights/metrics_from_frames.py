"""The four follower metrics, computed the same way for any frame files.

Per take, over checks every 0.2 s:
  bar accuracy   = checks in the right bar / all checks
  beat accuracy  = checks in the right beat / all checks
  timing error   = mean of |estimated - true position| in ms (local annotated tempo), over checks in the right bar
  lost           = median |estimated - true position| over the take > 4 beats
Summaries: bar/beat accuracy and timing error are means over takes; lost is a count of takes.

  ../yamaha_timing/.venv/bin/python metrics_from_frames.py runs/sweep_b7 [runs/tempo_variants ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def take_metrics(times, truth_bar, truth_position, bar, position):
    beats_per_second = np.gradient(truth_position, times)
    error = position - truth_position
    right_bar = bar == truth_bar
    ms = np.abs(error / np.maximum(beats_per_second, 1e-6) * 1000)
    return {"bar": float(right_bar.mean()),
            "beat": float((np.floor(position) == np.floor(truth_position)).mean()),
            "ms": float(ms[right_bar].mean()) if right_bar.any() else np.nan,
            "lost": bool(np.median(np.abs(error)) > 4)}


def load_run(run):
    """{follower name: [per-take metric dicts]} for a sweep (grid of weights) or a variants run."""
    out = {}
    for f in sorted((Path(run) / "frames").glob("*.npz")):
        z = np.load(f)
        names = ([f"w_h={h:g}, w_d={d:g}" for h, d in z["grid"]] if "grid" in z.files else [str(v) for v in z["variants"]])
        for k, name in enumerate(names):
            out.setdefault(name, []).append(take_metrics(z["times"], z["truth_bar"], z["truth_position"],
                                                         z["predicted_bar"][k], z["predicted_position"][k]))
    return out


def summary(rows):
    return {"takes": len(rows), "bar_accuracy": np.mean([r["bar"] for r in rows]),
            "beat_accuracy": np.mean([r["beat"] for r in rows]),
            "timing_error_ms": np.nanmean([r["ms"] for r in rows]), "lost_takes": sum(r["lost"] for r in rows)}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(f"== {run}")
        for name, rows in load_run(run).items():
            s = summary(rows)
            print(f"  {name:22s} takes {s['takes']:2d}  bar {s['bar_accuracy']:.3f}  beat {s['beat_accuracy']:.3f}  "
                  f"timing {s['timing_error_ms']:5.0f} ms  lost {s['lost_takes']}")
