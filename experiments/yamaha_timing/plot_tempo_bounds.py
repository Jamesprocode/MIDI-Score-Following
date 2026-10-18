"""Tempo maps before and after bounding the tempo (eval_tempo_bounds.py), for the slide examples.

Reads results/tempo_bounds/examples.npz (played-tempo condition; the current
decoder and min-gap+fill r=1.3) and draws, per take, the tempo implied by the
annotated beats, the current decoder's beats and the bounded decoder's beats.

  python plot_tempo_bounds.py --data "../../Dataset/SAMBA dataset"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from yamaha_timing.real_data import samba_takes

EXAMPLES = {"52-bernies_tune/alternate": "Bernie's Tune (alternate, live swing)",
            "65-misty/alternate": "Misty (alternate, live ballad)"}
BOUNDED = "min-gap+fill r=1.3"


def tempo_curve(beats):
    gaps = np.diff(beats)
    return beats[:-1] + gaps / 2, 60 / gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/tempo_bounds"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = np.load(args.results / "examples.npz")
    with open(args.results / "per_take.csv") as handle:
        rows = {(r["take"], r["condition"], r["decoder"]): r for r in csv.DictReader(handle)}
    takes = {t.name: t for t in samba_takes(args.data) if t.name in EXAMPLES}
    fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(12, 3.4 * len(EXAMPLES)))
    for ax, (name, title) in zip(np.atleast_1d(axes), EXAMPLES.items()):
        take = takes[name]
        truth = take.beats[(take.beats >= take.span[0]) & (take.beats <= take.span[1])]
        played = 60 / np.median(np.diff(take.beats))
        for label, beats, color, width in (
                ("current decoder", cache[f"{name}|current|beat"], "#9ab6e8", 1.0),
                (f"tempo bounded to ±30% of {played:.0f} BPM", cache[f"{name}|{BOUNDED}|beat"], "#2a78d6", 1.4),
                ("annotated beats", truth, "black", 1.6)):
            x, y = tempo_curve(beats)
            ax.plot(x, y, color=color, lw=width, label=label)
        before, after = rows[(name, "played", "current")], rows[(name, "played", BOUNDED)]
        ax.set_title(f"{title}: beat F {float(before['beat_f']):.2f} → {float(after['beat_f']):.2f}, "
                     f"downbeat F {float(before['downbeat_f']):.2f} → {float(after['downbeat_f']):.2f}", fontsize=11)
        ax.set_ylim(played / 2.2, played * 2.2)
        ax.set_ylabel("tempo (BPM)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=9, loc="upper right", ncol=3, frameon=False)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.results / "tempo_bounds_examples.png", dpi=150)
    print(args.results / "tempo_bounds_examples.png")


if __name__ == "__main__":
    main()
