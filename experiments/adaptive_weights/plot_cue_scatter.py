"""One dot per live take: bar accuracy of the rhythm-only vs the harmony-only follower (from saved frames).

Reads a variants run (run_tempo_variants.py) holding a rhythm-only and a harmony-only follower, e.g. the B5
single-cue run, and draws the scatter used on the seminar slide.

  ../yamaha_timing/.venv/bin/python plot_cue_scatter.py --run runs/b5_single_cue \
      --rhythm B5_rhythm_only --harmony B5_harmony_only --out fig.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABELS = {"sync/63-la_vie_en_rose/preferred": "La Vie en Rose", "sync/55-bye_bye_blackbird/preferred": "Bye Bye Blackbird"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rhythm", default="B5_rhythm_only")
    parser.add_argument("--harmony", default="B5_harmony_only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "pdf.fonttype": 42})

    takes, x, y = [], [], []
    for f in sorted((args.run / "frames").glob("*.npz")):
        z = np.load(f)
        names = [str(v) for v in z["variants"]]
        right = lambda n: (z["predicted_bar"][names.index(n)] == z["truth_bar"]).mean()
        takes.append(f.stem.replace("__", "/")); x.append(right(args.rhythm)); y.append(right(args.harmony))
    x, y = np.array(x), np.array(y)
    rhythm, harmony, grey = "#3A5DAE", "#E0AA0F", "#54585A"
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color="#c9d5de", lw=1, zorder=0)
    ax.scatter(x, y, s=22, color=np.where(x > y, rhythm, harmony), edgecolor="white", lw=.5, zorder=3)
    for name, label in LABELS.items():
        if name in takes:
            i = takes.index(name)
            ax.annotate(label, (x[i], y[i]), xytext=(-24 if x[i] > .5 else 8, 30 if x[i] > .5 else 10),
                        textcoords="offset points",
                        ha="right" if x[i] > .5 else "left", fontsize=9, color=grey,
                        arrowprops=dict(arrowstyle="-", color=grey, lw=.6))
    ax.text(.97, .05, f"beat tracking better\n({(x > y).sum()} takes)", ha="right", va="bottom", color=rhythm, fontsize=9,
            transform=ax.transAxes)
    ax.text(.04, .95, f"harmony better\n({(y > x).sum()} takes)", ha="left", va="top", color=harmony, fontsize=9,
            transform=ax.transAxes)
    ticks = [0, .25, .5, .75, 1]
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xticks(ticks, ["0%", "25%", "50%", "75%", "100%"]); ax.set_yticks(ticks, ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("bar accuracy, beat tracking only")
    ax.set_ylabel("bar accuracy, harmony only")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out, f"rhythm better {(x > y).sum()}, harmony better {(y > x).sum()}")


if __name__ == "__main__":
    main()
