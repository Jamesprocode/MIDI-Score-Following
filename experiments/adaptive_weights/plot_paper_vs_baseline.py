"""The paper's follower as written (B3) vs the reported baseline (B5): one dot per live take, and one take over time.

Reads the per-check predictions saved by run_tempo_variants.py (runs/tempo_variants/frames), so nothing is
recomputed. B3 starts from the written tempo, B5 from the played tempo; both use the paper's pure per-beat
tempo walk.

  ../yamaha_timing/.venv/bin/python plot_paper_vs_baseline.py --run runs/tempo_variants --out fig.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = "sync/51-amazing_grace/preferred"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "tempo_variants")
    parser.add_argument("--catalog", type=Path,
                        default=HERE.parent.parent / "Dataset" / "SAMBA dataset" / "JazzSAMBA-catalog" / "songs.csv")
    parser.add_argument("--paper", default="B3")
    parser.add_argument("--baseline", default="B5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = {r["song_subdir_basename"]: float(r["bpm"]) for r in csv.DictReader(open(args.catalog))}
    acc, example = {}, None
    for f in sorted((args.run / "frames").glob("*.npz")):
        z = np.load(f)
        names = [str(v) for v in z["variants"]]
        take = f.stem.replace("__", "/")
        right = {n: z["predicted_bar"][i] == z["truth_bar"] for i, n in enumerate(names)}
        acc[take] = (right[args.paper].mean(), right[args.baseline].mean())
        if take == EXAMPLE:
            example = {"times": z["times"], "truth": z["truth_position"],
                       "paper": z["predicted_position"][names.index(args.paper)],
                       "base": z["predicted_position"][names.index(args.baseline)]}
    takes = sorted(acc)
    x = np.array([acc[k][0] for k in takes]); y = np.array([acc[k][1] for k in takes])

    paper_c, base_c, band_c, grey = "#E04F39", "#003057", "#9aa5ad", "#54585A"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax1.plot([0, 1], [0, 1], color="#c9d5de", lw=1, zorder=0)
    ax1.scatter(x, y, s=24, color=base_c, edgecolor="white", lw=.5, zorder=3)
    i = takes.index(EXAMPLE)
    ax1.annotate("Amazing Grace", (x[i], y[i]), xytext=(12, -14), textcoords="offset points", fontsize=9, color=grey,
                 arrowprops=dict(arrowstyle="-", color=grey, lw=.6))
    ax1.set_xlim(-.02, 1.02); ax1.set_ylim(-.02, 1.02); ax1.set_aspect("equal")
    ticks = [0, .25, .5, .75, 1]
    ax1.set_xticks(ticks, ["0%", "25%", "50%", "75%", "100%"]); ax1.set_yticks(ticks, ["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel(f"paper as written ({args.paper}, written tempo): time in correct bar")
    ax1.set_ylabel(f"baseline ({args.baseline}, played tempo): time in correct bar")
    ax1.set_title(f"one dot per live take ({len(takes)})\n{args.paper} {x.mean():.2f}  →  {args.baseline} {y.mean():.2f}", fontsize=11)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    song = EXAMPLE.split("/")[1]
    t = example["times"]
    bars_per_beat = 1 / 4
    span = example["truth"][-1] - example["truth"][0]
    played = 60 * span / (t[-1] - t[0])
    ax2.plot(t, example["truth"] * bars_per_beat + 1, color=band_c, lw=4, label="band (annotated)", zorder=1)
    ax2.plot(t, example["base"] * bars_per_beat + 1, color=base_c, lw=1.6,
             label=f"baseline {args.baseline}: starts at the played tempo (about {played:.0f} BPM)", zorder=3)
    ax2.plot(t, example["paper"] * bars_per_beat + 1, color=paper_c, lw=1.6,
             label=f"paper as written {args.paper}: starts at the written tempo ({written[song]:.0f} BPM)", zorder=2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("bar")
    ax2.set_title(f"Amazing Grace (live): {args.paper} {acc[EXAMPLE][0]:.0%} vs {args.baseline} {acc[EXAMPLE][1]:.0%} in the correct bar",
                  fontsize=11)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
