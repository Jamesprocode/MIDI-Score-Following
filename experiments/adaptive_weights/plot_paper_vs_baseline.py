"""The paper's follower as written (B3) vs our baseline (B7): one dot per live take, and one take over time.

Per-take bar accuracy comes from the 10-15 runs (follow_b, follow_b67 metrics.csv on ICE, copied locally).
The example take is re-run here with both followers to draw position over time.

  ../yamaha_timing/.venv/bin/python plot_paper_vs_baseline.py --metrics B.csv B67.csv --out fig.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import sweep_fixed_weights as sweep
from run_tempo_variants import VARIANTS, run_variant
import follower_data

EXAMPLE = "sync/51-amazing_grace/preferred"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, default=Path(__file__).resolve().parent.parent.parent / "Dataset" / "SAMBA dataset")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent.parent / "yamaha_timing" / "runs" / "lakh_r3b" / "best_model.pt")
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for f in args.metrics for r in csv.DictReader(open(f)) if r["protocol"] == "sync"]
    acc = {code: {r["performance_id"]: float(r["bar_accuracy"]) for r in rows if r["method"].startswith(code + " ")}
           for code in ("B3", "B7")}
    takes = sorted(acc["B3"])
    meta = {r["performance_id"]: (float(r["written_bpm"]), float(r["played_bpm"])) for r in rows if r["method"].startswith("B7 ")}

    sweep._init(str(args.checkpoint))
    data, take = next(t for t in follower_data.samba_follow_takes(args.samba, protocols=("sync",))
                      if t[0]["performance_id"] == EXAMPLE)
    runs = {name: run_variant(data, take, name) for name in ("B3", "B7")}

    paper, base, band, grey = "#E04F39", "#003057", "#9aa5ad", "#54585A"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.array([acc["B3"][k] for k in takes]); y = np.array([acc["B7"][k] for k in takes])
    ax1.plot([0, 1], [0, 1], color="#c9d5de", lw=1, zorder=0)
    ax1.scatter(x, y, s=24, color=base, edgecolor="white", lw=.5, zorder=3)
    for k, label in ((EXAMPLE, "Amazing Grace"),):
        i = takes.index(k)
        ax1.annotate(label, (x[i], y[i]), xytext=(12, -14), textcoords="offset points", fontsize=9, color=grey,
                     arrowprops=dict(arrowstyle="-", color=grey, lw=.6))
    ax1.set_xlim(-.02, 1.02); ax1.set_ylim(-.02, 1.02); ax1.set_aspect("equal")
    ticks = [0, .25, .5, .75, 1]
    ax1.set_xticks(ticks, ["0%", "25%", "50%", "75%", "100%"]); ax1.set_yticks(ticks, ["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("paper as written (B3): time in correct bar")
    ax1.set_ylabel("baseline (B7): time in correct bar")
    ax1.set_title(f"one dot per live take (52)\nB3 {x.mean():.2f}  →  B7 {y.mean():.2f}", fontsize=11)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    written, played = meta[EXAMPLE]
    t = runs["B3"]["times"]
    bpb = data["beats_per_bar"]
    ax2.plot(t, runs["B3"]["truth_position"] / bpb + 1, color=band, lw=4, label="band (annotated)", zorder=1)
    ax2.plot(t, runs["B7"]["position"] / bpb + 1, color=base, lw=1.6, label=f"baseline B7: starts at the played tempo ({played:.0f} BPM), pull", zorder=3)
    ax2.plot(t, runs["B3"]["position"] / bpb + 1, color=paper, lw=1.6, label=f"paper as written B3: starts at the written tempo ({written:.0f} BPM)", zorder=2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("bar")
    ax2.set_title(f"Amazing Grace (live): B3 {acc['B3'][EXAMPLE]:.0%} vs B7 {acc['B7'][EXAMPLE]:.0%} in the correct bar", fontsize=11)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
