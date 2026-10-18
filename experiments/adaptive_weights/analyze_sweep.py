"""Summarise the fixed-weight sweep (sweep_fixed_weights.py) and draw the slide figures.

Bar accuracy is the share of 200 ms evaluation instants in the correct bar,
averaged over takes with each take counted equally (as in the report).
Passage-level accuracies pool the instants of each passage type within a take,
then average over the takes that have that passage type.

Three readings of the same runs:
  fixed      one (w_h, w_d) for every take: the best on these 52 takes, and
             (1, 1), Maezawa's product p_h * p_d (= B7)
  per take   the best setting for each take, chosen after seeing its result:
             an optimistic upper bound on choosing weights per song
  per passage the best single setting for each passage type over all takes.
             Descriptive: each run keeps one setting for the whole take, so the
             filter's history in a passage comes from that same setting.

  python analyze_sweep.py --run runs/sweep_b7 --figures ../../Research\\ Proposal/...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROLES = ("head", "piano_solo", "comping", "laid_out")
ROLE_NAMES = {"head": "head\n(melody)", "piano_solo": "piano\nsolo", "comping": "comping\n(other soloist)",
              "laid_out": "piano\nlaid out"}


def load(run):
    with open(run / "metrics.csv") as handle:
        rows = list(csv.DictReader(handle))
    takes = sorted({r["performance_id"] for r in rows})
    grid = sorted({(float(r["w_h"]), float(r["w_d"])) for r in rows})
    index = {g: i for i, g in enumerate(grid)}
    accuracy = np.full((len(takes), len(grid)), np.nan)
    lost = np.zeros((len(takes), len(grid)), dtype=bool)
    for r in rows:
        t, g = takes.index(r["performance_id"]), index[(float(r["w_h"]), float(r["w_d"]))]
        accuracy[t, g] = float(r["bar_accuracy"])
        lost[t, g] = float(r["median_abs_beats"]) > 4
    # passage-level accuracy per take and setting
    role_acc = np.full((len(ROLES), len(takes), len(grid)), np.nan)
    role_frames = np.zeros((len(ROLES), len(takes)), dtype=int)
    for t, take in enumerate(takes):
        z = np.load(run / "frames" / f"{take.replace('/', '__')}.npz")
        order = [index[(float(h), float(d))] for h, d in z["grid"]]
        correct = z["predicted_bar"] == z["truth_bar"][None, :]
        for k, role in enumerate(ROLES):
            mask = z["role"] == role
            role_frames[k, t] = mask.sum()
            if mask.any():
                role_acc[k, t, order] = correct[:, mask].mean(axis=1)
    return takes, grid, accuracy, lost, role_acc, role_frames


def summarise(takes, grid, accuracy, lost, role_acc, role_frames):
    mean = accuracy.mean(axis=0)
    g11, g01, g10, g00 = (grid.index(g) for g in ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
    best = int(np.argmax(mean))
    per_take_best = accuracy.max(axis=1)
    out = {
        "takes": len(takes),
        "single_cue": {"none (tempo only)": mean[g00], "rhythm only (0,1)": mean[g01],
                       "harmony only (1,0)": mean[g10], "both, Maezawa (1,1)": mean[g11]},
        "lost_takes": {"rhythm only": int(lost[:, g01].sum()), "harmony only": int(lost[:, g10].sum()),
                       "both (1,1)": int(lost[:, g11].sum()), "best fixed": int(lost[:, best].sum())},
        "best_fixed": {"w_h": grid[best][0], "w_d": grid[best][1], "bar_accuracy": mean[best]},
        "per_take_oracle": {"bar_accuracy": float(per_take_best.mean()),
                            "takes_where_best_fixed_is_within_0.01_of_oracle":
                                int((per_take_best - accuracy[:, best] <= .01).sum()),
                            "distinct_best_settings": len({tuple(grid[i]) for i in accuracy.argmax(axis=1)})},
        "per_take_best_setting": {takes[t]: {"w_h": grid[i][0], "w_d": grid[i][1], "best": accuracy[t, i],
                                             "maezawa": accuracy[t, g11], "best_fixed": accuracy[t, best]}
                                  for t, i in enumerate(accuracy.argmax(axis=1))},
        "passages": {},
    }
    for k, role in enumerate(ROLES):
        have = role_frames[k] > 0
        if not have.any():
            continue
        acc = np.nanmean(role_acc[k][have], axis=0)
        i = int(np.argmax(acc))
        out["passages"][role] = {
            "takes": int(have.sum()), "frames": int(role_frames[k].sum()),
            "rhythm only": acc[g01], "harmony only": acc[g10], "maezawa (1,1)": acc[g11],
            "best fixed overall": acc[best], "best here": acc[i], "best here w_h": grid[i][0], "best here w_d": grid[i][1],
            "accuracy by setting": {f"{h:g},{d:g}": acc[j] for j, (h, d) in enumerate(grid)}}
    return out


def figures(summary, grid, accuracy, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    rhythm, harmony, both, best_c, grey = "#3A5DAE", "#E0AA0F", "#003057", "#E04F39", "#54585A"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. passage types: rhythm only / harmony only / Maezawa / best for this passage
    passages = [r for r in ROLES if r in summary["passages"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(passages))
    series = [("rhythm only", rhythm, "rhythm only"), ("harmony only", harmony, "harmony only"),
              ("maezawa (1,1)", both, "fixed 1 : 1 (Yamaha)"), ("best here", best_c, "best weights for this passage")]
    width = .2
    for j, (key, color, label) in enumerate(series):
        values = [summary["passages"][p][key] for p in passages]
        ax.bar(x + (j - 1.5) * width, values, width * .92, color=color, label=label)
    for i, p in enumerate(passages):
        s = summary["passages"][p]
        ax.text(x[i] + 1.5 * width, s["best here"] + .015, f"H {s['best here w_h']:g} : R {s['best here w_d']:g}",
                ha="center", va="bottom", fontsize=8, color=best_c, rotation=90)
    ax.set_xticks(x, [ROLE_NAMES[p] for p in passages])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("time in correct bar")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.16))
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_passages.pdf")
    plt.close(fig)

    # 2. per take: fixed 1:1 vs the best setting for that take
    g11 = grid.index((1.0, 1.0))
    order = np.argsort(accuracy[:, g11])
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.vlines(np.arange(len(order)), accuracy[order, g11], accuracy[order].max(axis=1), color=grey, lw=1)
    ax.scatter(np.arange(len(order)), accuracy[order, g11], s=14, color=both, label="fixed 1 : 1 (Yamaha)", zorder=3)
    ax.scatter(np.arange(len(order)), accuracy[order].max(axis=1), s=14, color=best_c,
               label="best weights for this take", zorder=3)
    ax.set_xticks([])
    ax.set_xlabel("52 live takes, sorted by the fixed follower's accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("time in correct bar")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_per_take.pdf")
    plt.close(fig)

    # 3. the whole grid: mean accuracy for every fixed setting
    levels_h = sorted({h for h, _ in grid})
    levels_d = sorted({d for _, d in grid})
    table = np.full((len(levels_h), len(levels_d)), np.nan)
    for j, (h, d) in enumerate(grid):
        table[levels_h.index(h), levels_d.index(d)] = accuracy[:, j].mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    image = ax.imshow(table, origin="lower", cmap="cividis", vmin=0, vmax=1)
    for a in range(len(levels_h)):
        for b in range(len(levels_d)):
            ax.text(b, a, f"{100 * table[a, b]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if table[a, b] < .6 else "black")
    ax.set_xticks(range(len(levels_d)), [f"{v:g}" for v in levels_d])
    ax.set_yticks(range(len(levels_h)), [f"{v:g}" for v in levels_h])
    ax.set_xlabel("rhythm weight $w_R$")
    ax.set_ylabel("harmony weight $w_H$")
    fig.colorbar(image, ax=ax, label="time in correct bar")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_grid.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent / "runs" / "sweep_b7")
    parser.add_argument("--figures", type=Path, default=None, help="folder for the PDF figures")
    args = parser.parse_args()
    takes, grid, accuracy, lost, role_acc, role_frames = load(args.run)
    summary = summarise(takes, grid, accuracy, lost, role_acc, role_frames)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    brief = {k: v for k, v in summary.items() if k != "per_take_best_setting"}
    for p in brief["passages"].values():
        p.pop("accuracy by setting")
    print(json.dumps(brief, indent=1, default=lambda v: round(float(v), 3)))
    if args.figures:
        figures(summary, grid, accuracy, args.figures)


if __name__ == "__main__":
    main()
