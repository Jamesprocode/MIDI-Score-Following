"""Which cue is better, per take, and does each cue's stand-alone quality explain it?

"Better cue" for a take: the single-cue follower with the higher bar accuracy,
i.e. B7 with weights (harmony 0, rhythm 1) vs (harmony 1, rhythm 0), everything
else identical (sweep_fixed_weights.py frames). This script adds what that
comparison rests on:

  margin        rhythm-only minus harmony-only bar accuracy per take
  takes agree   do the two takes of a song (preferred, alternate) favour the
                same cue?
  stand-alone   the cue on its own, outside the follower:
                - rhythm: the network's downbeat F-measure on the take (played-
                  tempo condition; ../yamaha_timing/results/samba_breakdown)
                - harmony: frame-level chord accuracy of the harmonic
                  observation (argmax chord class vs annotated chord, triad
                  vocabulary, frames with piano notes held)
                and Spearman rank correlations with the single-cue followers.

  ../yamaha_timing/.venv/bin/python cue_reliability.py --samba "../../Dataset/SAMBA dataset"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
import follower_data  # noqa: E402
import run_study  # noqa: E402
from analyze_sweep import load  # noqa: E402


def chord_accuracy(data):
    """Argmax chord class of the harmonic observation vs the annotated chord, on frames with notes held."""
    case = run_study.get_features(data, {"triad"})["triad"]
    scores, available, _ = case["features"]["active_pooled"]
    target, valid = case["targets"]["target"], case["targets"]["valid"]
    keep = valid & available
    guess = scores.argmax(axis=1)
    return float((guess[keep] == target[keep]).mean()), float(available[valid].mean())


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "sweep_b7")
    parser.add_argument("--breakdown", type=Path,
                        default=HERE.parent / "yamaha_timing" / "results" / "samba_breakdown" / "per_take.csv")
    args = parser.parse_args()
    takes, grid, accuracy, lost, _, _ = load(args.run)
    g01, g10, g11 = grid.index((0.0, 1.0)), grid.index((1.0, 0.0)), grid.index((1.0, 1.0))
    with open(args.breakdown) as handle:
        downbeat = {f"{r['protocol']}/{r['name']}": float(r["downbeat_f_measure"]) for r in csv.DictReader(handle)}
    data_by_id = {d["performance_id"]: d for d, _ in follower_data.samba_follow_takes(args.samba, protocols=("sync",))}
    rows = []
    for i, take in enumerate(takes):
        acc_h, coverage = chord_accuracy(data_by_id[take])
        rows.append({"take": take, "song": take.split("/")[1], "rhythm_only": accuracy[i, g01],
                     "harmony_only": accuracy[i, g10], "fixed_1_1": accuracy[i, g11],
                     "margin": accuracy[i, g01] - accuracy[i, g10], "rhythm_only_lost": bool(lost[i, g01]),
                     "network_downbeat_f": downbeat[take], "harmony_chord_accuracy": acc_h,
                     "piano_notes_held": coverage})
    with open(args.run / "cue_reliability.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    margin = np.array([r["margin"] for r in rows])
    songs = sorted({r["song"] for r in rows})
    pairs = [[r for r in rows if r["song"] == s] for s in songs]
    pairs = [p for p in pairs if len(p) == 2]
    agree = sum((p[0]["margin"] > 0) == (p[1]["margin"] > 0) for p in pairs)
    song_rhythm = sum(all(r["margin"] > 0 for r in p) for p in pairs)
    song_harmony = sum(all(r["margin"] < 0 for r in p) for p in pairs)
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    summary = {
        "takes": len(rows), "songs": len(songs),
        "rhythm_better_takes": int((margin > 0).sum()), "harmony_better_takes": int((margin < 0).sum()),
        "ties": int((margin == 0).sum()),
        "margin_quartiles": [float(q) for q in np.percentile(margin, [25, 50, 75])],
        "takes_with_abs_margin_below_0.05": int((np.abs(margin) < .05).sum()),
        "takes_with_abs_margin_above_0.20": int((np.abs(margin) > .20).sum()),
        "rhythm_better_mean_margin": float(margin[margin > 0].mean()),
        "harmony_better_mean_margin": float(-margin[margin < 0].mean()),
        "song_pairs": len(pairs), "pairs_agreeing": int(agree),
        "songs_rhythm_both_takes": int(song_rhythm), "songs_harmony_both_takes": int(song_harmony),
        "rhythm_only_lost_takes": int(col("rhythm_only_lost").sum()),
        "spearman_rhythm_only_vs_downbeat_f": spearman(col("rhythm_only"), col("network_downbeat_f")),
        "spearman_harmony_only_vs_chord_accuracy": spearman(col("harmony_only"), col("harmony_chord_accuracy")),
        "spearman_margin_vs_downbeat_f": spearman(margin, col("network_downbeat_f")),
        "spearman_margin_vs_chord_accuracy": spearman(margin, col("harmony_chord_accuracy")),
        "downbeat_f_median_rhythm_better": float(np.median(col("network_downbeat_f")[margin > 0])),
        "downbeat_f_median_harmony_better": float(np.median(col("network_downbeat_f")[margin < 0])),
        "chord_accuracy_median_rhythm_better": float(np.median(col("harmony_chord_accuracy")[margin > 0])),
        "chord_accuracy_median_harmony_better": float(np.median(col("harmony_chord_accuracy")[margin < 0])),
        "rhythm_only_median_when_rhythm_better": float(np.median(col("rhythm_only")[margin > 0])),
        "rhythm_only_median_when_harmony_better": float(np.median(col("rhythm_only")[margin < 0])),
        "harmony_only_median_when_rhythm_better": float(np.median(col("harmony_only")[margin > 0])),
        "harmony_only_median_when_harmony_better": float(np.median(col("harmony_only")[margin < 0])),
    }
    (args.run / "cue_reliability.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nper take, sorted by margin (rhythm-only minus harmony-only):")
    for r in sorted(rows, key=lambda r: r["margin"]):
        print(f"  {r['take']:52s} R {r['rhythm_only']:.2f}  H {r['harmony_only']:.2f}  margin {r['margin']:+.2f}  "
              f"downbeat F {r['network_downbeat_f']:.2f}  chord acc {r['harmony_chord_accuracy']:.2f}")


if __name__ == "__main__":
    main()
