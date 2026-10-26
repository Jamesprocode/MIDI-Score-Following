"""First check of the melody cue, before any follower run: at the true place, does the cue prefer it?

For each live take, at the follower's evaluation checks (every 0.2 s), the cue scores every place in the unfolded
chart, using the annotated tempo to turn note ages into beats. Reported per section kind (head, horn solo, other
solo, other):

  fires  share of checks with evidence (at least min_notes horn notes in the window, in a bar the tune covers)
  AUC    chance that the true place scores higher than a place in a *different measure* of the form (ties count
         half): 0.5 = no information, 1 = always right. The same measure in other choruses is left out, since the
         tune cannot tell choruses apart and is not meant to.
  top    share of firing checks where no place in another measure scores higher than the true place

``--reference same`` scores against the take's own head: a code check, which should be near perfect in the heads.

  ../yamaha_timing/.venv/bin/python check_melody_evidence.py --samba "../../Dataset/SAMBA dataset" \
      --out runs/check_other
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from melody_evidence import MelodyConfig, MelodyEvidence, section_kinds
import follower_data

KINDS = ("head", "horn_solo", "other_solo", "other")


def check_take(data, root, config):
    evidence = MelodyEvidence(data, root, config)
    sample = np.arange(0, len(data["times"]), 2)
    times, truth = data["times"][sample], data["truth_position_beats"][sample]
    bpm = 60 * np.gradient(data["truth_position_beats"], data["times"])[sample]
    kinds = section_kinds(root, data["performance_id"], times)
    per_bar = int(round(data["beats_per_bar"] / evidence.step))
    measure = np.repeat([int(b["measure"]) for b in data["bars"]], per_bar)
    n = len(evidence.tune)
    auc, top = np.full(len(times), np.nan), np.full(len(times), np.nan)
    for k, (now, position, tempo) in enumerate(zip(times, truth, bpm)):
        share = evidence.scores(now, tempo, n)
        i = min(int(round(position / evidence.step)), n - 1)
        if share is None or not evidence.known[i]:
            continue
        wrong = evidence.known & (measure != measure[i])
        if wrong.any():
            auc[k] = (share[i] > share[wrong]).mean() + .5 * (share[i] == share[wrong]).mean()
            top[k] = float(share[i] >= share[wrong].max())
    rows = []
    for kind in KINDS:
        here = kinds == kind
        if here.any():
            fired = here & np.isfinite(auc)
            rows.append({"take": data["performance_id"], "kind": kind, "checks": int(here.sum()),
                         "fired": int(fired.sum()), "auc_sum": float(np.nansum(auc[fired])),
                         "top_sum": float(np.nansum(top[fired])), "horns": "+".join(evidence.horns),
                         "reference_notes": int(len(evidence.reference["beat"]))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", choices=("other", "same"), default="other")
    parser.add_argument("--window-beats", type=float, default=MelodyConfig.window_beats)
    parser.add_argument("--min-notes", type=int, default=MelodyConfig.min_notes)
    parser.add_argument("--limit", type=int, default=0, help="only the first N takes (quick look)")
    args = parser.parse_args()
    config = replace(MelodyConfig(), reference=args.reference, window_beats=args.window_beats,
                     min_notes=args.min_notes)
    takes = follower_data.samba_follow_takes(args.samba, protocols=("sync",))
    if args.limit:
        takes = takes[:args.limit]
    rows = []
    for data, _ in takes:
        rows += check_take(data, args.samba, config)
        print(data["performance_id"], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_take.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"config": vars(config) if hasattr(config, "__dict__") else str(config), "takes": len(takes)}
    for kind in KINDS:
        part = [r for r in rows if r["kind"] == kind]
        checks, fired = sum(r["checks"] for r in part), sum(r["fired"] for r in part)
        summary[kind] = {"checks": checks, "fires": round(fired / checks, 3) if checks else None,
                         "auc": round(sum(r["auc_sum"] for r in part) / fired, 3) if fired else None,
                         "top": round(sum(r["top_sum"] for r in part) / fired, 3) if fired else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
