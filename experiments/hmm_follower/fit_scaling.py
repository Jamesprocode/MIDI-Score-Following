"""Fit the per-cue z-score (scaling.py: "z", "zraw"): one mean and one std per cue, from every score the cue gave on
the training songs, and write them to a JSON file that run.py reads (--scaling-params).

The pool (James, 11-11, "pool B"): every place of the chart at every check (0.2 s) where the cue spoke, over every take
of the split, each check counting the same. It is exact from check_cue's per-check numbers: a check's places have
mean (mean - max) or mean, and spread std, so the pooled variance is the average spread squared plus the variance of
the check means. Two bases: "max" (the score minus the moment's best) and "raw" (the score itself). Third, "within"
(scaling.py's zw): the average spread squared alone, the part of the variance the follower sees (the check means are
added to every place at once, so they cancel); its mean is 0 because each moment is centred on its own mean.

Ours, labelled: the scores are check_cue's, which gives each cue the annotated tempo as its BPM hint (a follower gives
the follower's tempo); the melody tables (crepe_split) were fitted on the same training takes, so its training scores
are in-sample for those tables.

  cd experiments && yamaha_timing/.venv/bin/python -m hmm_follower.fit_scaling --split train \\
      --out hmm_follower/params/zscore_train.json \\
      harmony=melody_evidence/runs/hmmf_check_harmony rhythm=melody_evidence/runs/hmmf_check_rhythm \\
      melody_v2=melody_evidence/runs/hmmf_check_melody_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def checks(run, split):
    """The run's per-check numbers (fired, mean, std, max) over its takes in ``split``, and how many takes."""
    from . import legacy  # noqa: F401
    from samba_split import in_split
    files = [f for f in sorted((Path(run) / "frames").glob("*.npz"))
             if not f.stem.endswith(".tmp") and in_split(f.stem.replace("__", "/"), split)]
    if not files:
        raise SystemExit(f"no frames of split {split!r} in {run}")
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("fired", "mean", "std", "max")}, len(files)


def pooled(p, base):
    """Mean and std of every score (base "max": s - max; "raw": s; "within": s - the moment's mean) over every place
    and every check where the cue fired, each check counting the same; and the share of the variance that lies within
    a moment."""
    fired = p["fired"].astype(bool)
    within = float((p["std"][fired] ** 2).mean())
    if base == "within":
        return {"mean": 0., "std": float(np.sqrt(within)), "within_moment_share": 1.}
    centre = (p["mean"] - p["max"] if base == "max" else p["mean"])[fired]
    total = within + float(centre.var())
    return {"mean": float(centre.mean()), "std": float(np.sqrt(total)), "within_moment_share": within / total}


def fit(runs, split):
    cues, sources = {}, {}
    for cue, run in runs.items():
        p, takes = checks(run, split)
        cues[cue] = {base: pooled(p, base) for base in ("max", "raw", "within")}
        record = json.loads((Path(run) / "run.json").read_text())
        sources[cue] = {"run": str(run), "run_commit": record.get("commit"), "run_split": record.get("split"),
                        "bpm_hint": record.get("hint"), "takes": takes, "checks": int(len(p["fired"])),
                        "checks_fired": int(p["fired"].astype(bool).sum())}
    return {"what": "per-cue mean and std of every score, for scaling.py's z (base max), zraw (base raw) and zw "
                    "(base within)",
            "split": split, "pool": "every chart place at every check where the cue fired; each check counts the same",
            "cues": cues, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="cue=check_cue run folder")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    result = fit(runs, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for cue, numbers in result["cues"].items():
        print(f"{cue:10s} " + "   ".join(f"{b}: mean {v['mean']:+.3f} std {v['std']:.3f}" for b, v in numbers.items())
              + f"   ({result['sources'][cue]['takes']} takes)")
    print(args.out)


if __name__ == "__main__":
    main()
