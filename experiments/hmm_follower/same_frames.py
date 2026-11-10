"""Are two runs' frames the same? Every take in both, every follower in both (or renamed), every array compared bit
for bit. Used to check run.py against the old runners' saved runs.

  cd experiments && yamaha_timing/.venv/bin/python -m hmm_follower.same_frames NEW OLD --rename B5+H1=B5+v2w1

``--rename old=new`` matches a follower the old run named differently. Prints one line per take that differs and
a summary; exits 1 if anything differs or a take is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SHARED = ("times", "truth_bar", "truth_position", "role", "kind", "segment")


def compare(new, old, rename=None, followers=None):
    """{take: [differences]} over the takes in ``old``; [] means identical."""
    rename = rename or {}
    out = {}
    for f_old in sorted((Path(old) / "frames").glob("*.npz")):
        f_new = Path(new) / "frames" / f_old.name
        if not f_new.exists():
            out[f_old.stem] = ["missing in the new run"]
            continue
        a, b = np.load(f_new), np.load(f_old)
        problems = []
        for key in SHARED:
            if (key in a.files) != (key in b.files):
                problems.append(f"{key} only in {'new' if key in a.files else 'old'}")
            elif key in a.files and not np.array_equal(a[key], b[key]):
                problems.append(key)
        names_new = [str(v) for v in a["variants"]]
        names_old = [rename.get(str(v), str(v)) for v in b["variants"]]
        for j, name in enumerate(names_old):
            if followers and name not in followers:
                continue
            if name not in names_new:
                problems.append(f"{name}: not in the new run")
                continue
            i = names_new.index(name)
            for key in ("predicted_bar", "predicted_position"):
                x, y = a[key][i], b[key][j]
                if x.shape != y.shape:
                    problems.append(f"{name} {key}: shape {x.shape} vs {y.shape}")
                elif not np.array_equal(x, y):
                    worst = np.max(np.abs(x.astype(float) - y.astype(float)))
                    problems.append(f"{name} {key}: {int(np.sum(x != y))}/{len(x)} differ (max {worst:.3g})")
        out[f_old.stem] = problems
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("new")
    parser.add_argument("old")
    parser.add_argument("--rename", nargs="*", default=[], help="old_name=new_name")
    parser.add_argument("--followers", nargs="*", default=[], help="compare only these (new names)")
    args = parser.parse_args()
    rename = dict(r.split("=", 1) for r in args.rename)
    result = compare(args.new, args.old, rename, set(args.followers))
    for take, problems in result.items():
        if problems:
            print(f"{take}: " + "; ".join(problems))
    same = sum(not p for p in result.values())
    print(f"{same} of {len(result)} takes identical")
    sys.exit(0 if result and same == len(result) else 1)


if __name__ == "__main__":
    main()
