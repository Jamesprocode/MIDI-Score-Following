"""Why do 4/4 files fail the 90% grid rule? Re-score them with finer grids and wider tolerance.

Usage (from this folder): python lakh_middle_band.py results/rows.jsonl
Categories (first match wins), using pitched note starts in beats:
  finer_subdivisions: >=90% on a grid that adds 32nds (1/8) and 16th-triplets (1/6), tol 0.02 beat
  humanized:          >=90% on the 16th/triplet grid with tol 0.0625 beat (a 64th note)
  offset:             >=90% on the 16th/triplet grid after removing one constant offset
  played_in:          everything else (onsets unrelated to the tempo-map grid)
"""
import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from lakh_survey import FX_PROGRAMS, read


def dist(beats, divisions):
    frac = beats - np.floor(beats)
    return np.min([np.abs(frac * d - np.round(frac * d)) / d for d in divisions], axis=0)


def classify(path):
    tpb, notes, *_ = read(path)
    beats = np.array([n[0] for n in notes if not n[4] and n[5] not in FX_PROGRAMS], dtype=np.float64) / tpb
    base = float(np.mean(dist(beats, (4, 3)) <= 0.02))
    fine = float(np.mean(dist(beats, (8, 6, 4, 3)) <= 0.02))
    wide = float(np.mean(dist(beats, (4, 3)) <= 0.0625))
    # Best single offset (in 1/96 beat steps) applied to all notes.
    shifted = max(float(np.mean(dist(beats - k / 96, (4, 3)) <= 0.02)) for k in range(24))
    if fine >= 0.9:
        kind = "finer_subdivisions"
    elif wide >= 0.9:
        kind = "humanized"
    elif shifted >= 0.9:
        kind = "offset"
    else:
        kind = "played_in"
    return {"file": path, "base": base, "fine": fine, "wide": wide, "offset_best": shifted, "kind": kind}


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    band = [r["file"] for r in rows if "error" not in r and r["constant_44"]
            and r["duration_s"] >= 30 and r["n_pitched_notes"] >= 50
            and r["on_grid"] is not None and r["on_grid"] < 0.9]
    with Pool(8) as pool:
        out = pool.map(classify, band, chunksize=8)
    print(f"4/4 files failing the 90% rule: {len(out)}")
    by_band = {}
    for lo, hi in ((0.0, 0.5), (0.5, 0.75), (0.75, 0.9)):
        sel = [o for o in out if lo <= o["base"] < hi]
        by_band[f"{lo}-{hi}"] = (len(sel), Counter(o["kind"] for o in sel).most_common())
    for k, v in by_band.items():
        print(f"  base on-grid {k}: {v[0]:4d} files -> {v[1]}")
    print("overall:", Counter(o["kind"] for o in out).most_common())


if __name__ == "__main__":
    main()
