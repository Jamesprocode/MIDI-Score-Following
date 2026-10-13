"""Build the cleaned Lakh training set for the Maezawa baseline.

Steps (counts for each are written to the manifest):
  1. Matched subset: files listed in match_scores.json (LMD-matched, rebuilt
     from LMD-full); each file's song is its best-scoring MSD id.
  2. Remove every file whose original name contains a JazzSAMBA title as whole
     words. SAMBA is the test set, so this errs toward removing too much.
  3. Label-quality checks (yamaha_timing.lakh.CheckConfig): opens, length,
     explicit constant 4/4, notes follow the tempo-map grid, snare not on 1 & 3.
  4. One version per song: the best-matched passing file.
  5. Split by song (stable hash of the MSD id): train / validation.
Outputs train.npz, val.npz (notes in beats + tempo maps), files.csv (one row
per matched file with its decision) and manifest.json (inputs, hashes, rules).

  python prepare_lakh.py --lmd-full ../../Dataset/lmd_full \
      --metadata ../../Dataset/lmd_metadata \
      --samba-catalog "../../Dataset/SAMBA dataset/JazzSAMBA-catalog/songs.csv" \
      --out data/lakh_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np

from yamaha_timing.lakh import CheckConfig, check_score, read_score, save_pack, score_to_song


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def samba_matches(md5_to_paths, catalog):
    """md5 -> SAMBA titles found as whole words in any of the file's original names."""
    titles = [row["title"] for row in csv.DictReader(open(catalog, newline=""))]
    patterns = {title: normalize(title) for title in titles}
    hits = {}
    for md5, names in md5_to_paths.items():
        text = " ".join(normalize(re.sub(r"\.midi?$", "", name.split("/")[-1], flags=re.I)) for name in names)
        found = [title for title, pattern in patterns.items() if pattern in text]
        if found:
            hits[md5] = found
    return hits, len(titles)


def inspect(job):
    md5, path = job
    try:
        raw = read_score(path)
        passed, reason, metrics = check_score(raw)
    except Exception as exc:  # corrupt or unsupported file
        return md5, False, f"unreadable:{type(exc).__name__}", {}
    return md5, passed, reason, metrics


def extract(job):
    md5, path = job
    return md5, score_to_song(read_score(path))


def in_validation(msd_id, seed, fraction):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{msd_id}".encode()).digest()[:8], "big")
    return value / 2**64 < fraction


def git_state():
    """HEAD commit and whether this folder had uncommitted changes when the data was built."""
    here = Path(__file__).parent
    try:
        run = lambda *cmd: subprocess.run(["git", *cmd], capture_output=True, text=True, check=True, cwd=here).stdout
        return {"commit": run("rev-parse", "HEAD").strip(), "uncommitted_changes": bool(run("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lmd-full", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="folder with match_scores.json and md5_to_paths.json")
    parser.add_argument("--samba-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N matched files (testing)")
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    match_path = args.metadata / "match_scores.json"
    names_path = args.metadata / "md5_to_paths.json"
    scores = json.load(open(match_path))
    best = {}
    for msd_id, candidates in scores.items():
        for md5, score in candidates.items():
            if md5 not in best or score > best[md5][1]:
                best[md5] = (msd_id, score)
    matched = sorted(best)
    if args.limit:
        matched = matched[:args.limit]
    funnel = [("matched files", len(matched))]

    samba, n_titles = samba_matches(json.load(open(names_path)), args.samba_catalog)
    candidates = [md5 for md5 in matched if md5 not in samba]
    funnel.append(("not a SAMBA title", len(candidates)))

    jobs = [(md5, str(args.lmd_full / md5[0] / f"{md5}.mid")) for md5 in candidates]
    with Pool(args.workers) as pool:
        results = pool.map(inspect, jobs, chunksize=32)
    decisions = {md5: (passed, reason, metrics) for md5, passed, reason, metrics in results}
    reasons = Counter(reason for _, reason, _ in decisions.values())
    passing = [md5 for md5 in candidates if decisions[md5][0]]
    funnel.append(("pass label-quality checks", len(passing)))

    chosen = {}
    for md5 in passing:
        msd_id, score = best[md5]
        if msd_id not in chosen or (score, md5) > (best[chosen[msd_id]][1], chosen[msd_id]):
            chosen[msd_id] = md5
    funnel.append(("one version per song", len(chosen)))
    split = {msd_id: "val" if in_validation(msd_id, args.seed, args.val_fraction) else "train" for msd_id in chosen}
    funnel.append(("train songs", sum(v == "train" for v in split.values())))
    funnel.append(("validation songs", sum(v == "val" for v in split.values())))

    selected = {md5: msd_id for msd_id, md5 in chosen.items()}
    with Pool(args.workers) as pool:
        songs = dict(pool.map(extract, [(md5, str(args.lmd_full / md5[0] / f"{md5}.mid")) for md5 in sorted(selected)],
                              chunksize=32))
    pack_stats = {}
    for name in ("train", "val"):
        ids = sorted(md5 for md5, msd_id in selected.items() if split[msd_id] == name)
        save_pack(args.out / f"{name}.npz", [songs[md5] for md5 in ids], ids)
        pack_stats[name] = {"songs": len(ids), "notes": int(sum(len(songs[m]["start"]) for m in ids)),
                            "sha256": sha256(args.out / f"{name}.npz")}

    with open(args.out / "files.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["md5", "msd_id", "match_score", "decision", "samba_titles", "metrics"])
        for md5 in matched:
            msd_id, score = best[md5]
            if md5 in samba:
                decision = "excluded:samba_title"
            elif not decisions[md5][0]:
                decision = f"failed:{decisions[md5][1]}"
            elif md5 in selected:
                decision = f"selected:{split[msd_id]}"
            else:
                decision = "duplicate_version"
            writer.writerow([md5, msd_id, f"{score:.4f}", decision, "|".join(samba.get(md5, [])),
                             json.dumps(decisions.get(md5, (None, None, {}))[2], sort_keys=True)])

    manifest = {
        "purpose": "Lakh training data for the Maezawa (2023) beat/downbeat-phase baseline",
        "inputs": {
            "lmd_full": str(args.lmd_full.resolve()),
            "match_scores.json": sha256(match_path), "md5_to_paths.json": sha256(names_path),
            "samba_catalog": {"path": str(args.samba_catalog.resolve()), "sha256": sha256(args.samba_catalog),
                              "titles": n_titles},
        },
        "rules": {"checks": CheckConfig().to_dict(), "val_fraction": args.val_fraction, "seed": args.seed,
                  "song_identity": "best-scoring MSD id per file; one version per song (highest match score)",
                  "samba_exclusion": "SAMBA title as whole words in any original file name"},
        "funnel": funnel,
        "check_outcomes": dict(reasons.most_common()),
        "samba_excluded_matched_files": sum(md5 in samba for md5 in matched),
        "packs": pack_stats,
        "limit": args.limit or None,
        "code": git_state(), "python": platform.python_version(), "numpy": np.__version__,
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("funnel", "check_outcomes", "packs", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
