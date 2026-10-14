"""Build the cleaned Lakh training set for the Maezawa baseline.

Steps (counts for each are written to the manifest):
  1. Matched subset: files listed in match_scores.json (LMD-matched, rebuilt
     from LMD-full); each file's song is its best-scoring MSD id.
  2. Remove every file whose original name contains a JazzSAMBA title as whole
     words. SAMBA is the test set, so this errs toward removing too much.
  3. Label-quality checks (yamaha_timing.lakh.CheckConfig): opens; length; a
     labelled meter (time signature present, events on bar lines, bars of at
     most 8 quarter notes); notes follow the tempo-map grid; snare not on beats
     1 & 3 of 4/4 bars. Any meter is kept; the beat is its denominator note.
  4. Groups: files sharing a song id or identical note content form one group,
     so copies filed under different song ids land in the same split.
  5. Within a group, note-identical copies are removed (best match kept);
     distinct arrangements are kept. Training samples one version per group
     per epoch; validation uses each group's best-matched version.
  6. Split by group (stable hash of its first song id): train / validation.
Outputs train.npz, val.npz (notes, tempo and meter maps, per-song group and
meter class), files.csv (one row per matched file with its decision) and
manifest.json (inputs, hashes, rules, counts).

  python prepare_lakh.py --lmd-full ../../Dataset/lmd_full \
      --metadata ../../Dataset/lmd_metadata \
      --samba-catalog "../../Dataset/SAMBA dataset/JazzSAMBA-catalog/songs.csv" \
      --out data/lakh_v2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np

from yamaha_timing.lakh import (
    CheckConfig, check_score, content_hash, read_score, save_pack, score_to_song,
)


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
        fingerprint = content_hash(raw) if passed else None
    except Exception as exc:  # corrupt or unsupported file
        return md5, False, f"unreadable:{type(exc).__name__}", {}, None
    return md5, passed, reason, metrics, fingerprint


def extract(job):
    md5, path = job
    return md5, score_to_song(read_score(path))


def in_validation(key, seed, fraction):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
    return value / 2**64 < fraction


def git_state():
    """HEAD commit and whether this folder had uncommitted changes when the data was built."""
    here = Path(__file__).parent
    try:
        run = lambda *cmd: subprocess.run(["git", *cmd], capture_output=True, text=True, check=True, cwd=here).stdout
        return {"commit": run("rev-parse", "HEAD").strip(), "uncommitted_changes": bool(run("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return None


class Groups:
    """Union-find over string keys."""

    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


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

    path = lambda md5: str(args.lmd_full / md5[0] / f"{md5}.mid")
    with Pool(args.workers) as pool:
        results = pool.map(inspect, [(md5, path(md5)) for md5 in candidates], chunksize=32)
    decisions = {md5: (passed, reason, metrics) for md5, passed, reason, metrics, _ in results}
    fingerprints = {md5: fp for md5, passed, _, _, fp in results if passed}
    reasons = Counter(reason for _, reason, _ in decisions.values())
    passing = [md5 for md5 in candidates if decisions[md5][0]]
    funnel.append(("pass label-quality checks", len(passing)))

    # Groups: a shared song id or identical note content joins files.
    groups = Groups()
    for md5 in passing:
        groups.union(f"song:{best[md5][0]}", f"file:{md5}")
        groups.union(f"content:{fingerprints[md5]}", f"file:{md5}")
    rank = lambda md5: (-best[md5][1], md5)  # best match first
    kept, identical = [], set()
    by_content = defaultdict(list)
    for md5 in passing:
        by_content[(groups.find(f"file:{md5}"), fingerprints[md5])].append(md5)
    for copies in by_content.values():
        copies.sort(key=rank)
        kept.append(copies[0])
        identical.update(copies[1:])
    kept.sort()
    funnel.append(("after removing note-identical copies", len(kept)))
    members = defaultdict(list)
    for md5 in kept:
        members[groups.find(f"file:{md5}")].append(md5)
    group_key = {root: min(best[m][0] for m in files) for root, files in members.items()}
    split = {root: "val" if in_validation(key, args.seed, args.val_fraction) else "train"
             for root, key in group_key.items()}
    funnel.append(("song groups", len(members)))
    cross_song = sum(len({best[m][0] for m in files}) > 1 for files in members.values())

    with Pool(args.workers) as pool:
        songs = dict(pool.map(extract, [(md5, path(md5)) for md5 in kept], chunksize=32))
    pack_stats = {}
    for name in ("train", "val"):
        roots = sorted(root for root in members if split[root] == name)
        ids, group_index, meters, match = [], [], [], []
        for g, root in enumerate(roots):
            for md5 in sorted(members[root], key=rank):
                ids.append(md5)
                group_index.append(g)
                meters.append(decisions[md5][2]["meter"])
                match.append(best[md5][1])
        save_pack(args.out / f"{name}.npz", [songs[m] for m in ids], ids,
                  {"group": np.array(group_index, dtype=np.int64), "meter": np.array(meters, dtype="U16"),
                   "match_score": np.array(match, dtype=np.float32)})
        pack_stats[name] = {"groups": len(roots), "files": len(ids),
                            "notes": int(sum(len(songs[m]["start"]) for m in ids)),
                            "meters": dict(Counter(meters).most_common()),
                            "sha256": sha256(args.out / f"{name}.npz")}
        funnel.append((f"{name}: groups / files", f"{len(roots)} / {len(ids)}"))

    root_of = {md5: groups.find(f"file:{md5}") for md5 in kept}
    with open(args.out / "files.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["md5", "msd_id", "match_score", "decision", "group_key", "meter", "content_hash",
                         "samba_titles", "metrics"])
        for md5 in matched:
            msd_id, score = best[md5]
            if md5 in samba:
                decision = "excluded:samba_title"
            elif not decisions[md5][0]:
                decision = f"failed:{decisions[md5][1]}"
            elif md5 in identical:
                decision = "removed:identical_note_content"
            else:
                decision = f"kept:{split[root_of[md5]]}"
            metrics = decisions.get(md5, (None, None, {}))[2]
            writer.writerow([md5, msd_id, f"{score:.4f}", decision,
                             group_key.get(root_of.get(md5), ""), metrics.get("meter", ""),
                             fingerprints.get(md5, ""), "|".join(samba.get(md5, [])),
                             json.dumps(metrics, sort_keys=True)])

    largest = max(len(files) for files in members.values())
    manifest = {
        "purpose": "Lakh training data for the Maezawa (2023) beat/downbeat-phase baseline",
        "inputs": {
            "lmd_full": str(args.lmd_full.resolve()),
            "match_scores.json": sha256(match_path), "md5_to_paths.json": sha256(names_path),
            "samba_catalog": {"path": str(args.samba_catalog.resolve()), "sha256": sha256(args.samba_catalog),
                              "titles": n_titles},
        },
        "rules": {"checks": CheckConfig().to_dict(), "val_fraction": args.val_fraction, "seed": args.seed,
                  "meters": "all meters; beat = the denominator note (quarter in x/4, eighth in x/8); bars "
                            "follow time-signature events, 4/4 before the first; bars over 8 quarter notes rejected",
                  "groups": "files sharing a best-scoring MSD id or identical note content (start to 1/48 "
                            "quarter note, pitch); note-identical copies removed, distinct arrangements kept",
                  "samba_exclusion": "SAMBA title as whole words in any original file name"},
        "funnel": funnel,
        "check_outcomes": dict(reasons.most_common()),
        "samba_excluded_matched_files": sum(md5 in samba for md5 in matched),
        "groups_spanning_several_song_ids": cross_song,
        "largest_group_files": largest,
        "packs": pack_stats,
        "limit": args.limit or None,
        "code": git_state(), "python": platform.python_version(), "numpy": np.__version__,
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("funnel", "check_outcomes", "groups_spanning_several_song_ids",
                                               "largest_group_files", "packs", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
