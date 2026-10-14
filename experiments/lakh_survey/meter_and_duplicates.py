"""Evidence for two cleaning choices: are non-4/4 meters trustworthy, and are versions duplicates?

Usage (from this folder):
  python meter_and_duplicates.py ../yamaha_timing ../../Dataset/lmd_full \
      ../yamaha_timing/data/lakh_v1/files.csv results/meter_and_duplicates.json
Run on 2026-10-13 against the first (4/4-only) build, data/lakh_v1, whose
decisions are "selected"/"duplicate_version"; later builds use "kept"/"removed".
Read-only. Meter test uses kick-drum and bass-instrument onsets on quarter-note
beats: (a) does the pattern repeat every 3 or every 4 beats (autocorrelation),
and does that match the stated bar; (b) is the stated beat 1 the strongest
position. Duplicate test hashes each passing file's pitched note content.
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from yamaha_timing.lakh import DRUM_CHANNEL, FX_PROGRAMS, _grid_distance, read_score  # noqa: E402

LMD = Path(sys.argv[2])
KICKS = (35, 36)


def meter_class(signatures):
    meters = [(n, d) for _, n, d in signatures]
    if not meters:
        return "none", None
    distinct = sorted(set(meters))
    if len(distinct) > 1:
        return "changes", None
    n, d = distinct[0]
    return f"{n}/{d}", n * 4 / d


def accents(raw):
    """Kick + bass-instrument onsets per quarter-note beat (within 0.1 beat)."""
    tpb = raw["tpb"]
    onsets = [n[0] / tpb for n in raw["notes"]
              if (n[4] == DRUM_CHANNEL and n[2] in KICKS) or (n[4] != DRUM_CHANNEL and 32 <= n[5] <= 39)]
    if not onsets:
        return None
    onsets = np.array(onsets)
    onsets = onsets[np.abs(onsets - np.round(onsets)) <= 0.1]
    if len(onsets) < 16:
        return None
    return np.bincount(np.round(onsets).astype(int))


def autocorr(series, lag):
    x = series - series.mean()
    if len(x) <= lag or not x.any():
        return np.nan
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def meter_job(md5):
    try:
        raw = read_score(str(LMD / md5[0] / f"{md5}.mid"))
    except Exception:
        return None
    tpb = raw["tpb"]
    pitched = [n for n in raw["notes"] if n[4] != DRUM_CHANNEL and n[5] not in FX_PROGRAMS]
    if len(pitched) < 50:
        return None
    onsets = np.array([n[0] for n in pitched]) / tpb
    on_grid = max(np.mean(_grid_distance(onsets, (4, 3)) <= 1 / 16),
                  np.mean(_grid_distance(onsets, (8, 6, 4, 3)) <= 0.02))
    cls, bar = meter_class(raw["signatures"])
    row = {"meter": cls, "on_grid": bool(on_grid >= 0.9)}
    if cls == "changes":
        # How much of the file is under its most common meter (by beats)?
        sig = raw["signatures"]
        end = max(n[1] for n in pitched)
        spans = Counter()
        for i, (tick, n, d) in enumerate(sig):
            nxt = sig[i + 1][0] if i + 1 < len(sig) else end
            spans[(n, d)] += max(0, nxt - tick)
        total = sum(spans.values()) or 1
        top, share = spans.most_common(1)[0]
        row.update(dominant=f"{top[0]}/{top[1]}", dominant_share=share / total,
                   meters=sorted(f"{n}/{d}" for n, d in spans))
    a = accents(raw)
    if a is not None and len(a) >= 32:
        row["r3"], row["r4"] = autocorr(a.astype(float), 3), autocorr(a.astype(float), 4)
        if bar and float(bar).is_integer() and bar >= 2:
            origin = raw["signatures"][0][0] / tpb
            positions = (np.arange(len(a)) - round(origin)) % int(bar)
            profile = np.bincount(positions, weights=a, minlength=int(bar))
            row["beat1_strongest"] = bool(profile[0] >= profile.max())
    return row


def content_job(md5):
    try:
        raw = read_score(str(LMD / md5[0] / f"{md5}.mid"))
    except Exception:
        return md5, None
    tpb = raw["tpb"]
    notes = sorted((round(n[0] / tpb * 48), n[2]) for n in raw["notes"]
                   if n[4] != DRUM_CHANNEL and n[5] not in FX_PROGRAMS)
    return md5, hashlib.sha1(repr(notes).encode()).hexdigest()


def main():
    rows = list(csv.DictReader(open(sys.argv[3])))
    candidates = [r["md5"] for r in rows if r["decision"] != "excluded:samba_title"]
    sample = random.Random(0).sample(candidates, 12000)
    with Pool(8) as pool:
        meter_rows = [r for r in pool.map(meter_job, sample, chunksize=32) if r]
        passing = [r for r in rows if r["decision"].startswith(("selected", "duplicate", "kept", "removed:identical"))]
        hashes = dict(pool.map(content_job, [r["md5"] for r in passing], chunksize=32))

    out = {"meter": {}, "duplicates": {}}
    by_meter = defaultdict(list)
    for r in meter_rows:
        by_meter[r["meter"]].append(r)
    for cls, items in sorted(by_meter.items(), key=lambda kv: -len(kv[1])):
        tested = [r for r in items if "r3" in r and not np.isnan(r["r3"]) and not np.isnan(r["r4"])]
        triple = [r["r3"] > r["r4"] for r in tested]
        beat1 = [r["beat1_strongest"] for r in tested if "beat1_strongest" in r]
        out["meter"][cls] = {
            "files": len(items), "on_grid": round(np.mean([r["on_grid"] for r in items]), 3),
            "with_kick_or_bass_evidence": len(tested),
            "pattern_repeats_every_3_beats": round(float(np.mean(triple)), 3) if triple else None,
            "stated_beat1_strongest": round(float(np.mean(beat1)), 3) if beat1 else None,
        }
    changes = by_meter.get("changes", [])
    if changes:
        shares = np.array([r["dominant_share"] for r in changes])
        out["meter"]["changes_detail"] = {
            "dominant_meter": Counter(r["dominant"] for r in changes).most_common(5),
            "dominant_covers_95pct_or_more": round(float(np.mean(shares >= 0.95)), 3),
            "dominant_share_median": round(float(np.median(shares)), 3),
            "meter_sets": Counter(" + ".join(r["meters"]) for r in changes).most_common(8),
        }

    songs = defaultdict(list)
    for r in passing:
        songs[r["msd_id"]].append(r["md5"])
    versions = np.array([len(v) for v in songs.values()])
    multi = {s: v for s, v in songs.items() if len(v) > 1}
    exact_dupes = sum(len(v) - len({hashes[m] for m in v if hashes[m]}) for v in multi.values())
    distinct_in_multi = sum(len({hashes[m] for m in v if hashes[m]}) for v in multi.values())
    hash_songs = defaultdict(set)
    for s, v in songs.items():
        for m in v:
            if hashes[m]:
                hash_songs[hashes[m]].add(s)
    files_sorted = np.sort(versions)[::-1]
    out["duplicates"] = {
        "passing_files": int(versions.sum()), "songs": len(songs),
        "songs_with_1_version": int((versions == 1).sum()),
        "songs_with_2_to_5": int(((versions >= 2) & (versions <= 5)).sum()),
        "songs_with_6_to_20": int(((versions >= 6) & (versions <= 20)).sum()),
        "songs_with_over_20": int((versions > 20).sum()), "max_versions": int(versions.max()),
        "share_of_files_in_top_1pct_songs": round(float(files_sorted[:max(1, len(files_sorted) // 100)].sum() / versions.sum()), 3),
        "extra_versions": int(sum(len(v) - 1 for v in multi.values())),
        "extra_versions_identical_note_content": int(exact_dupes),
        "distinct_arrangements_in_multi_version_songs": int(distinct_in_multi),
        "identical_content_shared_across_different_song_ids": int(sum(len(s) > 1 for s in hash_songs.values())),
    }
    Path(sys.argv[4]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
