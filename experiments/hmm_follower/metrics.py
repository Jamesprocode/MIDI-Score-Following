"""Follower metrics from saved frames: one definition for every run (CODE_REVIEW finding 7). It replaces the four
copies (run_follow.position_metrics, metrics_from_frames.take_metrics, summarize_bar_beat.per_take,
jiang_metrics.frame_metrics), which agree, and gives the same numbers (tests/test_metrics.py).

Per take and follower, over the run's checks (every 0.2 s; a heads run has checks only inside the heads):
  bar    share of checks with the predicted bar = the true bar (the same bar of the same chorus)
  beat   share of checks with floor(predicted place) = floor(true place), places in beats from the take's start
  off    share of notes where the follower is more than 0.5 beat off: Jiang et al. (2025)'s reset check, without the
         resets. Notes are the piano and horn onsets (run_jiang_resets.note_onsets; the horns from the run's horn
         set), each checked at the first check at or after it, inside a followed stretch
  ms     mean |predicted - true place| in milliseconds at the local annotated tempo, over the checks in the right bar
  lost   the median |predicted - true place| over the take's checks is more than 4 beats
Groups: live (sync), click (async), all. Means over takes; lost is a count. Paired tests: per song, the two takes
of a song averaged first, Wilcoxon signed-rank.

``--section`` keeps only the checks (and notes) in one kind of section (sections.py: an 8-kind or a coarse label).

  cd experiments && yamaha_timing/.venv/bin/python -m hmm_follower.metrics RUN --samba "../Dataset/SAMBA dataset" \\
      --pairs B5+v2w1:B5 [--section head] [--horns ../Dataset/horn_sets/crepe_all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ("bar", "beat", "off", "ms", "lost")
GROUPS = {"live": ("sync",), "click": ("async",), "all": ("sync", "async")}


def take_metrics(z, notes=None, keep=None):
    """{follower: {metric: value}} for one frame file ``z`` (np.load). ``notes``: note onset times (s), or None to
    skip "off". ``keep``: a boolean mask over the checks (a section), or None for all."""
    times, truth_bar, truth = z["times"], z["truth_bar"], z["truth_position"]
    segment = z["segment"] if "segment" in z.files else np.zeros(len(times), int)
    keep = np.ones(len(times), bool) if keep is None else np.asarray(keep, bool)
    beats_per_second = np.empty(len(times))
    at = []
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        beats_per_second[idx] = np.gradient(truth[idx], times[idx]) if len(idx) > 1 else np.nan
        if notes is not None:
            inside = notes[(notes > times[idx[0]]) & (notes <= times[idx[-1]])]
            at.append(idx[np.searchsorted(times[idx], inside - 1e-9, side="left")])
    at = np.concatenate(at) if at else np.zeros(0, int)
    at = at[keep[at]]
    out = {}
    for k, name in enumerate(str(v) for v in z["variants"]):
        bar, position = z["predicted_bar"][k][keep], z["predicted_position"][k][keep]
        error = position - truth[keep]
        right = bar == truth_bar[keep]
        ms = np.abs(error / np.maximum(beats_per_second[keep], 1e-6) * 1000)
        out[name] = {"bar": float(right.mean()), "beat": float((np.floor(position) == np.floor(truth[keep])).mean()),
                     "off": (float((np.abs(z["predicted_position"][k][at] - truth[at]) > .5).mean())
                             if notes is not None and len(at) else np.nan),
                     "ms": float(ms[right].mean()) if right.any() else np.nan,
                     "lost": bool(np.median(np.abs(error)) > 4)}
    return out


def horn_set(runs, horns=None):
    """Set the horn set the runs were made with (their run.json), checked against ``horns`` when given. Horn sets
    are matched by folder name (crepe_all, ...), since a run made on PACE records a PACE path."""
    from .takes import use_horns
    found = set()
    for run in runs:
        for record in sorted(Path(run).glob("run*.json")):
            found.add(json.loads(record.read_text())["horns"])
    names = {Path(f).name for f in found}
    if horns is None:
        local = [f for f in found if f == "raw" or Path(f).is_dir()]
        if len(names) != 1 or not local:
            raise SystemExit("give --horns: " + ("the runs used different horn sets" if len(names) > 1 else
                                                 "no run.json says which horn set the runs used" if not names else
                                                 f"the runs' horn set {sorted(found)} is not on this machine"))
        horns = local[0]
    value = use_horns(horns)
    if names and names != {Path(value).name}:
        raise SystemExit(f"--horns {value} but the runs used {sorted(found)}")
    return value


def per_take(run, samba=None, section=None, notes=True, takes=None):
    """{(protocol, song, take): {follower: metrics}} for every frame file of ``run``. ``takes``: the loaded takes
    ({performance_id: data}) for the notes; needed when ``notes``."""
    from .sections import labels
    out = {}
    for f in sorted((Path(run) / "frames").glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        z = np.load(f)
        protocol, song, take = f.stem.split("__")
        performance_id = f"{protocol}/{song}/{take}"
        keep = None
        if section is not None:
            got = labels(samba, performance_id, z["times"])
            keep = (got["kind"] == section) | (got["coarse"] == section)
            if not keep.any():
                continue
        onsets = None
        if notes:
            from . import legacy  # noqa: F401
            from run_jiang_resets import note_onsets
            onsets = note_onsets(takes[performance_id], samba)
        out[(protocol, song, take)] = take_metrics(z, onsets, keep)
    return out


def group(values, protocols, follower, metric):
    """{(protocol/song, take): value} for the takes of ``protocols``."""
    return {(f"{p}/{s}", t): m[follower][metric] for (p, s, t), m in values.items()
            if p in protocols and follower in m}


def song_test(a, b):
    """Per-song a - b (two takes averaged first): mean, Wilcoxon p, songs better, songs worse, songs."""
    keys = sorted(set(a) & set(b))
    songs = sorted({k[0] for k in keys})
    d = np.array([np.nanmean([a[k] - b[k] for k in keys if k[0] == s]) for s in songs])
    d = d[np.isfinite(d)]
    p = wilcoxon(d).pvalue if len(d) and np.any(d) else 1.
    return float(d.mean()) if len(d) else np.nan, float(p), int((d > 0).sum()), int((d < 0).sum()), len(d)


def summary(values, followers):
    """{group: {follower: {metric: mean over takes (lost: count), "takes": n}}}"""
    out = {}
    for g, protocols in GROUPS.items():
        out[g] = {}
        for v in followers:
            row = {}
            for m in METRICS:
                x = list(group(values, protocols, v, m).values())
                row[m] = int(np.sum(x)) if m == "lost" else (float(np.nanmean(x)) if x else np.nan)
            row["takes"] = len(group(values, protocols, v, "bar"))
            out[g][v] = row
    return out


def report(run, values, pairs=()):
    followers = list(dict.fromkeys(v for m in values.values() for v in m))
    s = summary(values, followers)
    lines = [f"== {run}: " + ", ".join(f"{g} {s[g][followers[0]]['takes']} takes" for g in GROUPS)]
    lines.append(f"{'follower':14s}" + "".join(f"{g + ' bar':>11s}{'beat':>7s}{'off':>7s}{'ms':>6s}{'lost':>5s}"
                                                for g in GROUPS))
    for v in followers:
        lines.append(f"{v:14s}" + "".join(f"{s[g][v]['bar']:11.3f}{s[g][v]['beat']:7.3f}{s[g][v]['off']:7.3f}"
                                          f"{s[g][v]['ms']:6.0f}{s[g][v]['lost']:5d}" for g in GROUPS))
    if pairs:
        lines.append("\nper song, a - b (songs a better / worse; for off and ms, lower is better)")
        for a, b in pairs:
            for g, protocols in GROUPS.items():
                parts = []
                for m in ("bar", "beat", "off"):
                    d, p, up, down, n = song_test(group(values, protocols, a, m), group(values, protocols, b, m))
                    parts.append(f"{m} {d:+.3f} (p {p:.2g}, {up}/{down})")
                lines.append(f"{a + ' - ' + b:24s} {g:5s} {n:2d} songs  " + "   ".join(parts))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--horns", default=None, help="the horn set, if the runs have no run.json saying it")
    parser.add_argument("--pairs", nargs="*", default=[], help="a:b comparisons, reported as a - b")
    parser.add_argument("--section", default=None, help="only checks in this section kind (sections.py)")
    parser.add_argument("--no-notes", action="store_true", help='skip "off" (no takes to load)')
    parser.add_argument("--json", type=Path, default=None, help="also write the per-take metrics here")
    args = parser.parse_args()
    takes = None
    if not args.no_notes:
        horn_set(args.runs, args.horns)
        from .takes import load
        takes = {d["performance_id"]: d for d, _ in load(args.samba)}
    pairs = [tuple(p.split(":")) for p in args.pairs]
    dump = {}
    for run in args.runs:
        values = per_take(run, args.samba, args.section, not args.no_notes, takes)
        print(report(run, values, pairs) + "\n")
        dump[run] = {"/".join(k): v for k, v in values.items()}
    if args.json:
        args.json.write_text(json.dumps(dump, indent=1))


if __name__ == "__main__":
    main()
