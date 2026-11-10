"""Any cue on its own, before any follower run: at the true place, does the cue prefer it? (CODE_REVIEW step 3; it
takes over from melody_evidence/check_melody_evidence.py and harmony_cue/chord_rank.py, for every cue in
followers.CUES.)

At every check (every 0.2 s), the cue scores every place of the unfolded chart (the follower's place grid, 1/8 beat),
given the true tempo as its BPM hint (ours: the annotated tempo, as check_melody_evidence.py; the follower gets its
own tracked tempo instead). A check "fires" when the cue gives an opinion there (not None) and the true place is on
the chart. Per firing check, with s the cue's log-likelihood over the chart's places and i the true place:

  place AUC    chance that s_i is higher than s at a random place of the chart (ties half). A cue that ignores the
               input scores about 0.5. A cue that cannot tell choruses apart stays below 1 (chord_rank.py's "place
               AUC", here over places on the follower's grid instead of chords)
  measure AUC  the same against the places in the *other* measures of the form only: the same measure in other
               choruses is left out, since a cue that cannot tell choruses apart is not meant to (check_melody_evidence.py)
  top          no place in another measure scores higher than the true place
  bar          share of the best places (ties share) that are in the true measure of the form (any chorus): what the
               cue alone would name
  beat         the same, and within half a beat of the true place in the bar

and, for the scaling work (Alex 11-05; nothing below is used by a follower yet), the cue's numbers themselves:

  truth_minus_max   s_i - max s: the evidence the follower adds now ("max" scaling), per second of weight
  truth_log_softmax s_i - logsumexp s: the true place's log-probability if s is normalised over the chart
  truth_z           (s_i - mean s) / std s: the true place as a z-score over the chart
  mean, std, max    of s over the chart
  other_minus_max   s_j - max s at 8 random places in other measures (seeded per take): the scores elsewhere

Summaries are pooled over the firing checks of each section kind (sections.py: the 4 coarse kinds, then the 8 kinds
indented), for live, click and all takes.

  cd experiments && yamaha_timing/.venv/bin/python -m hmm_follower.check_cue --cue harmony --out RUN \\
      --samba "../Dataset/SAMBA dataset" --horns ../Dataset/horn_sets/crepe_all --split test --workers 8
"""
from __future__ import annotations

from .run import check_record, commit, init_worker, shard  # first: run.py sets one thread per worker before numpy

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import zlib

import numpy as np

from . import takes as T
from .followers import CUES, NEEDS, Inputs, Resources
from .hmm import HMMConfig
from .sections import COARSE_KINDS, KINDS, labels

SAMPLE = 8  # places in other measures kept per check
PER_CHECK = ("place_auc", "measure_auc", "top", "bar", "beat", "truth_minus_max", "truth_log_softmax", "truth_z",
             "mean", "std", "max")
SUMMARY = ("place_auc", "measure_auc", "top", "bar", "beat")


def chart(data, bins_per_beat):
    """The follower's place grid at the start, and for each place: on the chart?, its measure of the form, and
    its place in the bar (beats)."""
    step = 1. / bins_per_beat
    end = float(data["chord_ends_beats"][-1])
    positions = np.arange(int(np.ceil(end / step)) + 1, dtype=float) * step
    starts = np.asarray(data["bar_starts_beats"], dtype=float)
    bar = np.searchsorted(starts, positions, side="right") - 1
    inside = (bar >= 0) & (positions < end)
    form = np.array([int(b["measure"]) for b in data["bars"]])
    measure = np.where(inside, form[np.clip(bar, 0, len(form) - 1)], -1)
    in_bar = positions - starts[np.clip(bar, 0, len(starts) - 1)]
    return positions, inside, measure, in_bar


def check_take(data, take, cue_name, samba, resources, stride=T.CHECK_EVERY):
    sample = np.arange(0, len(data["times"]), stride)
    check_times = data["times"][sample]
    cue = CUES[cue_name](Inputs(data, take, T.update_times(data), Path(samba), resources))
    step = 1. / HMMConfig().bins_per_beat
    positions, inside, measure, in_bar = chart(data, HMMConfig().bins_per_beat)
    truth = data["truth_position_beats"][sample]
    bpm = 60 * np.gradient(data["truth_position_beats"], data["times"])[sample]
    beats_per_bar = float(data["beats_per_bar"])
    rng = np.random.default_rng(zlib.crc32(data["performance_id"].encode()))
    n = len(sample)
    out = {key: np.full(n, np.nan) for key in PER_CHECK}
    out["fired"] = np.zeros(n, bool)
    out["other_minus_max"] = np.full((n, SAMPLE), np.nan)
    for k, (now, place, tempo) in enumerate(zip(check_times, truth, bpm)):
        s = cue.log_likelihood(float(now), float(tempo), positions)
        i = int(round(place / step))
        if s is None or not 0 <= i < len(positions) or not inside[i]:
            continue
        s = np.asarray(s, dtype=float)[:len(positions)]
        on = s[inside]
        top_value = on.max()
        other = inside & (measure != measure[i])
        if not other.any():
            continue
        out["fired"][k] = True
        out["place_auc"][k] = ((s[i] > on).sum() + .5 * ((s[i] == on).sum() - 1)) / (len(on) - 1)
        rest = s[other]
        out["measure_auc"][k] = (s[i] > rest).mean() + .5 * (s[i] == rest).mean()
        out["top"][k] = float(s[i] >= rest.max())
        best = inside & (s >= top_value - 1e-12)
        same = best & (measure == measure[i])
        out["bar"][k] = same.sum() / best.sum()
        gap = np.abs(in_bar[same] - in_bar[i])
        out["beat"][k] = (np.minimum(gap, beats_per_bar - gap) <= .5).sum() / best.sum()
        out["truth_minus_max"][k] = s[i] - top_value
        out["truth_log_softmax"][k] = s[i] - (top_value + np.log(np.exp(on - top_value).sum()))
        out["mean"][k], out["std"][k], out["max"][k] = on.mean(), on.std(), top_value
        out["truth_z"][k] = (s[i] - on.mean()) / on.std() if on.std() > 0 else np.nan
        out["other_minus_max"][k] = rng.choice(rest, SAMPLE) - top_value
    got = labels(samba, data["performance_id"], check_times)
    return {"performance_id": data["performance_id"], "times": check_times, "truth_position": truth, "bpm": bpm,
            "kind": got["kind"], "coarse": got["coarse"], "piano": got["piano"], **out}


def save(folder, r):
    path = folder / f"{r['performance_id'].replace('/', '__')}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: v for k, v in r.items() if k != "performance_id"})
    tmp.replace(path)


def summarize(run):
    """{group: {section: {"checks", "fires", metric means over firing checks}}}, sections: all, coarse and 8 kinds."""
    files = [f for f in sorted((Path(run) / "frames").glob("*.npz")) if not f.stem.endswith(".tmp")]
    out = {}
    for group, protocols in {"live": ("sync",), "click": ("async",), "all": ("sync", "async")}.items():
        zs = [np.load(f) for f in files if f.stem.split("__")[0] in protocols]
        if not zs:
            continue
        pooled = {key: np.concatenate([z[key] for z in zs]) for key in ("fired", "kind", "coarse") + SUMMARY}
        rows = {}
        sections = [("all", np.ones(len(pooled["fired"]), bool))]
        sections += [(c, pooled["coarse"] == c) for c in COARSE_KINDS]
        sections += [(f"  {k}", pooled["kind"] == k) for k in KINDS if k not in ("head", "horn_solo")]  # the finer
        for section, here in sections:
            if not here.any():
                continue
            fired = here & pooled["fired"]
            rows[section] = {"checks": int(here.sum()), "fires": float(fired.mean() / here.mean()),
                             **{m: float(pooled[m][fired].mean()) if fired.any() else None for m in SUMMARY}}
        out[group] = {"takes": len(zs), "sections": rows}
    return out


def table(summary, group="all"):
    s = summary[group]
    f = lambda v: "    -" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"-- {group}: {s['takes']} takes", f"{'section':16s}{'checks':>8s}{'fires':>7s}" +
             "".join(f"{m.replace('_auc', ' AUC'):>13s}" for m in SUMMARY)]
    for section, r in s["sections"].items():
        lines.append(f"{section:16s}{r['checks']:8d}{r['fires']:7.2f}" + "".join(f"{f(r[m]):>13s}" for m in SUMMARY))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cue", choices=sorted(CUES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--horns", required=True, help='horn-set folder (Dataset/horn_sets/...), or "raw"')
    parser.add_argument("--checkpoint", default="", help="the timing network (rhythm)")
    parser.add_argument("--melody-params", default="", help="fitted note-HMM tables, a folder (melody v2)")
    parser.add_argument("--lead-sheets", default="", help="lead-sheet melodies (default: MelodyConfig's)")
    parser.add_argument("--protocols", nargs="+", choices=("sync", "async"), default=["sync", "async"])
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--takes", nargs="*", default=[])
    parser.add_argument("--stride", type=int, default=T.CHECK_EVERY, help="data frames per check (2 = 0.2 s)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    parser.add_argument("--summary-only", action="store_true", help="only print the summary of a finished --out")
    args = parser.parse_args()
    if args.summary_only:
        summary = summarize(args.out)
        print("\n\n".join(table(summary, g) for g in summary))
        return
    resources = Resources(*(str(Path(p).resolve()) if p else "" for p in
                            (args.checkpoint, args.melody_params, args.lead_sheets)))
    need = NEEDS.get(args.cue)
    if need and not getattr(resources, need):
        raise SystemExit(f"the {args.cue} cue needs --{need.replace('_', '-')}")
    horns = T.use_horns(args.horns)
    record = {"mode": "check_cue", "followers": {"cue": args.cue, "stride": args.stride}, "horns": horns, "resources": asdict(resources),
              "split": args.split, "protocols": sorted(args.protocols), "takes": sorted(args.takes),
              "stride": args.stride, "samba": str(args.samba.resolve()), **commit(),
              "started": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
              "python": platform.python_version(), "numpy": np.__version__,
              "hint": "annotated tempo (60 * d truth_position / dt)", "places": "the follower's grid, 1/8 beat"}
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    check_record(args.out / "run.json", record)
    todo = shard(T.load(args.samba, args.protocols, args.split, args.takes), *args.shard, "heads")
    done = {f.stem.replace("__", "/") for f in frames.glob("*.npz") if not f.stem.endswith(".tmp")}
    todo = sorted((t for t in todo if t[0]["performance_id"] not in done), key=lambda t: -len(t[0]["times"]))
    print(json.dumps({"cue": args.cue, "takes": len(todo), "skipped_done": len(done), "shard": list(args.shard)}),
          flush=True)
    if todo:
        with ProcessPoolExecutor(args.workers, initializer=init_worker,
                                 initargs=(resources, NEEDS.get(args.cue) == "checkpoint")) as pool:
            jobs = [pool.submit(check_take, data, take, args.cue, args.samba, resources, args.stride)
                    for data, take in todo]
            for count, job in enumerate(as_completed(jobs), 1):
                r = job.result()
                save(frames, r)
                print(json.dumps({"done_take": r["performance_id"], "fired": round(float(r["fired"].mean()), 3),
                                  "n": count, "of": len(jobs)}), flush=True)
    summary = summarize(args.out)
    (args.out / f"summary_shard{args.shard[0]}of{args.shard[1]}.json").write_text(json.dumps(summary, indent=1))
    print("\n\n".join(table(summary, g) for g in summary))


if __name__ == "__main__":
    main()
