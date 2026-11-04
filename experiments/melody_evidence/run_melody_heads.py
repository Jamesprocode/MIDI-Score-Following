"""Followers run only inside the heads, where the horns play the tune: how far does each cue alone carry the follower?

Each head (a stretch of sections.csv marked is_head; heads that follow each other are one stretch) is followed on
its own. The follower starts at the head's first instant at the true place, with the take's played tempo, the same
start B5 gets at the top of a take, and runs to the end of the head. Nothing outside the heads is followed or
scored. The single-cue followers use weight 1, as B5 does for harmony and downbeat:

  none        no evidence: counting from the start with the tempo model only (the floor)
  rhythm      network downbeat only
  harmony     piano chord match only
  B5          harmony x downbeat (restarted at each head, so not the whole-take B5)
  v1          melody cue v1 only (melody_evidence.py, lead-sheet tune)
  v2          melody cue v2 only (note_hmm.py)
  v2+rhythm   v2 x downbeat: melody in place of harmony
  B5+v2       B5 plus v2 at weight 0.5 (the whole-take best)

Frame files have the same layout as run_melody_follower.py (summarize_melody_follower.py reads them); the heads of a
take are put one after another, with ``segment`` saying which head each check belongs to.

  ../yamaha_timing/.venv/bin/python run_melody_heads.py --samba "../../Dataset/SAMBA dataset" \\
      --checkpoint ../yamaha_timing/runs/lakh_r3b/best_model.pt --protocols sync async --workers 10 --shard 0 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

import run_melody_follower  # noqa: F401  (sets up the import paths)
from melody_evidence import MelodyConfig, MelodyEvidence, section_kinds
from note_hmm import NoteHMMConfig, NoteHMMEvidence
import follower_data
import run_study
from evidence import NetworkPhase
from run_follow_b import STEP, played_bpm
from yamaha_filter import YamahaConfig
from sweep_fixed_weights import _init, piano_roles
import sweep_fixed_weights
from weighted_filter import follow_weighted

HERE = Path(__file__).resolve().parent

VARIANTS = {  # name: ((w_harmony, w_downbeat, w_melody), melody cue or None)
    "none": ((0., 0., 0.), None),
    "rhythm": ((0., 1., 0.), None),
    "harmony": ((1., 0., 0.), None),
    "B5": ((1., 1., 0.), None),
    "v1": ((0., 0., 1.), "v1"),
    "v2": ((0., 0., 1.), "v2"),
    "v2+rhythm": ((0., 1., 1.), "v2"),
    "B5+v2": ((1., 1., .5), "v2"),
}


def head_stretches(samba, performance_id, times):
    """(first, last + 1) indices into ``times`` of each run of head instants."""
    head = section_kinds(samba, performance_id, times) == "head"
    edges = np.flatnonzero(np.diff(np.concatenate([[0], head.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def run_variant(data, take, name, samba, lead_sheets, params):
    (w_h, w_d, w_m), cue = VARIANTS[name]
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(sweep_fixed_weights._MODEL, take) if w_d else None
    if cue == "v1":
        melody = MelodyEvidence(data, samba, MelodyConfig(reference="written", lead_sheets=str(lead_sheets)))
    elif cue == "v2":
        melody = NoteHMMEvidence(data, samba, NoteHMMConfig(params=str(params), lead_sheets=str(lead_sheets)))
    else:
        melody = None
    weights = (w_h, w_d) if melody is None else (w_h, w_d, w_m)
    keep, bar, position, segment = [], [], [], []
    for k, (first, last) in enumerate(head_stretches(samba, data["performance_id"], eval_times)):
        lo, hi = np.searchsorted(times, [eval_times[first] - 1e-6, eval_times[last - 1] + 1e-6])
        start = float(np.interp(times[lo], data["times"], data["truth_position_beats"]))
        out = follow_weighted(times[lo:hi], probabilities[lo:hi], available[lo:hi], data["chord_starts_beats"],
                              data["chord_ends_beats"], case["targets"]["ids"], data["bar_starts_beats"],
                              data["beats_per_bar"], played_bpm(data), weights, start_time=times[lo],
                              network=network, config=YamahaConfig(), melody=melody, start_position=start)
        at = np.round((eval_times[first:last] - times[lo]) / STEP).astype(int)
        keep.append(sample[first:last])
        bar.append(out["bar_index"][at])
        position.append(out["position_beats"][at])
        segment.append(np.full(last - first, k))
    keep = np.concatenate(keep) if keep else np.zeros(0, int)
    return {"performance_id": data["performance_id"], "variant": name, "times": data["times"][keep],
            "truth_bar": data["truth_bar_index"][keep], "truth_position": data["truth_position_beats"][keep],
            "bar": np.concatenate(bar) if bar else np.zeros(0, int),
            "position": np.concatenate(position) if position else np.zeros(0),
            "segment": np.concatenate(segment) if segment else np.zeros(0, int)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "melody_heads")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    parser.add_argument("--lead-sheets", default=MelodyConfig.lead_sheets)
    parser.add_argument("--params", default=str(HERE / "params" / "crepe_all"), help="for the v2 variants")
    parser.add_argument("--protocols", nargs="+", choices=("sync", "async"), default=["sync"])
    parser.add_argument("--limit", type=int, default=0, help="only the first N takes (quick look)")
    args = parser.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba, protocols=tuple(args.protocols))
    if args.limit:
        takes = takes[:args.limit]
    k, n = args.shard
    if n > 1:  # greedy by take length, deterministic
        load, owner = [0.0] * n, {}
        for data, _ in sorted(takes, key=lambda t: (-len(t[0]["times"]), t[0]["performance_id"])):
            i = load.index(min(load))
            owner[data["performance_id"]] = i
            load[i] += len(data["times"])
        takes = [t for t in takes if owner[t[0]["performance_id"]] == k]
    done = {f.stem.replace("__", "/") for f in (args.out / "frames").glob("*.npz")}
    takes = [t for t in takes if t[0]["performance_id"] not in done]
    takes.sort(key=lambda t: -len(t[0]["times"]))
    print(json.dumps({"takes": len(takes), "skipped_done": len(done), "variants": args.variants,
                      "workers": args.workers, "shard": [k, n]}), flush=True)
    pending = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_variant, data, take, name, args.samba, args.lead_sheets, args.params)
                for data, take in takes for name in args.variants]
        for count, job in enumerate(as_completed(jobs), 1):
            r = job.result()
            parts = pending.setdefault(r["performance_id"], {})
            parts[r["variant"]] = r
            if len(parts) == len(args.variants):
                first, names = next(iter(parts.values())), list(args.variants)
                np.savez_compressed(
                    args.out / "frames" / f"{first['performance_id'].replace('/', '__')}.npz",
                    times=first["times"], variants=np.array(names), truth_bar=first["truth_bar"],
                    truth_position=first["truth_position"], segment=first["segment"],
                    predicted_bar=np.array([parts[v]["bar"] for v in names]),
                    predicted_position=np.array([parts[v]["position"] for v in names]),
                    role=piano_roles(args.samba, first["performance_id"], first["times"]).astype(str),
                    kind=section_kinds(args.samba, first["performance_id"], first["times"]).astype(str))
                pending.pop(first["performance_id"])
                print(json.dumps({"done_take": first["performance_id"], "tasks_done": count, "of": len(jobs)}),
                      flush=True)


if __name__ == "__main__":
    main()
