"""One runner for every follower (followers.py), whole takes or heads only, on the Mac or PACE.

  --mode whole   each take followed from its first instant, from place 0 at the played tempo (as
                 melody_evidence/run_melody_follower.py)
  --mode heads   each head (a stretch of sections.csv is_head) followed on its own, started at its first instant at the
                 true place with the played tempo; nothing outside the heads is followed or scored (as
                 run_melody_heads.py)

Everything a run depends on is an argument and goes into OUT/run.json: the followers (cues, weights, scaling, HMM
settings), the horn set, the network checkpoint, the fitted melody tables, the lead sheets, the split, and the code's
commit. A folder holds one run: resuming with other settings is refused.

Frames: one .npz per take in OUT/frames, in the old runners' layout, so summarize_bar_beat.py, summarize_split_heads.py
and jiang_metrics.py read them (``variants`` = the followers; heads runs add ``segment``, the head each check is in).
New: ``info``, JSON with the fitted file each melody cue used for that take.

  cd experiments && yamaha_timing/.venv/bin/python -m hmm_follower.run --mode heads --out RUN \\
      --samba "../Dataset/SAMBA dataset" --horns ../Dataset/horn_sets/crepe_all \\
      --checkpoint yamaha_timing/runs/lakh_r3b/best_model.pt --melody-params melody_evidence/params/crepe_split \\
      --split test --followers B5 v1 v2 v2max B5+v1w1 B5+v2w1 B5+v2maxw1 --workers 10 --shard 0 2
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per worker, as the old
    os.environ.setdefault(_variable, "1")                                           # runners (before numpy loads)

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from . import takes as T
from .followers import FOLLOWERS, Inputs, Resources, build, needs, parse
from .hmm import follow

MODES = ("whole", "heads")


def init_worker(resources, load_model):
    """Each worker: torch on one thread (as sweep_fixed_weights._init; the network's numbers depend on it), and the
    network loaded once."""
    import torch
    torch.set_num_threads(1)
    if load_model:
        resources.model()


def run_one(data, take, label, spec, mode, samba, resources):
    """One follower on one take: predicted bar and place at each check (whole: every check; heads: checks inside
    the heads, with the head's number)."""
    sample = T.check_frames(data)
    check_times = data["times"][sample]
    times = T.update_times(data)
    cues, config, info = build(spec, Inputs(data, take, times, Path(samba), resources))
    plan = (data["chord_ends_beats"][-1], data["bar_starts_beats"], T.played_bpm(data))
    if mode == "whole":
        out = follow(times, cues, *plan, start_time=data["start_time"], config=config)
        at = T.update_index(times, check_times)
        keep, bar, position, segment = sample, out["bar_index"][at], out["position_beats"][at], None
    else:
        keep, bar, position, segment = [], [], [], []
        for k, (first, last) in enumerate(T.head_stretches(samba, data["performance_id"], check_times)):
            lo, hi = np.searchsorted(times, [check_times[first] - 1e-6, check_times[last - 1] + 1e-6])
            start = float(np.interp(times[lo], data["times"], data["truth_position_beats"]))
            out = follow(times[lo:hi], cues, *plan, start_time=times[lo], start_position=start, config=config)
            at = T.update_index(times[lo:hi], check_times[first:last])
            keep.append(sample[first:last])
            bar.append(out["bar_index"][at])
            position.append(out["position_beats"][at])
            segment.append(np.full(last - first, k))
        keep = np.concatenate(keep) if keep else np.zeros(0, int)
        bar = np.concatenate(bar) if bar else np.zeros(0, int)
        position = np.concatenate(position) if position else np.zeros(0)
        segment = np.concatenate(segment) if segment else np.zeros(0, int)
    return {"performance_id": data["performance_id"], "label": label, "keep": keep, "bar": bar,
            "position": position, "segment": segment, "info": info}


def save_take(folder, data, samba, labels, parts):
    """One frame file per take, in the old runners' layout (plus ``info``)."""
    first = parts[labels[0]]
    keep = first["keep"]
    times = data["times"][keep]
    role, kind = T.labels(samba, data["performance_id"], times)
    extra = {} if first["segment"] is None else {"segment": first["segment"]}
    path = folder / f"{data['performance_id'].replace('/', '__')}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, times=times, variants=np.array(labels), truth_bar=data["truth_bar_index"][keep],
                        truth_position=data["truth_position_beats"][keep], **extra,
                        predicted_bar=np.array([parts[v]["bar"] for v in labels]),
                        predicted_position=np.array([parts[v]["position"] for v in labels]),
                        role=role, kind=kind, info=json.dumps({v: parts[v]["info"] for v in labels}))
    tmp.replace(path)


def commit():
    """The code's commit, and whether hmm_follower/ differs from it (PACE runs check out a commit, so it never does)."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                               text=True, check=True).stdout.strip()
        return {"commit": head, "hmm_follower_changed": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "hmm_follower_changed": None}


def check_record(path, record):
    """Write run.json, or check that an existing one describes the same run (the fields that change results)."""
    what = ("mode", "followers", "horns", "resources", "split", "protocols", "takes")
    if path.exists():
        old = json.loads(path.read_text())
        differ = [k for k in what if old.get(k) != record[k]]
        if differ:
            raise SystemExit(f"{path} describes another run (differs in {differ}); use a new --out")
        return
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)


def shard(takes, k, n, mode):
    """Shard k of n: greedy by cost, deterministic (whole: cost grows with the square of the take's length)."""
    if n <= 1:
        return takes
    load, owner = [0.] * n, {}
    for data, _ in sorted(takes, key=lambda t: (-len(t[0]["times"]), t[0]["performance_id"])):
        i = load.index(min(load))
        owner[data["performance_id"]] = i
        load[i] += len(data["times"]) ** (2 if mode == "whole" else 1)
    return [t for t in takes if owner[t[0]["performance_id"]] == k]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--followers", nargs="+", required=True,
                        help=f"named ({', '.join(FOLLOWERS)}) or label=cue:weight,... (followers.py)")
    parser.add_argument("--samba", type=Path, required=True, help="the SAMBA dataset folder")
    parser.add_argument("--horns", required=True, help='horn-set folder (Dataset/horn_sets/...), or "raw"')
    parser.add_argument("--checkpoint", default="", help="the timing network (needed by the rhythm cue)")
    parser.add_argument("--melody-params", default="", help="fitted note-HMM tables, a folder (needed by melody v2)")
    parser.add_argument("--lead-sheets", default="", help="lead-sheet melodies (default: MelodyConfig's)")
    parser.add_argument("--protocols", nargs="+", choices=("sync", "async"), default=["sync", "async"])
    parser.add_argument("--split", choices=("all", "train", "test"), default="all", help="samba_split.py")
    parser.add_argument("--takes", nargs="*", default=[], help="only these performance ids (a quick check)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    args = parser.parse_args()

    try:
        followers = dict(parse(f) for f in args.followers)
    except ValueError as error:
        raise SystemExit(str(error))
    if len(followers) != len(args.followers):
        raise SystemExit("two followers have the same label")
    resources = Resources(*(str(Path(p).resolve()) if p else "" for p in
                            (args.checkpoint, args.melody_params, args.lead_sheets)))
    missing = [n for n in needs(followers.values()) if not getattr(resources, n)]
    if missing:
        raise SystemExit(f"these followers need --{', --'.join(m.replace('_', '-') for m in missing)}")
    horns = T.use_horns(args.horns)
    import torch  # noqa: F401  (recorded below)
    record = {"mode": args.mode, "followers": {k: s.record() for k, s in followers.items()}, "horns": horns,
              "resources": asdict(resources),
              "split": args.split, "protocols": sorted(args.protocols), "takes": sorted(args.takes),
              "samba": str(args.samba.resolve()), **commit(), "started": datetime.now(timezone.utc).isoformat(),
              "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
              "torch": torch.__version__, "steps": {"update_s": T.STEP, "check_every_data_frames": T.CHECK_EVERY}}
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    check_record(args.out / "run.json", record)

    takes = T.load(args.samba, args.protocols, args.split, args.takes)
    k, n = args.shard
    takes = shard(takes, k, n, args.mode)
    done = {f.stem.replace("__", "/") for f in frames.glob("*.npz") if not f.stem.endswith(".tmp")}
    takes = sorted((t for t in takes if t[0]["performance_id"] not in done), key=lambda t: -len(t[0]["times"]))
    labels = list(followers)
    print(json.dumps({"takes": len(takes), "skipped_done": len(done), "followers": labels, "workers": args.workers,
                      "shard": [k, n], "horns": horns}), flush=True)
    if not takes:
        return
    by_id = {d["performance_id"]: d for d, _ in takes}
    pending = {}
    load_model = "checkpoint" in needs(followers.values())
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(resources, load_model)) as pool:
        jobs = [pool.submit(run_one, data, take, label, followers[label], args.mode, args.samba, resources)
                for data, take in takes for label in labels]
        for count, job in enumerate(as_completed(jobs), 1):
            r = job.result()
            parts = pending.setdefault(r["performance_id"], {})
            parts[r["label"]] = r
            if len(parts) == len(labels):
                save_take(frames, by_id[r["performance_id"]], args.samba, labels, parts)
                pending.pop(r["performance_id"])
                print(json.dumps({"done_take": r["performance_id"], "tasks_done": count, "of": len(jobs)}),
                      flush=True)


if __name__ == "__main__":
    main()
