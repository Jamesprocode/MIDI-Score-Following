"""Does bounding the tempo remove the obvious tempo spikes of the decoded beats?

Advisor's suggestion (2026-10-18): the spikes in our tempo maps (two beats a
fraction of a second apart, or a missed beat) could be removed by bounding the
highest and lowest tempo. Our choice, not in Maezawa (2023), whose follower uses
the phase posteriors and never decodes beats.

The network runs once per JazzSAMBA take for each BPM condition (written tempo,
and played tempo = median annotated tempo, the paper's "player states the
tempo" assumption). Its beat and bar phase posteriors are decoded by:

  current      evaluation.decode_events: an event when the expected phase wraps,
               re-armed after the phase visits mid-cycle (what all reported
               numbers use)
  min-gap      the same events, but an event closer than period / r to the last
               kept event is dropped: the tempo can be at most r x the condition
  min-gap+fill also, when no event comes for period * r, one is added at the
               last kept event plus the running period: the tempo can be at
               least condition / r. Decided only once the gap is too long, so an
               added beat is placed up to (r - 1) periods in the past: fine for
               scoring, but a live system would hear it late.

period = 60 / condition for beats, times the beats per bar for downbeats. The
running period starts at that value and follows kept intervals (halfway per
event), clipped to the bounds. Scores: mir_eval F-measure and CMLt (+-70 ms), and
the share of predicted beat intervals more than 1.5x off the annotated local
tempo ("spikes").

  python eval_tempo_bounds.py --data "../../Dataset/SAMBA dataset" --checkpoint best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch

from eval_real import run_take
from yamaha_timing.evaluation import decode_events, expected_phase
from yamaha_timing.lakh import SynthesisConfig
from yamaha_timing.real_data import samba_takes
from yamaha_timing.training import load_checkpoint

RATIOS = (1.2, 1.3, 1.5, 2.0)


def decode_bounded(phase, rate, period, ratio, fill):
    """Events from decode_events with every kept interval within [period / ratio, period * ratio].

    An event too soon after the last kept one is dropped. With ``fill``, a gap
    longer than period * ratio first gets beats at the running period; an event
    then too close to an added beat is dropped (the added beat stays), so the
    bounds hold for every interval."""
    low, high = period / ratio, period * ratio
    kept, current = [], period
    for t in decode_events(phase, rate):
        if kept:
            if fill:
                while t - kept[-1] > high:
                    kept.append(kept[-1] + current)
            gap = t - kept[-1]
            if gap < low:
                continue
            current = float(np.clip(.5 * current + .5 * gap, low, high))
        kept.append(t)
    return np.asarray(kept)


def f_and_cmlt(reference, estimated):
    """mir_eval beat F-measure and CMLt with mir_eval.beat.evaluate's 5 s trim; the
    same values as evaluation.tracking_metrics, without the measures we don't report."""
    import warnings

    import mir_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = mir_eval.beat.trim_beats(np.sort(np.asarray(reference, dtype=float)))
        estimated = mir_eval.beat.trim_beats(np.sort(np.asarray(estimated, dtype=float)))
        f = mir_eval.beat.f_measure(reference, estimated)
        _, cmlt, _, _ = mir_eval.beat.continuity(reference, estimated)
    return {"f_measure": float(np.nan_to_num(f)), "cmlt": float(np.nan_to_num(cmlt))}


def spike_share(estimated, reference, factor=1.5):
    """Share of predicted inter-event intervals more than `factor` off the annotated local interval."""
    if len(estimated) < 2 or len(reference) < 2:
        return float("nan")
    gaps = np.diff(estimated)
    middle = estimated[:-1] + gaps / 2
    local = np.interp(middle, reference[1:] - np.diff(reference) / 2, np.diff(reference))
    ratio = gaps / local
    return float(np.mean((ratio > factor) | (ratio < 1 / factor)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/tempo_bounds"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = load_checkpoint(args.checkpoint, "cpu")[0].eval()
    max_note = SynthesisConfig().max_note_seconds
    rate = model.config.frame_rate
    takes = [t for t in samba_takes(args.data) if not t.info.get("excluded")]
    rows, cache = [], {}
    for k, take in enumerate(takes, 1):
        played = float(60 / np.median(np.diff(take.beats)))
        per_bar = int(round(np.median(np.diff(take.downbeats)) / np.median(np.diff(take.beats))))
        for condition, bpm in (("written", float(take.bpm)), ("played", played)):
            with torch.no_grad():
                _, example = run_take(model, take, bpm, "cpu", max_note)
            within = lambda e: e[(e >= take.span[0]) & (e <= take.span[1])]
            phases = {"beat": expected_phase(example["beat_probs"]), "bar": expected_phase(example["bar_probs"])}
            reference = {"beat": example["true_beats"], "bar": example["true_downbeats"]}
            period = {"beat": 60 / bpm, "bar": 60 / bpm * per_bar}
            decoders = [("current", None, False)] + [(f"min-gap r={r:g}", r, False) for r in RATIOS] + \
                       [(f"min-gap+fill r={r:g}", r, True) for r in RATIOS]
            for name, ratio, fill in decoders:
                row = {"take": take.name, "protocol": take.info["protocol"], "condition": condition, "decoder": name,
                       "written_bpm": float(take.bpm), "played_bpm": played, "beats_per_bar": per_bar}
                for task, event in (("beat", "beat"), ("bar", "downbeat")):
                    estimated = decode_events(phases[task], rate) if ratio is None else \
                        decode_bounded(phases[task], rate, period[task], ratio, fill)
                    estimated = within(estimated)
                    scores = f_and_cmlt(reference[task], estimated)
                    row[f"{event}_f"] = scores["f_measure"]
                    row[f"{event}_cmlt"] = scores["cmlt"]
                    row[f"{event}_spikes"] = spike_share(estimated, reference[task])
                    row[f"{event}_count_ratio"] = len(estimated) / max(len(reference[task]), 1)
                    if name in ("current", "min-gap+fill r=1.3", "min-gap+fill r=1.5") and condition == "played":
                        cache[(take.name, name, task)] = estimated
                rows.append(row)
        print(json.dumps({"done": k, "of": len(takes), "take": take.name}), flush=True)
    with open(args.out / "per_take.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out / "examples.npz", **{"|".join(key): value for key, value in cache.items()})
    keys = ["beat_f", "beat_cmlt", "beat_spikes", "beat_count_ratio", "downbeat_f", "downbeat_cmlt", "downbeat_spikes"]
    lines = ["| subset | condition | decoder | takes | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
    for subset in ("all", "async", "sync"):
        for condition in ("played", "written"):
            for name in dict.fromkeys(r["decoder"] for r in rows):
                members = [r for r in rows if r["condition"] == condition and r["decoder"] == name
                           and subset in ("all", r["protocol"])]
                values = [np.nanmean([r[c] for r in members]) for c in keys]
                lines.append(f"| {subset} | {condition} | {name} | {len(members)} | "
                             + " | ".join(f"{v:.3f}" for v in values) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
