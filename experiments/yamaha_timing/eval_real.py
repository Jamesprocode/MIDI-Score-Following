"""Score a timing checkpoint on real piano performances (JazzSAMBA or ASAP).

Each take runs as one causal pass with the silent lead-in, the same as live
streaming, twice: with the BPM condition given (SAMBA: the written tempo;
ASAP: the median performed tempo, an oracle) and withheld. The input roll is
held keys at 20 fps with notes capped at SynthesisConfig.max_note_seconds, as in
training. Only the annotated span is scored: decoded events outside it are
dropped and frames outside it are not counted. Per take: mir_eval beat and
downbeat measures from the same causal decoder as validation, and per-frame
beat/bar phase accuracy and loss against the phase implied by the annotations.
Data details and the SAMBA sync timing repair: yamaha_timing/real_data.py.

Writes rows.csv (one row per take and condition), summary.json and summary.md
(means per subset), takes.csv (per-take data notes) and figures/ to --out
(default $RUN_DIR), and logs the summary, the per-take table and the figures
to W&B (project midi-score-following; WANDB_MODE=disabled for local runs).

  python eval_real.py samba --data DIR --checkpoint runs/lakh_r3b/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from yamaha_timing.evaluation import decode_events, expected_phase, tracking_metrics
from yamaha_timing.lakh import SynthesisConfig, rasterize
from yamaha_timing.plots import plot_tracking
from yamaha_timing.real_data import asap_takes, samba_takes
from yamaha_timing.training import load_checkpoint

HEADLINE = ("beat_f_measure", "beat_cmlt", "beat_amlt", "downbeat_f_measure", "downbeat_cmlt", "downbeat_amlt",
            "bar_acc", "beat_acc")


def phase_labels(times, events, bins, end=None):
    """Phase bin of each time within its event interval; -1 outside the events."""
    edges = np.append(events, end) if end is not None else np.asarray(events)
    i = np.searchsorted(edges, times, side="right") - 1
    ok = (i >= 0) & (i < len(edges) - 1)
    labels = np.full(len(times), -1)
    j = i[ok]
    phase = (times[ok] - edges[j]) / (edges[j + 1] - edges[j])
    labels[ok] = np.minimum((phase * bins).astype(int), bins - 1)
    return labels


def frame_scores(log_probs, labels):
    keep = labels >= 0
    if not keep.any():
        return {"acc": float("nan"), "acc_within1": float("nan"), "loss": float("nan")}
    lp, y = log_probs[keep], labels[keep]
    bins = lp.shape[1]
    error = np.mod(lp.argmax(1) - y, bins)
    return {"acc": float(np.mean(error == 0)), "acc_within1": float(np.mean(np.minimum(error, bins - error) <= 1)),
            "loss": float(-lp[np.arange(len(y)), y].mean())}


@torch.no_grad()
def run_take(model, take, bpm, device, max_note_seconds):
    cfg = model.config
    rate, lead = cfg.frame_rate, cfg.context_frames - 1
    end = max(take.span[1], float(take.offsets.max()) if len(take.offsets) else 0.0)
    frames = int(np.ceil(end * rate)) + 1
    offsets = np.minimum(take.offsets, take.onsets + max_note_seconds)
    roll = np.concatenate([np.zeros((lead, cfg.input_pitches), np.uint8),
                           rasterize(take.onsets, offsets, take.pitches, frames, rate)])
    x = torch.from_numpy(roll).float()[None].to(device)
    hint = torch.full((1, len(roll)), float(bpm or 120.0), device=device)
    valid = torch.zeros((1, len(roll)), dtype=torch.bool, device=device)
    if bpm is not None:
        valid[:, lead:] = True  # the lead-in never carries a hint, as in training
    out = model(x, bpm_hint=hint, hint_valid=valid)
    log_probs = {task: torch.log_softmax(out[f"{task}_logits"][0, lead:].float(), -1).cpu().numpy()
                 for task in ("beat", "bar")}
    times = np.arange(frames) / rate
    in_span = (times >= take.span[0]) & (times <= take.span[1])
    within = lambda events: events[(events >= take.span[0]) & (events <= take.span[1])]
    result, decoded = {}, {}
    for task, event, reference in (("beat", "beat", take.beats), ("bar", "downbeat", take.downbeats)):
        probabilities = np.exp(log_probs[task])
        estimated = within(decode_events(expected_phase(probabilities), rate))
        scores = tracking_metrics(within(reference), estimated)
        result.update({f"{event}_{k}": v for k, v in scores.items()})
        bins = probabilities.shape[1]
        labels = phase_labels(times, reference, bins, end=take.span[1])
        labels[~in_span] = -1
        result.update({f"{task}_{k}": v for k, v in frame_scores(log_probs[task], labels).items()})
        decoded[task] = (probabilities, estimated, scores, labels)
    example = {"roll": roll[lead:], "beat_probs": decoded["beat"][0], "bar_probs": decoded["bar"][0],
               "beat_labels": decoded["beat"][3], "bar_labels": decoded["bar"][3],
               "true_beats": within(take.beats), "true_downbeats": within(take.downbeats),
               "pred_beats": decoded["beat"][1], "pred_downbeats": decoded["bar"][1],
               "scores": {"beat": decoded["beat"][2], "downbeat": decoded["bar"][2]}}
    return result, example


def subsets(dataset, info):
    """Summary groups a take belongs to."""
    if dataset == "samba":
        name = "async" if info["protocol"] == "async" else "sync_repaired"
        groups = [name]
        if info["split"] == "test":
            groups.append(f"{name}_test")
        return groups
    return ["all", info["meter_class"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("samba", "asap"))
    parser.add_argument("--data", type=Path, required=True, help="SAMBA root (catalog + *_audio) or ASAP clone")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt from train_lakh.py")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/eval")))
    parser.add_argument("--figures", type=int, default=4, help="takes drawn (fixed seed) for figures, BPM given")
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    model, provenance = load_checkpoint(args.checkpoint, "cpu")  # float64 buffers; then move
    model = model.to(device).eval()
    start = time.time()
    takes = samba_takes(args.data) if args.dataset == "samba" else asap_takes(args.data)
    if args.limit:
        takes = takes[:args.limit]
    print(json.dumps({"dataset": args.dataset, "takes": len(takes), "load_s": round(time.time() - start),
                      "checkpoint": str(args.checkpoint), "epoch": provenance.get("epoch"), "device": str(device)}),
          flush=True)
    with open(args.out / "takes.csv", "w", newline="") as handle:
        keys = sorted({k for take in takes for k in take.info})
        writer = csv.DictWriter(handle, fieldnames=["name", "notes", "span_start", "span_end", "bpm", *keys])
        writer.writeheader()
        for take in takes:
            writer.writerow({"name": take.name, "notes": len(take.onsets), "span_start": take.span[0],
                             "span_end": take.span[1], "bpm": take.bpm, **take.info})

    import wandb
    wandb.init(project="midi-score-following", job_type="eval", config={
        "dataset": args.dataset, "checkpoint": str(args.checkpoint), "checkpoint_epoch": provenance.get("epoch"),
        "checkpoint_validation_loss": (provenance.get("validation") or {}).get("loss"), "takes": len(takes)})
    figure_picks = set(np.random.default_rng(0).choice(len(takes), size=min(args.figures, len(takes)), replace=False))
    max_note = SynthesisConfig().max_note_seconds
    rows, images = [], []
    for k, take in enumerate(takes):
        for condition, bpm in (("given", take.bpm), ("withheld", None)):
            scores, example = run_take(model, take, bpm, device, max_note)
            rows.append({"name": take.name, "hint": condition, "groups": "|".join(subsets(args.dataset, take.info)),
                         **{key: take.info.get(key, "") for key in ("protocol", "split", "meter", "meter_class",
                                                                   "composer", "timing")},
                         **{key: round(value, 5) for key, value in scores.items()}})
            if condition == "given" and k in figure_picks:
                try:
                    example["title"] = f"{args.dataset} {take.name}   BPM hint {bpm:.0f} given"
                    path = args.out / "figures" / f"{take.name.replace('/', '__')}.png"
                    plot_tracking(example, path, model.config.frame_rate)
                    images.append(wandb.Image(str(path), caption=take.name))
                except Exception as exc:  # figures must never stop the evaluation
                    print(json.dumps({"plot_failed": take.name, "error": repr(exc)}), flush=True)
        if (k + 1) % 50 == 0:
            print(json.dumps({"done": k + 1, "of": len(takes), "elapsed_s": round(time.time() - start)}), flush=True)

    metric_keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    with open(args.out / "rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for group in sorted({g for row in rows for g in row["groups"].split("|")}):
        for condition in ("given", "withheld"):
            members = [r for r in rows if group in r["groups"].split("|") and r["hint"] == condition]
            summary[f"{group}/{condition}"] = {"takes": len(members), **{
                key: float(np.nanmean([r[key] for r in members])) for key in metric_keys}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| subset / BPM hint | takes | " + " | ".join(HEADLINE) + " |", "|" + "---|" * (len(HEADLINE) + 2)]
    lines += [f"| {name} | {values['takes']} | " + " | ".join(f"{values[key]:.3f}" for key in HEADLINE) + " |"
              for name, values in summary.items()]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    wandb.log({**{f"{args.dataset}/{name}/{key}": values[key] for name, values in summary.items() for key in HEADLINE},
               f"{args.dataset}/takes": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows]),
               **({f"{args.dataset}/figures": images} if images else {})})
    wandb.summary.update({f"{args.dataset}/{name}/{key}": values[key]
                          for name, values in summary.items() for key in HEADLINE})
    wandb.finish()
    (args.out / "DONE").touch()
    print(json.dumps({"finished": True, "elapsed_s": round(time.time() - start)}), flush=True)


if __name__ == "__main__":
    main()
