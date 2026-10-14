"""Slide figures: ground truth vs predicted beats, plus a tempo map, per dataset.

One PNG per dataset (Lakh validation, ASAP, JazzSAMBA) for one typical take:
the take whose beat and downbeat F-measure are closest to its dataset's median
(BPM condition given). Rows: the input piano roll and the true vs predicted
beat/downbeat events over a 30 s excerpt, then the tempo map over the whole
take: tempo from the annotated beats, tempo implied by the predicted beats
(the network outputs phase, not tempo), and the BPM condition. Predictions are
the causal decoder's, as in validation and eval_real.py.

  python report_figures.py --checkpoint best_model.pt --samba DIR --asap DIR --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eval_real import run_take
from train_lakh import FIXED_EPOCH
from yamaha_timing.evaluation import decode_events, expected_phase, tracking_metrics
from yamaha_timing.lakh import SongPack, SynthesisConfig, render_example
from yamaha_timing.real_data import asap_takes, samba_takes
from yamaha_timing.training import load_checkpoint

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TRUTH, PRED, CONDITION = "#52514e", "#2a78d6", "#eb6834"


@torch.no_grad()
def lakh_example(model, pack, index):
    """A validation performance (fixed augmentation, BPM condition given)."""
    rng = np.random.default_rng([0, FIXED_EPOCH, index])
    ex = render_example(pack.song(index), rng, model.config, SynthesisConfig())
    lead = model.config.context_frames - 1
    valid = np.zeros(len(ex["roll"]), dtype=bool)
    valid[lead:] = True
    out = model(torch.from_numpy(ex["roll"]).float()[None], bpm_hint=torch.from_numpy(ex["bpm"])[None],
                hint_valid=torch.from_numpy(valid)[None])
    probs = {t: torch.softmax(out[f"{t}_logits"][0, lead:], -1).numpy() for t in ("beat", "bar")}
    rate = model.config.frame_rate
    pred = {t: decode_events(expected_phase(probs[t]), rate) for t in probs}
    scores = {"beat": tracking_metrics(ex["beat_times"], pred["beat"]),
              "downbeat": tracking_metrics(ex["downbeat_times"], pred["bar"])}
    return {"roll": ex["roll"][lead:], "true_beats": ex["beat_times"], "true_downbeats": ex["downbeat_times"],
            "pred_beats": pred["beat"], "pred_downbeats": pred["bar"], "scores": scores,
            "bpm_condition": ex["bpm"][lead:]}


def tempo_curve(beats, smooth=4):
    """(times, BPM) from beat times, median over `smooth` intervals."""
    if len(beats) < smooth + 2:
        return np.array([]), np.array([])
    ioi = np.diff(beats)
    bpm = 60.0 / np.array([np.median(ioi[max(0, i - smooth + 1):i + 1]) for i in range(len(ioi))])
    return beats[1:], bpm


def figure(example, title, subtitle, path, rate, excerpt=(20.0, 50.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    duration = len(example["roll"]) / rate
    t0 = min(excerpt[0], max(0.0, duration - (excerpt[1] - excerpt[0])))
    t1 = min(duration, t0 + (excerpt[1] - excerpt[0]))
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURFACE)
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 2.3], hspace=0.58, left=0.1, right=0.98, top=0.86, bottom=0.08)

    ax = fig.add_subplot(grid[0])
    k0, k1 = int(t0 * rate), int(t1 * rate)
    roll = example["roll"][k0:k1]
    keys = np.flatnonzero(roll.any(axis=0))
    lo, hi = (keys.min(), keys.max() + 1) if len(keys) else (30, 60)
    ax.imshow(roll[:, lo:hi].T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest",
              extent=[t0, t1, lo + 21, hi + 21], vmin=0, vmax=1.6)
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Input: piano roll (30 s excerpt)", loc="left", fontsize=15, color=MUTED)
    ax.set_xlim(t0, t1)
    ax.set_facecolor(SURFACE)

    ax = fig.add_subplot(grid[1], sharex=fig.axes[0])
    within = lambda v: v[(v >= t0) & (v <= t1)]
    ax.vlines(within(example["true_beats"]), 0.58, 0.88, color=TRUTH, lw=1.2)
    ax.vlines(within(example["true_downbeats"]), 0.58, 1.0, color=TRUTH, lw=3.2)
    ax.vlines(within(example["pred_beats"]), 0.12, 0.42, color=PRED, lw=1.2)
    ax.vlines(within(example["pred_downbeats"]), 0.0, 0.42, color=PRED, lw=3.2)
    ax.set_yticks([0.25, 0.75], ["predicted", "ground truth"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds   (thin = beats, thick = downbeats)")
    ax.set_title("Beats and downbeats", loc="left", fontsize=15, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax = fig.add_subplot(grid[2])
    tt, bt = tempo_curve(np.asarray(example["true_beats"]))
    tp, bp = tempo_curve(np.asarray(example["pred_beats"]))
    ax.axvspan(t0, t1, color=GRID, alpha=0.6, lw=0)
    if example.get("bpm_condition") is not None:
        condition = np.asarray(example["bpm_condition"], dtype=float)
        ax.plot(np.arange(len(condition)) / rate, condition, color=CONDITION, lw=2, ls=(0, (5, 4)),
                label="BPM condition given to the network")
    ax.plot(tp, bp, color=PRED, lw=2, label="tempo implied by predicted beats")
    ax.plot(tt, bt, color=TRUTH, lw=2.4, label="ground-truth tempo (annotated beats)")
    ref = np.median(bt) if len(bt) else 120
    ax.set_ylim(max(0, ref * 0.4), ref * 1.8)
    ax.set_xlim(0, duration)
    ax.set_ylabel("beats per minute")
    ax.set_xlabel("seconds   (whole take; shaded = excerpt above)")
    ax.set_title("Tempo map", loc="left", fontsize=15, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_facecolor(SURFACE)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=13, ncol=3, borderaxespad=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    s = example["scores"]
    fig.text(0.1, 0.945, title, fontsize=23, color=INK, weight="bold")
    fig.text(0.1, 0.905, f"Example: {subtitle}   ·   this take: beat F1 {s['beat']['f_measure']:.2f}, "
                         f"downbeat F1 {s['downbeat']['f_measure']:.2f}", fontsize=16, color=MUTED)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lakh", type=Path, default=Path("data/lakh_v2"))
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--samba-take", default="65-misty/alternate")
    parser.add_argument("--asap-label", default="Haydn, Keyboard Sonata Hob. XVI:32, mvt. 1")
    parser.add_argument("--samba-label", default="Misty, alternate take (live band)")
    parser.add_argument("--asap", type=Path, required=True)
    parser.add_argument("--lakh-songs", type=int, default=80, help="validation songs scanned for a median one")
    parser.add_argument("--out", type=Path, default=Path("report_figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    model.eval()
    rate = model.config.frame_rate

    pack = SongPack(args.lakh / "val.npz")
    group = np.asarray(pack.meta["group"])
    firsts = [int(np.flatnonzero(group == g)[0]) for g in np.unique(group)][:args.lakh_songs]
    examples = [(i, lakh_example(model, pack, i)) for i in firsts]
    mb = np.median([e["scores"]["beat"]["f_measure"] for _, e in examples])
    md = np.median([e["scores"]["downbeat"]["f_measure"] for _, e in examples])
    index, ex = min(examples, key=lambda p: abs(p[1]["scores"]["beat"]["f_measure"] - mb)
                    + abs(p[1]["scores"]["downbeat"]["f_measure"] - md))
    figure(ex, "Lakh validation  ·  717 held-out songs  ·  beat F1 0.81  ·  downbeat F1 0.61",
           "a synthetic performance of a held-out song", args.out / "1_lakh_validation.png", rate)
    print("lakh song", pack.ids[index] if hasattr(pack, "ids") else index, ex["scores"]["beat"]["f_measure"],
          ex["scores"]["downbeat"]["f_measure"])

    max_note = SynthesisConfig().max_note_seconds
    for name, takes, wanted, title, subtitle, path in (
            ("asap", asap_takes(args.asap), args.asap_label,
             "ASAP  ·  1,067 classical performances  ·  beat F1 0.58  ·  downbeat F1 0.25", None, "2_asap.png"),
            ("samba", [t for t in samba_takes(args.samba) if t.name == args.samba_take], args.samba_label,
             "JazzSAMBA  ·  152 jazz piano takes  ·  beat F1 0.52  ·  downbeat F1 0.33", None, "3_samba.png")):
        take = takes[0]
        _, example = run_take(model, take, take.bpm, torch.device("cpu"), max_note)
        rows = int(np.ceil(take.span[1] * rate)) + 1
        example["bpm_condition"] = np.full(rows, take.bpm)
        figure(example, title, wanted, args.out / path, rate)
        print(name, take.name, example["scores"]["beat"]["f_measure"], example["scores"]["downbeat"]["f_measure"])


if __name__ == "__main__":
    main()
