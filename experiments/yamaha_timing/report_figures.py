"""Slide figures: ground truth vs predicted beats, plus a tempo map, per dataset.

One PNG per dataset (Lakh validation, ASAP, JazzSAMBA) for one typical take:
the take whose beat and downbeat F-measure are closest to its dataset's median
(BPM condition given). Top: true vs predicted beat/downbeat events over a
30 s excerpt. Bottom: the tempo map over the whole take: tempo from the annotated beats, tempo implied by the predicted beats
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


def figure(example, title, subtitle, path, rate, excerpt=(20.0, 80.0), source=""):
    """Plain matplotlib: true vs predicted beats over an excerpt, and the tempo map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    duration = len(example["roll"]) / rate
    t0 = min(excerpt[0], max(0.0, duration - (excerpt[1] - excerpt[0])))
    t1 = min(duration, t0 + (excerpt[1] - excerpt[0]))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6.75), dpi=150,
                                      gridspec_kw={"height_ratios": [1, 2.2]})

    within = lambda v: np.asarray(v)[(np.asarray(v) >= t0) & (np.asarray(v) <= t1)]
    top.vlines(within(example["true_beats"]), 0.55, 0.85, color="k", lw=0.8, label="ground truth")
    top.vlines(within(example["true_downbeats"]), 0.55, 0.95, color="k", lw=2.2)
    top.vlines(within(example["pred_beats"]), 0.1, 0.4, color="C0", lw=0.8, label="predicted")
    top.vlines(within(example["pred_downbeats"]), 0.05, 0.4, color="C0", lw=2.2)
    top.set_yticks([0.25, 0.7], ["predicted", "ground truth"])
    top.set_ylim(0, 1)
    top.set_xlim(t0, t1)
    top.set_xlabel("time (s)")
    top.set_title("Beats (thin) and downbeats (thick), 60 s excerpt")

    tt, bt = tempo_curve(np.asarray(example["true_beats"]))
    tp, bp = tempo_curve(np.asarray(example["pred_beats"]))
    bottom.plot(tt, bt, color="k", lw=1.4, zorder=3, label="ground truth (annotated beats)")
    bottom.plot(tp, bp, color="C0", lw=1.0, label="predicted (from predicted beats)")
    if example.get("bpm_condition") is not None:
        condition = np.asarray(example["bpm_condition"], dtype=float)
        bottom.plot(np.arange(len(condition)) / rate, condition, color="C1", ls="--", lw=1.2,
                    label="BPM condition given to the network")
    bottom.axvspan(t0, t1, color="0.9", zorder=0)
    ref = np.median(bt) if len(bt) else 120
    bottom.set_ylim(max(0, ref * 0.4), ref * 1.8)
    bottom.set_xlim(0, duration)
    bottom.set_xlabel("time (s)  (shaded: the excerpt above)")
    bottom.set_ylabel("tempo (BPM)")
    bottom.set_title("Tempo map")
    bottom.legend(loc="upper right", fontsize=9)
    bottom.grid(alpha=0.3)

    s = example["scores"]
    fig.suptitle(f"{title}\nExample: {subtitle}: beat F1 {s['beat']['f_measure']:.2f}, "
                 f"downbeat F1 {s['downbeat']['f_measure']:.2f}\nInput: {source}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lakh", type=Path, default=Path("data/lakh_v2"))
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--samba-take", default="52-bernies_tune/alternate")
    parser.add_argument("--asap-label", default="Haydn, Keyboard Sonata Hob. XVI:32, mvt. 1")
    parser.add_argument("--samba-label", default="Bernie's Tune, alternate take (live band, steady swing)")
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
    figure(ex, "Lakh validation (717 held-out songs): beat F1 0.81, downbeat F1 0.61",
           "a synthetic performance of a held-out song", args.out / "1_lakh_validation.png", rate,
           source="a random combination of the song's pitched tracks (no drums), merged into one piano roll")
    print("lakh song", pack.ids[index] if hasattr(pack, "ids") else index, ex["scores"]["beat"]["f_measure"],
          ex["scores"]["downbeat"]["f_measure"])

    max_note = SynthesisConfig().max_note_seconds
    for name, takes, wanted, title, source, path in (
            ("asap", asap_takes(args.asap), args.asap_label,
             "ASAP (1,067 classical piano performances): beat F1 0.58, downbeat F1 0.25",
             "solo piano performance MIDI", "2_asap.png"),
            ("samba", [t for t in samba_takes(args.samba) if t.name == args.samba_take], args.samba_label,
             "JazzSAMBA (152 jazz piano takes): beat F1 0.52, downbeat F1 0.33",
             "the piano part only (recorded piano MIDI); the rest of the band is not given to the model",
             "3_samba.png")):
        take = takes[0]
        _, example = run_take(model, take, take.bpm, torch.device("cpu"), max_note)
        rows = int(np.ceil(take.span[1] * rate)) + 1
        example["bpm_condition"] = np.full(rows, take.bpm)
        figure(example, title, wanted, args.out / path, rate, source=source)
        print(name, take.name, example["scores"]["beat"]["f_measure"], example["scores"]["downbeat"]["f_measure"])


if __name__ == "__main__":
    main()
