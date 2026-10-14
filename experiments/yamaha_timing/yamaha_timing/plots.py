"""Validation figure: predicted versus true beats and downbeats for one song.

Four rows over a 30 s excerpt: the input piano roll; the beat-phase and
bar-phase probabilities as heat maps with the true phase overlaid; and a strip
with true events (top) against decoded predictions (bottom).
"""
from __future__ import annotations

import numpy as np


def plot_tracking(example, path, frame_rate=20, seconds=30.0, skip_seconds=10.0, low_pitch=21):
    """Save the figure for ``example`` (dict from train_lakh.validate) to ``path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = len(example["beat_labels"])
    length = int(seconds * frame_rate)
    start = min(int(skip_seconds * frame_rate), max(0, frames - length))
    stop = min(frames, start + length)
    t0, t1 = start / frame_rate, stop / frame_rate
    times = np.arange(start, stop) / frame_rate

    fig, axes = plt.subplots(4, 1, figsize=(14, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.4, 1.4, 0.9]})
    roll = example["roll"][start:stop]
    keys = np.flatnonzero(roll.any(axis=0))
    lo, hi = (keys.min(), keys.max() + 1) if len(keys) else (39, 60)
    axes[0].imshow(roll[:, lo:hi].T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest",
                   extent=[t0, t1, lo + low_pitch, hi + low_pitch])
    axes[0].set_ylabel("MIDI pitch")

    for ax, name, label in ((axes[1], "beat", "beat phase"), (axes[2], "bar", "bar phase")):
        probabilities = example[f"{name}_probs"][start:stop]
        bins = probabilities.shape[1]
        ax.imshow(probabilities.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
                  extent=[t0, t1, 0, 1], vmin=0, vmax=max(float(probabilities.max()), 1e-6))
        truth = (example[f"{name}_labels"][start:stop] + 0.5) / bins
        ax.plot(times, truth, ".", color="cyan", markersize=2.5, label="true phase")
        ax.set_ylabel(f"{label}\n(0 to 1)")
        ax.legend(loc="upper right", fontsize=8, markerscale=3)

    strip = axes[3]
    within = lambda values: values[(values >= t0) & (values <= t1)]
    strip.vlines(within(example["true_beats"]), 0.55, 0.85, color="0.55", lw=1)
    strip.vlines(within(example["true_downbeats"]), 0.55, 1.0, color="black", lw=2)
    strip.vlines(within(example["pred_beats"]), 0.15, 0.45, color="tab:blue", lw=1)
    strip.vlines(within(example["pred_downbeats"]), 0.0, 0.45, color="tab:red", lw=2)
    strip.set_yticks([0.25, 0.75], ["predicted", "true"])
    strip.set_ylim(0, 1)
    strip.set_xlabel("seconds from the song's first note  (beats thin, downbeats thick)")
    strip.set_xlim(t0, t1)

    s = example["scores"]
    fig.suptitle(f"{example['title']}\nbeat F {s['beat']['f_measure']:.2f}  CMLt {s['beat']['cmlt']:.2f}  "
                 f"AMLt {s['beat']['amlt']:.2f}   |   downbeat F {s['downbeat']['f_measure']:.2f}  "
                 f"CMLt {s['downbeat']['cmlt']:.2f}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
