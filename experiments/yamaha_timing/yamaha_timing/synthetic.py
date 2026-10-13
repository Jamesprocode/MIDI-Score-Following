"""Toy periodic keyboard activity for implementation checks, not a benchmark.

Each beat has a distinct pitch; the downbeat also has a bass note. This makes
the example deliberately easy. Success checks that the training path works,
not that the model follows expressive music or generalizes to jazz.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .config import TimingConfig


def periodic_roll(config: TimingConfig, frames: int, bpm: float = 120.0,
                  transpose: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return binary [T,88] activity and beat/bar labels on a four-beat cycle."""
    if frames < 1 or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("frames and BPM must be positive")
    times = np.arange(frames, dtype=np.float64) / config.frame_rate
    beat_position = times * bpm / 60.0
    # Roundoff at an exact grid beat must not put the frame in the preceding bin.
    beat_position = np.round(beat_position, 10)
    beat_number = np.floor(beat_position).astype(int)
    beat_phase = beat_position - beat_number
    bar_phase = np.mod(beat_position, 4.0) / 4.0
    roll = np.zeros((frames, config.input_pitches), dtype=np.float32)
    for frame, (number, phase) in enumerate(zip(beat_number, beat_phase)):
        if phase < 0.45:
            pitch = (60, 64, 67, 71)[number % 4] + transpose
            if config.midi_low <= pitch < config.midi_low + config.input_pitches:
                roll[frame, pitch - config.midi_low] = 1.0
            if number % 4 == 0:
                bass = 48 + transpose
                if config.midi_low <= bass < config.midi_low + config.input_pitches:
                    roll[frame, bass - config.midi_low] = 1.0
    beat_labels = np.minimum(np.floor(beat_phase * config.beat_bins).astype(np.int64), config.beat_bins - 1)
    bar_labels = np.minimum(np.floor(bar_phase * config.bar_bins).astype(np.int64), config.bar_bins - 1)
    return roll, beat_labels, bar_labels


def overfit_windows(config: TimingConfig, bpm: float = 120.0) -> dict[str, torch.Tensor]:
    """One full bar of window endpoints, supervising only each newest frame.

    Every window contains exactly the past context used by streaming inference.
    Earlier outputs are masked instead of treating partial windows as full
    contexts. The BPM is an explicitly supplied toy setting, not an estimator.
    """
    frames_per_bar = int(round(config.frame_rate * 60.0 / bpm * 4))
    count = max(frames_per_bar, config.bar_bins)
    roll, beat_labels, bar_labels = periodic_roll(config, config.context_frames + count, bpm)
    windows, beat_targets, bar_targets = [], [], []
    for end in range(config.context_frames, config.context_frames + count):
        start = end - config.context_frames + 1
        windows.append(roll[start:end + 1])
        beat_targets.append(beat_labels[start:end + 1])
        bar_targets.append(bar_labels[start:end + 1])
    mask = torch.zeros((count, config.context_frames), dtype=torch.bool)
    mask[:, -1] = True
    return {
        "piano_roll": torch.from_numpy(np.stack(windows)),
        "beat_targets": torch.from_numpy(np.stack(beat_targets)),
        "bar_targets": torch.from_numpy(np.stack(bar_targets)),
        "beat_mask": mask.clone(), "bar_mask": mask.clone(),
        "bpm_hint": torch.full((count,), float(bpm)),
    }
