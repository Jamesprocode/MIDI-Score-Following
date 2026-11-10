"""Rhythm plug-in: a tracker listens to the piano's rhythm; the cue reads, at each place, the place's phase.

A tracker is anything with ``log_probs(now, bpm)``: log-probabilities over K equal phase classes of a period, from
input up to ``now``, for the BPM hint ``bpm``. The period is a bar for a downbeat tracker (the network: 32 classes per
bar) or a beat for a beat tracker. PhaseCue maps each place to its phase class in that period.
"""
from __future__ import annotations

import numpy as np


class PhaseCue:
    name = "rhythm"

    def __init__(self, tracker, period_beats):
        self.tracker, self.period_beats = tracker, float(period_beats)

    def log_likelihood(self, now, bpm, positions):
        log_p = self.tracker.log_probs(now, bpm)
        phase = np.mod(positions, self.period_beats) / self.period_beats
        return log_p[np.minimum((phase * len(log_p)).astype(int), len(log_p) - 1)]


class NetworkTracker:
    """The Lakh-trained timing network's bar phase (yamaha_follower/evidence.NetworkPhase: the backbone runs once per
    take, the BPM gate and the bar head at each call)."""

    def __init__(self, network_phase):
        self.network = network_phase

    def log_probs(self, now, bpm):
        return self.network.bar_log_probs(self.network.frame_at(now), bpm)


def network_downbeat(model, take, beats_per_bar):
    """B5's rhythm cue: the network on the take's piano, one period per bar. ``model``: a loaded checkpoint
    (yamaha_timing.training.load_checkpoint)."""
    from .. import legacy  # noqa: F401  (yamaha_follower on sys.path)
    from evidence import NetworkPhase
    return PhaseCue(NetworkTracker(NetworkPhase(model, take)), beats_per_bar)
