"""Harmony plug-in: a chord model listens to the piano; the cue reads, at each place, the chord written there.

A chord model is anything that gives, at each update time, probabilities over some chord classes, and says which
class each chord of the take's plan is. ChordCue does the rest (the chord written at each place, the floor, the log).
A new chord model only has to fill ChordCue's arrays; ``b5_harmony`` is the one B5 uses.
"""
from __future__ import annotations

import numpy as np


class ChordCue:
    name = "harmony"

    def __init__(self, times, probabilities, available, chord_ids, chord_starts_beats, chord_ends_beats,
                 floor_mix=.02):
        """times          the update times the model was run at (each step of the follower must be one of them)
        probabilities  (len(times), C) chord-class probabilities; rows are normalised here
        available      (len(times),) False where the model heard nothing (the cue then has no opinion)
        chord_ids      the class of each chord of the plan (-1: no class)
        chord_*_beats  where each chord of the plan starts and ends, in beats on the unfolded chart
        floor_mix      ours (Stage A's probability_floor_mix): every class keeps floor_mix / C"""
        self.times = np.asarray(times, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        self.probability = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), np.finfo(float).tiny)
        self.available = np.asarray(available, dtype=bool)
        self.ids = np.asarray(chord_ids)
        self.starts, self.ends = np.asarray(chord_starts_beats, dtype=float), np.asarray(chord_ends_beats, dtype=float)
        self.floor_mix = floor_mix
        self._positions, self._chords = None, None

    def chords_at(self, positions):
        """The class of the chord written at each place (-1: none written there)."""
        if positions is not self._positions:  # the follower keeps one grid array until it grows
            interval = np.searchsorted(self.starts, positions, side="right") - 1
            valid = (interval >= 0) & (positions < self.ends[np.maximum(interval, 0)])
            self._chords = np.full(len(positions), -1, dtype=int)
            self._chords[valid] = self.ids[interval[valid]]
            self._positions = positions
        return self._chords

    def log_likelihood(self, now, bpm, positions):
        frame = int(np.searchsorted(self.times, now - 1e-6))
        if frame == len(self.times) or abs(self.times[frame] - now) > 1e-6:
            raise ValueError(f"the chord model was not run at {now:.3f} s")
        if not self.available[frame]:
            return None
        n_classes = self.probability.shape[1]
        smooth = (1 - self.floor_mix) * self.probability[frame] + self.floor_mix / n_classes
        chords = self.chords_at(positions)
        known = chords >= 0
        observation = np.full(len(positions), 1. / n_classes)
        observation[known] = smooth[chords[known]]
        return np.log(observation)


def b5_harmony(data, times, floor_mix=.02):
    """B5's chord model (Maezawa 2023, as the ensemble benchmark built it): the piano notes sounding at each time
    against triad templates of the song's chords, z-scored per frame and softmaxed (run_benchmark.score_probabilities).
    ``data`` is a follower_data take; ``times`` the follower's update times."""
    from .. import legacy  # noqa: F401  (the benchmark folders on sys.path)
    import run_study
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    return ChordCue(times, probabilities, available, case["targets"]["ids"], data["chord_starts_beats"],
                    data["chord_ends_beats"], floor_mix)
