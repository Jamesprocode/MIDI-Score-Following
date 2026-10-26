"""The B7 follower with cue weights that can change at every update: the hook for adaptive weighting.

yamaha_follower/yamaha_filter.follow combines its two cues as

    log phi_t(s) = elapsed * (w_h * log H_t(s) + w_d * log D_t(s))

with w_h = emission_power_per_second and w_d = downbeat_power_per_second fixed
for the whole take. ``follow_weighted`` is the same filter, line for line,
except that the weights come from a per-update array ``weights[t] = (w_h, w_d)``.
With a constant array it gives identical output (tests/test_weighted_filter.py
checks this). Anything that produces weights causally (a rule, or a small
network reading the music) can drive it; a schedule from annotated passage
types is an oracle, used only to show what moving the faders could do.

An optional third cue, ``melody`` (an object with ``log_likelihood(now, bpm,
n_positions)``, e.g. melody_evidence.MelodyEvidence), adds
``elapsed * w_m * log M_t(s)``; then ``weights[t] = (w_h, w_d, w_m)``. It reads
the same fed-back tempo as the network. Without it the filter is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "yamaha_follower"))
import follower_data  # noqa: E402,F401  (sys.path for the benchmark modules)
from adaptive_grid import _advance, _tempo_predict, bar_indices  # noqa: E402
from yamaha_filter import YamahaConfig, _beat_kernel, _beat_transition, _tempo_grid, _walk  # noqa: E402


def follow_weighted(times, chord_probabilities, available, chord_starts_beats, chord_ends_beats, chord_ids,
                    bar_starts_beats, beats_per_bar, initial_bpm, weights, *, start_time, network=None,
                    config=YamahaConfig(), melody=None):
    """yamaha_filter.follow with weights[t] = (w_harmony, w_downbeat) per update instead of the config's
    emission_power_per_second / downbeat_power_per_second; (w_harmony, w_downbeat, w_melody) with ``melody``."""
    times = np.asarray(times, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(times), 2 if melody is None else 3))
    probability = np.asarray(chord_probabilities, dtype=float)
    probability = probability / np.maximum(probability.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    starts, ends, ids = map(np.asarray, (chord_starts_beats, chord_ends_beats, chord_ids))
    bars = np.asarray(bar_starts_beats, dtype=float)
    step = 1. / config.bins_per_beat
    positions = np.arange(int(np.ceil(ends[-1] / step)) + 1, dtype=float) * step
    interval = np.searchsorted(starts, positions, side="right") - 1
    valid = (interval >= 0) & (positions < ends[np.maximum(interval, 0)])
    state_chords = np.full(len(positions), -1, dtype=int)
    state_chords[valid] = ids[interval[valid]]
    ratios = _tempo_grid(config)
    tempi = initial_bpm * ratios
    prior = np.exp(-.5 * (np.log(ratios) / config.initial_tempo_log_std) ** 2)
    prior /= prior.sum()
    posterior = np.zeros((len(tempi), len(positions)))
    posterior[:, 0] = prior
    taps, kernel = _beat_kernel(config)
    transition = _beat_transition(config, ratios) if config.beat_tempo_reversion else None
    n_classes = probability.shape[1]
    reachable, hint, previous = 0, float(initial_bpm), float(start_time)
    out = {key: np.empty(len(times)) for key in ("position_beats", "tempo_bpm", "mean_tempo_bpm", "bpm_condition",
                                                 "position_confidence")}
    for frame, now in enumerate(times):
        elapsed = now - previous
        w_h, w_d = weights[frame][:2]
        w_m = weights[frame][2] if melody is not None else 0.
        if elapsed:
            beats_moved = tempi * elapsed / 60.
            reachable += int(np.ceil(beats_moved.max() / step))
            if reachable + 2 > len(positions):
                pad = max(reachable + 2, len(positions) + 256) - len(positions)
                posterior = np.pad(posterior, ((0, 0), (0, pad)))
                state_chords = np.pad(state_chords, (0, pad), constant_values=-1)
                positions = np.arange(len(positions) + pad, dtype=float) * step
            if config.tempo_model == "continuous":
                posterior = _tempo_predict(posterior, elapsed * config.continuous_rate_per_second, prior)
            posterior = _advance(posterior, beats_moved / step)
            if config.tempo_model == "beat":
                fraction = np.mod(positions + 1e-9, 1.0)
                crossed = posterior * (fraction[None, :] < beats_moved[:, None] - 1e-9)
                moved = _walk(crossed, taps, kernel) if transition is None else transition @ crossed
                posterior = posterior - crossed + moved
            known = state_chords >= 0
            log_weight = np.zeros(len(positions))
            if available[frame] and w_h:
                smooth = (1 - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
                observation = np.full(len(positions), 1. / n_classes)
                observation[known] = smooth[state_chords[known]]
                log_h = np.log(observation)
                log_weight += (log_h - log_h.max()) * elapsed * w_h
            if network is not None and w_d:
                log_d = network.bar_log_probs(network.frame_at(now), hint)
                phase = np.mod(positions, beats_per_bar) / beats_per_bar
                log_pd = log_d[np.minimum((phase * len(log_d)).astype(int), len(log_d) - 1)]
                log_weight += (log_pd - log_pd.max()) * elapsed * w_d
            if melody is not None and w_m:
                log_m = melody.log_likelihood(now, hint, len(positions))
                if log_m is not None:
                    log_weight += (log_m - log_m.max()) * elapsed * w_m
            posterior *= np.exp(log_weight)[None, :]
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("follower lost all probability mass")
        posterior /= total
        position_marginal, tempo_marginal = posterior.sum(axis=0), posterior.sum(axis=1)
        best = position_marginal.argmax()
        out["position_beats"][frame] = positions[best]
        out["position_confidence"][frame] = position_marginal[best]
        out["tempo_bpm"][frame] = tempi[tempo_marginal.argmax()]
        out["mean_tempo_bpm"][frame] = tempo_marginal @ tempi
        out["bpm_condition"][frame] = hint
        if config.feedback and elapsed:
            hint += min(1., elapsed / config.feedback_time_constant) * (out["mean_tempo_bpm"][frame] - hint)
        previous = now
    out["bar_index"] = bar_indices(out["position_beats"], bars, ends[-1])
    out["config"] = asdict(config)
    return out
