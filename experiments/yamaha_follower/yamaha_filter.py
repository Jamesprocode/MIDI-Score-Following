"""Stage B: the follower as Maezawa (2023, sec. 2.2) describes it.

Differences from Stage A (improvement_study/adaptive_grid.follow_positions,
whose position advance and continuous tempo step are reused here unchanged):

1. Tempo at each beat. "Our score follower assumes that a user draws a tempo
   at each beat as a random walk, and plays in that tempo inside the beat"
   (an HHSMM from Maezawa & Yamamoto 2017, not available to us). Probability
   mass that crosses a beat boundary during an update takes one random-walk
   step in log tempo; within a beat the tempo is constant. Stage A instead
   drifts tempo continuously and pulls it back toward the start tempo.
2. Tempo feedback. "At each time step, the tracked tempo is low-pass filtered
   and is fed back as the target BPM condition of the downbeat phase tracker."
   Stage A gives the network the written tempo throughout.
3. Updates at every network frame (50 ms); Stage A updates every 200 ms.

Kept from Stage A: the grid of unfolded positions and tempi, linear stochastic
rounding of position advances, the likelihood p_h(x|s) p_d(s) with each term
raised to elapsed seconds times its strength, the harmonic observation, and the
MAP position output. Our choices where the paper is silent (labelled in the
config): the tempo grid (0.5x to 2x the start tempo, 2.5% steps), the per-beat
step size (3% std in log tempo), reflection at the grid ends, the start-tempo
prior (10% std), and the feedback time constant (2 s, exponential smoothing of
the posterior-mean tempo). Everything is causal: the condition used at an
update comes from the posterior before it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import follower_data  # noqa: F401  (sys.path for the benchmark modules)
from adaptive_grid import _advance, _tempo_predict, bar_indices


@dataclass(frozen=True)
class YamahaConfig:
    bins_per_beat: int = 8                  # 32 positions per 4/4 bar, the network's bar-phase resolution
    tempo_min_ratio: float = 0.5            # ours
    tempo_max_ratio: float = 2.0            # ours
    tempo_step: float = 0.025               # ours: ratio between neighbouring tempo bins
    initial_tempo_log_std: float = 0.10     # ours, as Stage A's initial_tempo_ratio_std
    tempo_model: str = "beat"               # "beat" (paper) or "continuous" (Stage A's dynamics)
    beat_tempo_log_std: float = 0.03        # ours: random-walk step per beat
    continuous_rate_per_second: float = .45  # Stage A's diffusion on 2.5% bins (benchmark s07)
    emission_power_per_second: float = 1.0
    downbeat_power_per_second: float = 1.0
    probability_floor_mix: float = .02
    feedback: bool = True                   # paper: tracked tempo -> BPM condition
    feedback_time_constant: float = 2.0     # ours: low-pass time constant, seconds


def _tempo_grid(config):
    k_max = int(np.floor(np.log(config.tempo_max_ratio) / np.log1p(config.tempo_step) + 1e-9))
    k_min = int(np.ceil(np.log(config.tempo_min_ratio) / np.log1p(config.tempo_step) - 1e-9))
    return (1 + config.tempo_step) ** np.arange(k_min, k_max + 1)


def _beat_kernel(config):
    sigma = config.beat_tempo_log_std / np.log1p(config.tempo_step)
    taps = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-.5 * (taps / sigma) ** 2)
    return taps, weights / weights.sum()


def _walk(mass, taps, weights):
    """One random-walk step along the tempo axis; mass leaving the grid stays put."""
    out = np.zeros_like(mass)
    n = len(mass)
    kept = np.zeros(n)
    for tap, weight in zip(taps, weights):
        lo, hi = max(0, -tap), min(n, n - tap)
        out[lo + tap:hi + tap] += weight * mass[lo:hi]
        kept[lo:hi] += weight
    return out + mass * (1 - kept)[:, None]


def follow(times, chord_probabilities, available, chord_starts_beats, chord_ends_beats, chord_ids,
           bar_starts_beats, beats_per_bar, initial_bpm, *, start_time, network=None, config=YamahaConfig()):
    times = np.asarray(times, dtype=float)
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
    taps, weights = _beat_kernel(config)
    n_classes = probability.shape[1]
    reachable, hint, previous = 0, float(initial_bpm), float(start_time)
    out = {key: np.empty(len(times)) for key in ("position_beats", "tempo_bpm", "mean_tempo_bpm", "bpm_condition",
                                                 "position_confidence")}
    for frame, now in enumerate(times):
        elapsed = now - previous
        if elapsed:
            beats_moved = tempi * elapsed / 60.
            reachable += int(np.ceil(beats_moved.max() / step))
            if reachable + 2 > len(positions):  # grow the grid causally, as in Stage A
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
                posterior = posterior - crossed + _walk(crossed, taps, weights)
            known = state_chords >= 0
            log_weight = np.zeros(len(positions))
            if available[frame] and config.emission_power_per_second:
                smooth = (1 - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
                observation = np.full(len(positions), 1. / n_classes)
                observation[known] = smooth[state_chords[known]]
                log_h = np.log(observation)
                log_weight += (log_h - log_h.max()) * elapsed * config.emission_power_per_second
            if network is not None and config.downbeat_power_per_second:
                log_d = network.bar_log_probs(network.frame_at(now), hint)
                phase = np.mod(positions, beats_per_bar) / beats_per_bar
                log_pd = log_d[np.minimum((phase * len(log_d)).astype(int), len(log_d) - 1)]
                log_weight += (log_pd - log_pd.max()) * elapsed * config.downbeat_power_per_second
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
