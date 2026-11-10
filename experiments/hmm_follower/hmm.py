"""The HMM follower, with the cues as plug-ins.

Hidden state: (tempo, place). A place is a position on the unfolded chart in beats, on a grid of ``bins_per_beat``
steps per beat; a tempo is one of a grid of BPMs around the start tempo. At each update (every 50 ms in our runs):

  1. predict  every (tempo, place) moves forward by tempo x elapsed time (linear stochastic rounding on the grid); the
              tempo takes a random-walk step when a path crosses a beat ("beat" model, Maezawa 2023) or drifts
              continuously ("continuous", Stage A of the benchmark);
  2. update   each plugged cue c gives log L_c(place) (or abstains with None), and the posterior is multiplied by
                  exp( sum_c  scale_c(log L_c) * elapsed * w_c )
              (scaling.py; "max" for every cue so far);
  3. output   the place with the most probability, and the tempo marginal. The posterior-mean tempo, low-pass filtered,
              is the BPM hint the cues get at the next update (Maezawa's tempo feedback).

Only the cues know about music. Harmony, rhythm and melody are three plug-ins (cues/harmony.py, cues/rhythm.py,
cues/melody.py), each with one method, ``log_likelihood(now, bpm, positions)``. A new chord model, beat tracker or
melody matcher is a new cue, or a new model inside one; this file does not change.

This is yamaha_follower/yamaha_filter.follow, adaptive_weights/weighted_filter.follow_weighted and
melody_evidence/run_jiang_resets.follow_with_resets in one place. With the same cues and weights it gives the same
output, bit for bit (tests/test_same_as_old.py). Choices where the paper is silent are labelled "ours", as in
YamahaConfig.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from .scaling import SCALINGS


@dataclass(frozen=True)
class HMMConfig:
    bins_per_beat: int = 8                  # 32 places per 4/4 bar, the network's bar-phase resolution
    tempo_min_ratio: float = 0.5            # ours
    tempo_max_ratio: float = 2.0            # ours
    tempo_step: float = 0.025               # ours: ratio between neighbouring tempo bins
    initial_tempo_log_std: float = 0.10     # ours, as Stage A's initial_tempo_ratio_std
    tempo_model: str = "beat"               # "beat" (paper) or "continuous" (Stage A's dynamics)
    beat_tempo_log_std: float = 0.03        # ours: random-walk step per beat
    beat_tempo_reversion: float = 0.0       # ours: per-beat pull of log tempo toward the start tempo (B7: 0.02)
    exact_beat_crossing: bool = False       # ours, 11-04: one tempo step per beat crossing (TEMPO_DRIFT.md); False = B5
    continuous_rate_per_second: float = .45  # Stage A's diffusion on 2.5% bins (benchmark s07)
    feedback: bool = True                   # paper: tracked tempo -> the cues' BPM hint
    feedback_time_constant: float = 2.0     # ours: low-pass time constant, seconds

    @classmethod
    def like(cls, other):
        """The fields of another config (e.g. yamaha_filter.YamahaConfig) that this one has."""
        return cls(**{f.name: getattr(other, f.name) for f in fields(cls) if hasattr(other, f.name)})


@dataclass
class Plugged:
    """A cue in the follower, with its weight (per second of evidence) and the scaling of its log-likelihood."""
    cue: object
    weight: float = 1.
    scaling: str = "max"


class Follower:
    """The follower for one take, one update at a time (``step``), as it would run live."""

    def __init__(self, cues, score_end_beats, initial_bpm, *, start_time, start_position=0., config=HMMConfig()):
        self.cues = [c if isinstance(c, Plugged) else Plugged(*c) if isinstance(c, tuple) else Plugged(c) for c in cues]
        self.config = config
        self.step_beats = 1. / config.bins_per_beat
        self.positions = np.arange(int(np.ceil(score_end_beats / self.step_beats)) + 1, dtype=float) * self.step_beats
        self.ratios = _tempo_grid(config)
        self.prior = np.exp(-.5 * (np.log(self.ratios) / config.initial_tempo_log_std) ** 2)
        self.prior /= self.prior.sum()
        self.taps, self.kernel = _beat_kernel(config)
        self.transition = _beat_transition(config, self.ratios) if config.beat_tempo_reversion else None
        self.previous, self.reachable = float(start_time), 0
        self._elapsed, self._mean_tempo = 0., float(initial_bpm)
        self._start(start_position, initial_bpm)

    def step(self, now, weights=None):
        """Move to time ``now`` and take in each cue's evidence; returns the estimate. ``weights`` (one per cue)
        replaces the plugged weights for this update only (the hook for adaptive weighting)."""
        config = self.config
        if config.feedback and self._elapsed:  # the last update's tempo, low-pass filtered, becomes the hint
            self.hint += min(1., self._elapsed / config.feedback_time_constant) * (self._mean_tempo - self.hint)
        elapsed = now - self.previous
        if elapsed:
            self._predict(elapsed)
            log_weight = np.zeros(len(self.positions))
            for k, plug in enumerate(self.cues):
                w = plug.weight if weights is None else weights[k]
                if not w:
                    continue
                log_l = plug.cue.log_likelihood(now, self.hint, self.positions)
                if log_l is not None:
                    log_weight += SCALINGS[plug.scaling](log_l) * elapsed * w
            self.posterior *= np.exp(log_weight)[None, :]
        total = self.posterior.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("follower lost all probability mass")
        self.posterior /= total
        self._elapsed, self.previous = elapsed, now
        return self._estimate()

    def reset(self, position, bpm):
        """Start again at ``position`` (beats) with tempo ``bpm``: Jiang et al. (2025)'s hard reset, as
        run_jiang_resets.py does it (the tempo grid is re-centred on ``bpm``). Returns the new estimate."""
        self._start(position, bpm)
        return self._estimate()

    def _start(self, position, bpm):
        start = int(round(position / self.step_beats))
        if start + 2 > len(self.positions):
            self._grow(start + 2)
        self.tempi = float(bpm) * self.ratios
        self.posterior = np.zeros((len(self.tempi), len(self.positions)))
        self.posterior[:, start] = self.prior
        self.reachable, self.hint = max(self.reachable, start), float(bpm)

    def _grow(self, size):
        """Extend the place grid to at least ``size`` places (causally: only as far as some path could have reached);
        returns how many places were added."""
        pad = max(size, len(self.positions) + 256) - len(self.positions)
        self.positions = np.arange(len(self.positions) + pad, dtype=float) * self.step_beats
        return pad

    def _predict(self, elapsed):
        config, step = self.config, self.step_beats
        beats_moved = self.tempi * elapsed / 60.
        self.reachable += int(np.ceil(beats_moved.max() / step))
        if self.reachable + 2 > len(self.positions):
            self.posterior = np.pad(self.posterior, ((0, 0), (0, self._grow(self.reachable + 2))))
        if config.tempo_model == "continuous":
            self.posterior = _tempo_predict(self.posterior, elapsed * config.continuous_rate_per_second, self.prior)
        if config.tempo_model == "beat" and config.exact_beat_crossing:
            self.posterior, crossed = _advance_crossed(self.posterior, beats_moved / step, config.bins_per_beat)
        else:
            self.posterior = _advance(self.posterior, beats_moved / step)
        if config.tempo_model == "beat":
            if not config.exact_beat_crossing:  # B5: landing place against the unrounded move
                fraction = np.mod(self.positions + 1e-9, 1.0)
                crossed = self.posterior * (fraction[None, :] < beats_moved[:, None] - 1e-9)
            moved = _walk(crossed, self.taps, self.kernel) if self.transition is None else self.transition @ crossed
            self.posterior = self.posterior - crossed + moved

    def _estimate(self):
        position_marginal, tempo_marginal = self.posterior.sum(axis=0), self.posterior.sum(axis=1)
        best = position_marginal.argmax()
        self._mean_tempo = tempo_marginal @ self.tempi
        return {"position_beats": self.positions[best], "position_confidence": position_marginal[best],
                "tempo_bpm": self.tempi[tempo_marginal.argmax()], "mean_tempo_bpm": self._mean_tempo,
                "bpm_condition": self.hint}


KEYS = ("position_beats", "tempo_bpm", "mean_tempo_bpm", "bpm_condition", "position_confidence")


def follow(times, cues, score_end_beats, bar_starts_beats, initial_bpm, *, start_time, start_position=0.,
           config=HMMConfig(), weights=None, reset=None):
    """Run a Follower over ``times``; one array per estimate (as yamaha_filter.follow), plus ``bar_index``.

    cues      Plugged cues, (cue, weight[, scaling]) tuples, or bare cues (weight 1)
    weights   optional per-update weights, broadcast to (len(times), len(cues))
    reset     optional function (frame, estimate) -> (position_beats, bpm) or None: a hard reset there
    """
    follower = Follower(cues, score_end_beats, initial_bpm, start_time=start_time, start_position=start_position,
                        config=config)
    times = np.asarray(times, dtype=float)
    if weights is not None:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(times), len(follower.cues)))
    out = {key: np.empty(len(times)) for key in KEYS}
    out["reset"] = np.zeros(len(times), bool)
    for frame, now in enumerate(times):
        estimate = follower.step(now, None if weights is None else weights[frame])
        target = reset(frame, estimate) if reset is not None else None
        if target is not None:
            estimate = follower.reset(*target)
            out["reset"][frame] = True
        for key in KEYS:
            out[key][frame] = estimate[key]
    out["bar_index"] = bar_indices(out["position_beats"], bar_starts_beats, score_end_beats)
    out["config"] = asdict(config)
    return out


def bar_indices(position_beats, bar_starts_beats, score_end_beats):
    """Zero-based unfolded bar index (a bar line belongs to the bar after it); len(bars) at or past the end
    (improvement_study/adaptive_grid.bar_indices)."""
    bars = np.asarray(bar_starts_beats, dtype=float)
    if bars.ndim != 1 or not len(bars) or not np.isfinite(bars).all() or np.any(np.diff(bars) <= 0):
        raise ValueError("bar_starts_beats must be finite and strictly increasing")
    if not np.isfinite(score_end_beats) or score_end_beats <= bars[-1]:
        raise ValueError("score_end_beats must be finite and after the last bar start")
    positions = np.asarray(position_beats)
    return np.where(positions >= score_end_beats, len(bars), np.searchsorted(bars, positions, side="right") - 1)


# ---------------------------------------------------------------- the tempo and position moves (copied, unchanged)
# from yamaha_follower/yamaha_filter.py and improvement_study/adaptive_grid.py, so this package does not depend on
# their private helpers


def _tempo_grid(config):
    k_max = int(np.floor(np.log(config.tempo_max_ratio) / np.log1p(config.tempo_step) + 1e-9))
    k_min = int(np.ceil(np.log(config.tempo_min_ratio) / np.log1p(config.tempo_step) - 1e-9))
    return (1 + config.tempo_step) ** np.arange(k_min, k_max + 1)


def _beat_kernel(config):
    sigma = config.beat_tempo_log_std / np.log1p(config.tempo_step)
    taps = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-.5 * (taps / sigma) ** 2)
    return taps, weights / weights.sum()


def _beat_transition(config, ratios):
    """Per-beat tempo transition matrix (to, from) with reversion toward ratio 1."""
    k = np.log(ratios) / np.log1p(config.tempo_step)
    sigma = config.beat_tempo_log_std / np.log1p(config.tempo_step)
    mean = k * (1 - config.beat_tempo_reversion)
    weights = np.exp(-.5 * ((k[:, None] - mean[None, :]) / sigma) ** 2)
    return weights / weights.sum(axis=0, keepdims=True)


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


def _tempo_predict(posterior, hop, stationary_prior):
    """Stage A's continuous tempo drift: local Metropolis moves with the start-tempo prior stationary."""
    right = np.zeros(len(stationary_prior))
    left = np.zeros(len(stationary_prior))
    right[:-1] = hop * np.minimum(1., stationary_prior[1:] / stationary_prior[:-1])
    left[1:] = hop * np.minimum(1., stationary_prior[:-1] / stationary_prior[1:])
    prediction = posterior * (1. - left - right)[:, None]
    prediction[1:] += posterior[:-1] * right[:-1, None]
    prediction[:-1] += posterior[1:] * left[1:, None]
    return prediction


def _advance(posterior, shifts):
    """Linear stochastic rounding of position increments; the caller has grown the grid to every reachable place."""
    predicted = np.zeros_like(posterior)
    size = posterior.shape[1]
    for j, shift in enumerate(shifts):
        whole = int(np.floor(shift))
        fraction = shift - whole
        for offset, weight in ((whole, 1. - fraction), (whole + 1, fraction)):
            if weight == 0:
                continue
            if offset >= size:
                if np.any(posterior[j]):
                    raise ValueError("position grid does not cover reachable destinations")
            else:
                predicted[j, offset:] += posterior[j, :size-offset] * weight
                if offset and np.any(posterior[j, size-offset:]):
                    raise ValueError("position grid does not cover reachable destinations")
    return predicted


def _advance_crossed(posterior, shifts, bins_per_beat):
    """_advance, and the part of its result that crossed a beat boundary in this update (exact_beat_crossing).

    The rounding moves each tempo row's mass by floor(shift) or floor(shift) + 1 whole bins. Mass that moved k bins
    crossed a boundary exactly when it lands fewer than k bins past a beat (at most one boundary per update).
    """
    predicted = _advance(posterior, shifts)
    crossed = np.zeros_like(posterior)
    size = posterior.shape[1]
    phase = np.arange(size) % bins_per_beat
    for j, shift in enumerate(shifts):
        whole = int(np.floor(shift))
        fraction = shift - whole
        for offset, weight in ((whole, 1. - fraction), (whole + 1, fraction)):
            if weight == 0 or offset == 0 or offset >= size:
                continue
            if offset > bins_per_beat:
                raise ValueError("more than one beat per update; exact_beat_crossing needs a shorter update step")
            crossed[j, offset:] += posterior[j, :size-offset] * weight * (phase[offset:] < offset)
    return predicted, crossed
