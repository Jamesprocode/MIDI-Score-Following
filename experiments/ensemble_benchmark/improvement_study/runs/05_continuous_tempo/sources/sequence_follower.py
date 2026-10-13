"""Experimental online Viterbi beam follower with continuous beat positions.

This is an experimental baseline, not a reproduction of a published follower.
It maximizes a causal harmonic-sequence path score over position/tempo histories,
whereas the original benchmark sums probabilities on a fixed position grid.
Positions advance exactly by tempo * elapsed seconds / 60. Candidate tempo paths
branch locally; nearby (position, tempo) states are merged by maximum path score,
then a finite beam is retained. Merging never rounds the surviving position.

The initial tempo prior is Gaussian around the supplied approximate BPM. After
initialization there is no attraction back to that BPM. The local tempo random
walk reflects at its configured limits while preserving the same self-transition
probability at every tempo, avoiding a Viterbi preference for boundary tempi.
Silence or uniform evidence therefore preserves the initial continuous clock
when ratio1 is in the grid. Future observations never revise past outputs.

Inputs are an untimed, already unfolded chord plan plus synchronized starting
position/time. No timed chord/bar annotations, inferred evaluation offsets, or
future frame times enter inference. This baseline cannot follow unplanned repeats,
skips, or unknown starts. Chord probabilities are compatibility potentials, not
claimed calibrated likelihoods. Parameters must be fixed across evaluated songs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class SequenceConfig:
    beam_width: int = 256
    merge_bins_per_beat: int = 8
    tempo_ratios: tuple = (.7, .775, .85, .925, 1., 1.075, 1.15, 1.225, 1.3)
    initial_tempo_ratio_std: float = .10
    tempo_neighbor_rate_per_second: float = .05
    emission_power_per_second: float = 1.
    probability_floor_mix: float = .02
    boundary_policy: str = "continuous_neutral_overflow"


FilterConfig = SequenceConfig


def _prune(position, tempo_index, log_score, config, n_tempi):
    """Approximate max-product state merge, followed by a bounded top-score beam.

    Exact merging keys use a position cell and tempo index. Among paths in a cell
    retain the highest score, keeping its actual continuous position. Equal scores
    prefer the smaller position, then the smaller tempo index, deterministically.
    """
    key = np.floor(position * config.merge_bins_per_beat).astype(np.int64) * n_tempi + tempo_index
    order = np.lexsort((position, -log_score, key))
    first = np.r_[True, np.diff(key[order]) != 0]
    survivors = order[first]
    rank = np.lexsort((tempo_index[survivors], position[survivors], -log_score[survivors]))
    keep = survivors[rank[:config.beam_width]]
    score = log_score[keep]
    score = score - score[0]
    return position[keep], tempo_index[keep], score


def _expand(position, tempo_index, log_score, tempi, elapsed, hop):
    """Branch into stay/adjacent tempi, then advance continuous beat position."""
    n_tempi = len(tempi)
    if n_tempi == 1 or hop == 0:
        return position + tempi[tempo_index] * elapsed / 60., tempo_index.copy(), log_score.copy()
    new_index = tempo_index[:, None] + np.array([-1, 0, 1])
    weight = np.tile([hop, 1. - 2. * hop, hop], (len(position), 1))
    # Reflect the invalid jump inward, keeping stay probability unchanged.
    weight[tempo_index == 0, 2] += hop
    weight[tempo_index == n_tempi - 1, 0] += hop
    valid = (new_index >= 0) & (new_index < n_tempi)
    parent = np.broadcast_to(np.arange(len(position))[:, None], new_index.shape)[valid]
    next_tempo = new_index[valid]
    next_position = position[parent] + tempi[next_tempo] * elapsed / 60.
    next_score = log_score[parent] + np.log(weight[valid])
    return next_position, next_tempo, next_score


def follow_positions(times, chord_probabilities, available, chord_starts_beats,
                     chord_ends_beats, chord_ids, bar_starts_beats, initial_bpm, *,
                     initial_position_beats=0., start_time=None, config=None):
    """Return each frame's best causal Viterbi-path endpoint, with no traceback.

    Interface mirrors the original filter. Probabilities are (T,K), availability
    is (T,), score intervals are ordered/nonoverlapping beat intervals, chord IDs
    are K-column indices or -1 for unknown/no-chord, and bar starts exclude the
    endpoint sentinel. Config may be SequenceConfig, a dictionary, or None.

    Observation log-potentials are multiplied by elapsed seconds and the fixed
    emission_power_per_second. False availability skips the observation. The
    emitted q may move backward when another retained forward-moving history
    becomes best. This is an online decision, not retrospective smoothing.

    Beyond-plan positions receive a neutral potential and bar_index=len(bars).
    Positions are never clamped. Beam scores/gaps are uncalibrated path scores;
    they must not be interpreted as posterior confidence or comparable likelihoods
    across algorithms. A larger beam/finer merge grid is a sensitivity condition,
    not an automatic per-song accuracy-tuned choice.
    """
    config = SequenceConfig() if config is None else config
    if isinstance(config, dict):
        config = SequenceConfig(**config)
    if not isinstance(config, SequenceConfig):
        raise ValueError("config must be SequenceConfig, dict, or None")
    for name in ("beam_width", "merge_bins_per_beat"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer")
    settings = [config.initial_tempo_ratio_std, config.tempo_neighbor_rate_per_second,
                config.emission_power_per_second, config.probability_floor_mix]
    if not np.isfinite(settings).all() or settings[0] <= 0 or min(settings[1:3]) < 0 or not 0 <= settings[3] <= 1:
        raise ValueError("invalid tempo or observation configuration")
    if config.boundary_policy != "continuous_neutral_overflow":
        raise ValueError("only continuous_neutral_overflow is supported")
    ratios = np.asarray(config.tempo_ratios, dtype=float)
    if ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios <= 0) or np.any(np.diff(ratios) <= 0):
        raise ValueError("tempo ratios must be positive, finite, strictly increasing")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be nonempty, finite, strictly increasing")
    if not np.isfinite(initial_bpm) or initial_bpm <= 0:
        raise ValueError("initial_bpm must be positive and finite")
    start_time = float(times[0]) if start_time is None else float(start_time)
    if not np.isfinite(start_time) or start_time > times[0]:
        raise ValueError("start_time must be finite and no later than the first frame")
    if not np.isfinite(initial_position_beats) or initial_position_beats < 0:
        raise ValueError("initial position must be finite and nonnegative")
    starts = np.asarray(chord_starts_beats, dtype=float)
    ends = np.asarray(chord_ends_beats, dtype=float)
    ids = np.asarray(chord_ids)
    bars = np.asarray(bar_starts_beats, dtype=float)
    probability = np.asarray(chord_probabilities, dtype=float)
    available = np.asarray(available, dtype=bool)
    if probability.ndim != 2 or probability.shape[0] != len(times) or probability.shape[1] < 1:
        raise ValueError("chord_probabilities must have nonempty shape (T,K)")
    if not np.isfinite(probability).all() or np.any(probability < 0) or available.shape != (len(times),):
        raise ValueError("invalid probability or availability arrays")
    mass = probability.sum(axis=1)
    if np.any(mass[available] <= 0):
        raise ValueError("available probability rows need positive mass")
    probability = probability / np.maximum(mass[:, None], np.finfo(float).tiny)
    if starts.ndim != 1 or not len(starts) or ends.shape != starts.shape or ids.shape != starts.shape:
        raise ValueError("score starts, ends and IDs must be matching nonempty vectors")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("invalid score beat intervals")
    if np.any(np.diff(starts) <= 0) or np.any(starts[1:] < ends[:-1] - 1e-9):
        raise ValueError("score intervals must be ordered and nonoverlapping")
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)) or np.any(ids < -1) or np.any(ids >= probability.shape[1]):
        raise ValueError("chord IDs must be -1 or probability-column integers")
    if bars.ndim != 1 or not len(bars) or not np.isfinite(bars).all() or np.any(np.diff(bars) <= 0) or bars[-1] >= ends[-1]:
        raise ValueError("bar starts must be ordered, finite, and exclude the end sentinel")
    if initial_position_beats >= ends[-1]:
        raise ValueError("initial position must precede score end")
    ids = ids.astype(int)
    tempi = initial_bpm * ratios
    position = np.full(len(tempi), initial_position_beats, dtype=float)
    tempo_index = np.arange(len(tempi))
    log_score = -.5 * ((ratios - 1.) / config.initial_tempo_ratio_std) ** 2
    position, tempo_index, log_score = _prune(position, tempo_index, log_score, config, len(tempi))

    q_output = np.empty(len(times))
    tempo_output = np.empty(len(times))
    score_gap = np.empty(len(times))
    beam_size = np.empty(len(times), dtype=int)
    n_classes = probability.shape[1]
    previous_time = start_time
    for frame, now in enumerate(times):
        elapsed = now - previous_time
        if elapsed:
            # Fixed bound keeps stay probability > any individual tempo-change
            # branch, including after an unusually long gap between frames.
            substeps = max(1, int(np.ceil(elapsed * config.tempo_neighbor_rate_per_second / .1)))
            dt = elapsed / substeps
            hop = dt * config.tempo_neighbor_rate_per_second
            for substep in range(substeps):
                position, tempo_index, log_score = _expand(position, tempo_index, log_score, tempi, dt, hop)
                if substep < substeps - 1:
                    position, tempo_index, log_score = _prune(position, tempo_index, log_score, config, len(tempi))
        if available[frame] and elapsed and config.emission_power_per_second:
            interval = np.searchsorted(starts, position, side="right") - 1
            valid = (interval >= 0) & (position < ends[np.maximum(interval, 0)])
            chord = np.full(len(position), -1, dtype=int)
            chord[valid] = ids[interval[valid]]
            smooth = (1. - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
            potential = np.full(len(position), 1. / n_classes)
            known = chord >= 0
            potential[known] = smooth[chord[known]]
            log_score += np.log(np.maximum(potential, np.finfo(float).tiny)) * elapsed * config.emission_power_per_second
        position, tempo_index, log_score = _prune(position, tempo_index, log_score, config, len(tempi))
        q_output[frame] = position[0]
        tempo_output[frame] = tempi[tempo_index[0]]
        score_gap[frame] = log_score[0] - log_score[1] if len(log_score) > 1 else np.inf
        beam_size[frame] = len(position)
        previous_time = now
    bar_index = np.searchsorted(bars, q_output, side="right") - 1
    beyond_plan = q_output >= ends[-1]
    bar_index[beyond_plan] = len(bars)
    return {"position_beats": q_output, "tempo_bpm": tempo_output,
            "bar_index": bar_index, "out_of_score": beyond_plan,
            "path_score_gap": score_gap, "beam_size": beam_size,
            "final_beam_positions": position, "final_beam_tempo_bpm": tempi[tempo_index],
            "final_beam_relative_log_scores": log_score,
            "config": asdict(config), "implementation": "experimental_online_viterbi_beam"}


def checks():
    """Synthetic checks only; no song annotation is used to select parameters."""
    starts = np.arange(0., 160., 4.)
    ends = starts + 4.
    ids = np.arange(len(starts)) % 7
    times = np.arange(0., 60., .2)
    uniform = np.ones((len(times), 7)) / 7.
    silence = follow_positions(times, uniform, np.zeros(len(times), bool), starts, ends, ids, starts, 120)
    clock = times * 2.
    assert np.allclose(silence["position_beats"], clock, atol=1e-10, rtol=0)
    assert np.all(silence["tempo_bpm"] == 120.)
    uninformative = follow_positions(times, uniform, np.ones(len(times), bool), starts, ends, ids, starts, 120)
    assert np.array_equal(uninformative["position_beats"], silence["position_beats"])
    assert np.all(silence["beam_size"] <= SequenceConfig().beam_width)
    assert silence["beam_size"][-1] > 1

    q = np.where(times <= 12., times * 2., 24. + (times - 12.) * 2.4)
    labels = ids[np.minimum((q / 4).astype(int), len(ids)-1)]
    evidence = np.full((len(times), 7), .005)
    evidence[np.arange(len(times)), labels] = .97
    dynamic = follow_positions(times, evidence, np.ones(len(times), bool), starts, ends, ids, starts, 120)
    late = times >= 40.
    error = np.abs(dynamic["position_beats"] - q)
    assert error[late].mean() < np.abs(clock[late] - q[late]).mean()
    assert dynamic["tempo_bpm"][late].mean() > 125.
    cut = 193
    prefix = follow_positions(times[:cut], evidence[:cut], np.ones(cut, bool), starts, ends, ids, starts, 120)
    assert np.array_equal(prefix["position_beats"], dynamic["position_beats"][:cut])
    assert np.array_equal(prefix["tempo_bpm"], dynamic["tempo_bpm"][:cut])

    short_starts = np.arange(0., 16., 4.)
    overflow = follow_positions(times, uniform, np.zeros(len(times), bool), short_starts, short_starts+4,
                                np.arange(len(short_starts)) % 7, short_starts, 120)
    assert np.allclose(overflow["position_beats"], clock, atol=1e-10, rtol=0)
    assert overflow["bar_index"][-1] == len(short_starts)
    assert overflow["out_of_score"][-1]
    # Broad grid and irregular frame gaps must not create a silent tempo change.
    sparse_times = np.array([0., .2, .4, 20., 20.2, 120.])
    broad = follow_positions(sparse_times, np.ones((6, 2)), np.zeros(6, bool), [0.,4.], [4.,8.],
                             [0,1], [0.,4.], 120,
                             config={"tempo_ratios": tuple(np.round(np.arange(.5,1.81,.1),2))})
    assert np.allclose(broad["position_beats"], sparse_times*2., atol=1e-10, rtol=0)
    assert np.all(broad["tempo_bpm"] == 120.)
    return {"silence_max_error_beats": float(np.max(np.abs(silence["position_beats"] - clock))),
            "late_tempo_change_mean_abs_error_beats": float(error[late].mean()),
            "late_tempo_change_mean_estimated_bpm": float(dynamic["tempo_bpm"][late].mean()),
            "final_beam_size": int(dynamic["beam_size"][-1])}


if __name__ == "__main__":
    started = time.perf_counter()
    print(checks())
    print(f"Experimental Viterbi-beam checks passed in {time.perf_counter()-started:.3f}s")
