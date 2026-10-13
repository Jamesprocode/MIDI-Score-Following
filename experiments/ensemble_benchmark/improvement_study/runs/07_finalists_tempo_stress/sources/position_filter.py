"""Shared causal position/tempo grid filter for the ensemble benchmark.

This is an explicitly specified baseline, not a reproduction of Maezawa, Pardo,
Matchmaker, or a new score-following algorithm. A Bayesian forward recursion
retains a distribution over unfolded beat position and tempo. Position advances
with tempo; tempo performs a local, mean-reverting random walk. Chord compatibility supplies a
tempered observation potential. No timed performance annotations enter inference.

The known input score is an already unfolded chord plan, in musical beats, with
bar starts. Initialization supplies its known starting position, synchronized
start time and an approximate initial BPM. Repeats/skips differing from that plan
and unknown starts are outside this baseline's scope. Silence predicts forward;
it never freezes position or queries reference timing.

There is no absorbing score endpoint. The latent grid grows causally to retain
all reachable positions, including neutral-evidence positions beyond the plan.
Such predictions remain numerically beyond the plan and count as out of score;
they are not clamped into the final bar. This prevents diffuse forward probability
from collapsing into an artificial point mass at the endpoint.

Joint position/tempo estimation is established score-following practice; see
Jiang & Raphael, ISMIR 2020, https://archives.ismir.net/ismir2020/paper/000159.pdf.
The particular discretization and constants here are benchmark choices, not that
paper's switching-state-space implementation. Chord probabilities are treated as
compatibility potentials, not claimed calibrated observation likelihoods.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    bins_per_beat: int = 2
    tempo_ratios: tuple = (.7, .775, .85, .925, 1., 1.075, 1.15, 1.225, 1.3)
    initial_tempo_ratio_std: float = .10
    tempo_neighbor_rate_per_second: float = .05
    emission_power_per_second: float = 1.0
    probability_floor_mix: float = .02
    boundary_policy: str = "causal_neutral_overflow"


def _times_and_start(times, initial_bpm, initial_position_beats, start_time):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a nonempty, finite, strictly increasing array")
    if not np.isfinite(initial_bpm) or initial_bpm <= 0:
        raise ValueError("initial_bpm must be positive and finite")
    if not np.isfinite(initial_position_beats) or initial_position_beats < 0:
        raise ValueError("initial_position_beats must be finite and nonnegative")
    start_time = float(times[0]) if start_time is None else float(start_time)
    if not np.isfinite(start_time) or start_time > times[0]:
        raise ValueError("start_time must be finite and no later than the first frame")
    return times, start_time


def _bars(bar_starts_beats):
    bars = np.asarray(bar_starts_beats, dtype=float)
    if bars.ndim != 1 or not len(bars) or not np.isfinite(bars).all() or np.any(np.diff(bars) <= 0):
        raise ValueError("bar_starts_beats must be finite and strictly increasing")
    return bars


def bar_indices(position_beats, bar_starts_beats, score_end_beats=None):
    """Zero-based unfolded bar index; bar boundaries belong to the following bar.

    Pass actual bar starts, excluding an optional score-end sentinel. Positions
    before the first bar return -1. With score_end_beats supplied, positions at or
    beyond that endpoint return len(bar_starts), an explicit out-of-score index.
    Without it, the final bar's end is unknown and cannot be checked; benchmark
    inference always supplies the known score endpoint.
    """
    bars = _bars(bar_starts_beats)
    positions = np.asarray(position_beats)
    result = np.searchsorted(bars, positions, side="right") - 1
    if score_end_beats is not None:
        if not np.isfinite(score_end_beats) or score_end_beats <= bars[-1]:
            raise ValueError("score_end_beats must be finite and after the last bar start")
        result = np.where(positions >= score_end_beats, len(bars), result)
    return result


def clock_positions(times, bar_starts_beats, initial_bpm, *,
                    initial_position_beats=0., start_time=None, score_end_beats=None):
    """Independent constant-tempo baseline using only the supplied initialization.

    Optional score_end_beats marks beyond-plan positions as out of score. It never
    clamps positions or supplies a performance end time. Pass it when comparing
    with follow_positions so both systems use the same endpoint convention.
    """
    times, start_time = _times_and_start(times, initial_bpm, initial_position_beats, start_time)
    position = initial_position_beats + (times - start_time) * initial_bpm / 60.
    if score_end_beats is not None:
        if not np.isfinite(score_end_beats) or score_end_beats <= initial_position_beats:
            raise ValueError("score_end_beats must be after the initial position")
    return {"position_beats": position,
            "tempo_bpm": np.full(len(times), initial_bpm, dtype=float),
            "bar_index": bar_indices(position, bar_starts_beats, score_end_beats)}


def _tempo_predict(posterior, hop, stationary_prior):
    """Local Metropolis transitions with the initialization prior stationary.

    This weak pull toward the approximate initial tempo prevents an unobserved
    tempo random walk from eventually assigning equal probability to the extreme
    tempo bounds. Evidence can still move probability away from the prior.
    """
    right = np.zeros(len(stationary_prior))
    left = np.zeros(len(stationary_prior))
    right[:-1] = hop * np.minimum(1., stationary_prior[1:] / stationary_prior[:-1])
    left[1:] = hop * np.minimum(1., stationary_prior[:-1] / stationary_prior[1:])
    prediction = posterior * (1. - left - right)[:, None]
    prediction[1:] += posterior[:-1] * right[:-1, None]
    prediction[:-1] += posterior[1:] * left[1:, None]
    return prediction


def _advance(posterior, shifts):
    """Linear stochastic rounding of position increments on an expanded grid.

    This sparse local transition is O(number of states), without a dense state
    transition matrix. Rounding adds grid-dependent phase diffusion, so resolution
    is part of the model specification. Every latent path is nondecreasing even
    though framewise MAP estimates can move backward when hypotheses reweight.
    The caller must allocate all reachable destination bins before calling; an
    undersized grid raises instead of silently dropping or accumulating mass.
    """
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


def follow_positions(times, chord_probabilities, available, chord_starts_beats,
                     chord_ends_beats, chord_ids, bar_starts_beats, initial_bpm, *,
                     initial_position_beats=0., start_time=None, config=None):
    """Filter framewise chord evidence into MAP unfolded position and tempo.

    probabilities: (T,K), nonnegative, normalized internally for available rows.
    available: (T,) bool; False skips the observation update completely.
    chord_starts/ends_beats: ordered, nonoverlapping untimed score intervals.
    chord_ids: one vocabulary column per interval; -1 means unknown/no-chord and
        gives a neutral observation potential. Gaps in the plan are also neutral.
    bar_starts_beats: real unfolded bar starts, WITHOUT an end sentinel.
    initial_bpm/start_time/initial_position_beats: initialization information only.

    Every available observation is raised to
    ``elapsed_seconds * emission_power_per_second``. For a trailing window W,
    using 1/W is a simple common tempering choice, not a claim that overlapping
    observations are independent. Fix this parameter across compared matchers.
    At elapsed time zero no evidence is applied. No look-ahead, smoothing, Viterbi
    traceback, reference performance timing, or data-trained parameter is used.

    Returned position and tempo are their respective posterior marginal MAPs;
    they need not form the joint MAP pair. Confidence is the marginal mass at the
    selected position bin, not a calibrated correctness probability. The final
    joint posterior is returned for diagnostic tests only. Position is never
    clamped: q >= the final chord-plan endpoint has bar_index=len(bar_starts).
    Beyond-plan bins have the same neutral potential as unknown/no-chord bins.
    The grid expands using only elapsed time and the declared tempo support;
    neither future frame times nor reference performance endpoints determine it.
    """
    config = FilterConfig() if config is None else config
    if isinstance(config, dict):
        config = FilterConfig(**config)
    if not isinstance(config, FilterConfig):
        raise ValueError("config must be FilterConfig, a configuration dict, or None")
    times, start_time = _times_and_start(times, initial_bpm, initial_position_beats, start_time)
    bars = _bars(bar_starts_beats)
    probability = np.asarray(chord_probabilities, dtype=float)
    available = np.asarray(available, dtype=bool)
    if probability.ndim != 2 or probability.shape[0] != len(times) or probability.shape[1] < 1:
        raise ValueError("chord_probabilities must have nonempty shape (T, K)")
    if available.shape != (len(times),) or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("probabilities must be finite/nonnegative, availability shape (T,)")
    mass = probability.sum(axis=1)
    if np.any(mass[available] <= 0):
        raise ValueError("available probability rows must have positive total mass")
    probability = probability / np.maximum(mass[:, None], np.finfo(float).tiny)

    starts = np.asarray(chord_starts_beats, dtype=float)
    ends = np.asarray(chord_ends_beats, dtype=float)
    ids = np.asarray(chord_ids)
    if starts.ndim != 1 or not len(starts) or ends.shape != starts.shape or ids.shape != starts.shape:
        raise ValueError("chord starts, ends and IDs must be matching nonempty vectors")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("chord intervals must be finite, nonnegative and positive length")
    if np.any(np.diff(starts) <= 0) or np.any(starts[1:] < ends[:-1] - 1e-9):
        raise ValueError("chord intervals must be ordered and nonoverlapping")
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)) or np.any(ids < -1) or np.any(ids >= probability.shape[1]):
        raise ValueError("chord IDs must be -1 or valid probability-column integers")
    ids = ids.astype(int)
    ratios = np.asarray(config.tempo_ratios, dtype=float)
    if ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios <= 0) or np.any(np.diff(ratios) <= 0):
        raise ValueError("tempo_ratios must be positive and strictly increasing")
    if config.bins_per_beat < 1 or int(config.bins_per_beat) != config.bins_per_beat:
        raise ValueError("bins_per_beat must be a positive integer")
    settings = [config.initial_tempo_ratio_std, config.tempo_neighbor_rate_per_second,
                config.emission_power_per_second, config.probability_floor_mix]
    if not np.isfinite(settings).all() or config.initial_tempo_ratio_std <= 0 or config.tempo_neighbor_rate_per_second < 0 or config.emission_power_per_second < 0:
        raise ValueError("tempo spread must be positive; diffusion and emission rates nonnegative")
    if not 0 <= config.probability_floor_mix <= 1:
        raise ValueError("probability_floor_mix must lie in [0,1]")
    if config.boundary_policy != "causal_neutral_overflow":
        raise ValueError("only causal_neutral_overflow boundary handling is supported")
    if bars[-1] >= ends[-1]:
        raise ValueError("bar starts must exclude the endpoint and lie before the chord-plan end")

    step = 1. / config.bins_per_beat
    # Allocate the known score initially; extend later without an absorbing edge.
    positions = np.arange(int(np.ceil(ends[-1] / step)) + 1, dtype=float) * step
    if initial_position_beats >= ends[-1]:
        raise ValueError("initial position must precede the score endpoint")
    interval = np.searchsorted(starts, positions, side="right") - 1
    valid = (interval >= 0) & (positions < ends[np.maximum(interval, 0)])
    state_chords = np.full(len(positions), -1, dtype=int)
    state_chords[valid] = ids[interval[valid]]
    tempi = initial_bpm * ratios
    tempo_prior = np.exp(np.maximum(-.5 * ((ratios - 1.) / config.initial_tempo_ratio_std) ** 2, -700.))
    tempo_prior /= tempo_prior.sum()
    posterior = np.zeros((len(ratios), len(positions)), dtype=float)
    initial_bin = initial_position_beats / step
    lower = int(np.floor(initial_bin))
    fraction = initial_bin - lower
    posterior[:, lower] = tempo_prior * (1. - fraction)
    if fraction:
        posterior[:, lower + 1] = tempo_prior * fraction
    reachable_bin = lower + int(fraction > 0)

    q_output = np.empty(len(times))
    tempo_output = np.empty(len(times))
    confidence = np.empty(len(times))
    tempo_mean = np.empty(len(times))
    overflow_mass = np.empty(len(times))
    previous_time = start_time
    n_classes = probability.shape[1]
    known = state_chords >= 0
    beyond_plan = positions >= ends[-1]
    for frame, now in enumerate(times):
        elapsed = now - previous_time
        if elapsed:
            # Substeps preserve valid transition probabilities across frame gaps.
            n_substeps = max(1, int(np.ceil(elapsed * config.tempo_neighbor_rate_per_second / .45)))
            dt = elapsed / n_substeps
            hop = dt * config.tempo_neighbor_rate_per_second
            for _ in range(n_substeps):
                shifts = tempi * dt / (60. * step)
                # This bound includes even tiny-probability rounding paths. Keep
                # a guard bin after it, and never inspect a future frame time.
                reachable_bin += int(np.ceil(shifts.max()))
                required = reachable_bin + 2
                if required > len(positions):
                    # Chunk allocation affects only zero-filled storage, not the
                    # model, its probabilities, or the timing of prior outputs.
                    new_size = max(required, len(positions) + 256)
                    padding = new_size - len(positions)
                    posterior = np.pad(posterior, ((0, 0), (0, padding)))
                    state_chords = np.pad(state_chords, (0, padding), constant_values=-1)
                    positions = np.arange(new_size, dtype=float) * step
                    known = state_chords >= 0
                    beyond_plan = positions >= ends[-1]
                posterior = _advance(_tempo_predict(posterior, hop, tempo_prior), shifts)
        if available[frame] and elapsed and config.emission_power_per_second:
            smooth = (1. - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
            observation = np.ones(len(positions)) / n_classes
            observation[known] = smooth[state_chords[known]]
            # Subtracting the largest log potential preserves normalized results
            # and avoids underflow when a frame arrives after a long interval.
            log_weight = np.log(np.maximum(observation, np.finfo(float).tiny))
            log_weight = (log_weight - log_weight.max()) * elapsed * config.emission_power_per_second
            posterior *= np.exp(log_weight)[None, :]
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("position filter lost all probability mass")
        posterior /= total
        position_marginal = posterior.sum(axis=0)
        tempo_marginal = posterior.sum(axis=1)
        position_i = position_marginal.argmax()
        tempo_i = tempo_marginal.argmax()
        q_output[frame] = positions[position_i]
        tempo_output[frame] = tempi[tempo_i]
        confidence[frame] = position_marginal[position_i]
        tempo_mean[frame] = tempo_marginal @ tempi
        overflow_mass[frame] = position_marginal[beyond_plan].sum()
        previous_time = now
    return {"position_beats": q_output, "tempo_bpm": tempo_output,
            "bar_index": bar_indices(q_output, bars, ends[-1]),
            "position_confidence": confidence, "mean_tempo_bpm": tempo_mean,
            "out_of_score": q_output >= ends[-1],
            "posterior_out_of_score_mass": overflow_mass,
            "final_posterior": posterior, "position_grid_beats": positions,
            "tempo_grid_bpm": tempi, "config": asdict(config)}


def checks():
    """Synthetic scientific checks, without benchmark performance annotations."""
    starts = np.arange(0., 160., 4.)
    ends = starts + 4
    ids = np.arange(len(starts)) % 7
    bars = starts.copy()
    times = np.arange(0., 40., .2)
    silent = np.ones((len(times), 7)) / 7
    silent_result = follow_positions(times, silent, np.zeros(len(times), bool), starts, ends, ids, bars, 120,
                                     config=FilterConfig(tempo_ratios=(1.,), tempo_neighbor_rate_per_second=0.))
    clock = clock_positions(times, bars, 120)
    assert np.max(np.abs(silent_result["position_beats"] - clock["position_beats"])) <= .5 + 1e-9
    assert silent_result["position_beats"][-1] > 75, "Silence must keep advancing"
    uniform = follow_positions(times, silent, np.ones(len(times), bool), starts, ends, ids, bars, 120,
                               config=FilterConfig(tempo_ratios=(1.,), tempo_neighbor_rate_per_second=0.))
    assert np.allclose(uniform["final_posterior"], silent_result["final_posterior"])

    # With several tempo candidates, uninformative input must preserve the tempo
    # prior rather than accumulate probability at the maximum/minimum BPM bounds.
    default_silence = follow_positions(times, silent, np.zeros(len(times), bool), starts, ends, ids, bars, 120)
    assert np.max(np.abs(default_silence["position_beats"] - clock["position_beats"])) <= 1.
    assert np.all(default_silence["tempo_bpm"] == 120.)
    stationary = np.exp(-.5 * ((np.asarray(FilterConfig().tempo_ratios) - 1.) / .1) ** 2)
    stationary /= stationary.sum()
    assert np.allclose(default_silence["final_posterior"].sum(axis=1), stationary)

    # A synthetic performer changes from120 to144 BPM after12 seconds; evidence
    # is generated from the untimed score solely for this implementation check.
    q = np.where(times <= 12, times * 2., 24. + (times - 12) * 2.4)
    true_ids = ids[np.minimum((q / 4).astype(int), len(ids)-1)]
    evidence = np.full((len(times), 7), .005)
    evidence[np.arange(len(times)), true_ids] = .97
    dynamic = follow_positions(times, evidence, np.ones(len(times), bool), starts, ends, ids, bars, 120)
    assert np.mean(np.abs(dynamic["position_beats"][-50:] - q[-50:])) < np.mean(np.abs(clock["position_beats"][-50:] - q[-50:]))
    assert np.mean(dynamic["mean_tempo_bpm"][-30:]) > 125, "Tempo should adapt to observations"
    cut = 113
    prefix = follow_positions(times[:cut], evidence[:cut], np.ones(cut, bool), starts, ends, ids, bars, 120)
    assert np.array_equal(prefix["position_beats"], dynamic["position_beats"][:cut]), "Future evidence leakage"
    assert np.array_equal(prefix["tempo_bpm"], dynamic["tempo_bpm"][:cut]), "Future tempo leakage"
    assert np.count_nonzero(dynamic["final_posterior"] > 1e-8) > 1, "Keep multiple state hypotheses"
    assert np.isclose(dynamic["final_posterior"].sum(), 1.)
    assert np.array_equal(bar_indices([0, 3.99, 4, 8], [0, 4, 8]), [0, 0, 1, 2])

    # Regression: uncertainty must not pile up at a short score's endpoint.
    # Continue for well beyond that endpoint, without supplying an end time.
    short_starts = np.arange(0., 32., 4.)
    overflow = follow_positions(times, silent, np.zeros(len(times), bool),
                                short_starts, short_starts + 4.,
                                np.arange(len(short_starts)) % 7, short_starts, 120)
    assert np.max(np.abs(overflow["position_beats"] - clock["position_beats"])) <= 1.
    assert overflow["position_beats"][-1] > 75., "Do not absorb at score end32"
    assert overflow["bar_index"][-1] == len(short_starts), "Overflow cannot count as last bar"
    assert overflow["out_of_score"][-1]
    assert overflow["posterior_out_of_score_mass"][-1] > .99
    assert np.all(overflow["final_posterior"][:, -1] == 0), "Keep zero-mass guard bin"
    assert np.isclose(overflow["final_posterior"].sum(), 1.)
    # Changing the score's finite endpoint cannot change an unobserved trajectory.
    assert np.array_equal(overflow["position_beats"], default_silence["position_beats"])
    cut = 141  # After both the score endpoint and online grid expansion.
    overflow_prefix = follow_positions(times[:cut], silent[:cut], np.zeros(cut, bool),
                                      short_starts, short_starts + 4.,
                                      np.arange(len(short_starts)) % 7, short_starts, 120)
    assert np.array_equal(overflow_prefix["position_beats"], overflow["position_beats"][:cut])
    assert np.array_equal(overflow_prefix["bar_index"], overflow["bar_index"][:cut])
    clock_overflow = clock_positions(times, short_starts, 120, score_end_beats=32.)
    assert clock_overflow["position_beats"][-1] == clock["position_beats"][-1]
    assert clock_overflow["bar_index"][-1] == len(short_starts)
    assert np.array_equal(bar_indices([-1., 0., 31.99, 32., 100.], short_starts, 32.),
                          [-1, 0, 7, 8, 8])
    try:
        _advance(np.array([[0., 1.]]), np.array([1.]))
    except ValueError:
        pass
    else:
        raise AssertionError("Undersized grids must not silently accumulate/drop probability")
    return {"silence_max_error_beats": float(np.max(np.abs(silent_result["position_beats"] - clock["position_beats"]))),
            "changed_tempo_final_mean_bpm": float(dynamic["mean_tempo_bpm"][-1]),
            "changed_tempo_last_10s_mean_error_beats": float(np.mean(np.abs(dynamic["position_beats"][-50:] - q[-50:]))),
            "overflow_final_position_beats": float(overflow["position_beats"][-1]),
            "overflow_score_end_beats": 32.,
            "boundary_policy": FilterConfig().boundary_policy}


if __name__ == "__main__":
    started = time.perf_counter()
    print(checks())
    print(f"Position-filter checks passed in {time.perf_counter()-started:.3f}s")
