"""Separate melody-study copy of the continuous bootstrap particle follower.

The source in improvement_study/particle_follower.py is preserved unchanged.
This copy adds only an optional causal, position-specific melodic observation.
With that observation disabled or identically zero, its numeric outputs and RNG
history exactly reproduce the original follower under the same configuration.
The p02 configuration is initial_tempo_ratio_std=.3 and
tempo_diffusion_per_sqrt_second=.003; those must be supplied explicitly, as in
the previous study. Existing defaults are retained for reproducibility.

Particles represent current (unfolded beat position, BPM). Initial BPM is sampled
from a truncated normal centered on the supplied catalog tempo; there is no
special point mass at the catalog value. BPM follows a Gaussian random walk,
with standard deviation proportional to the previous BPM and sqrt(elapsed time),
reflected at fixed ratios of the initial BPM. Continuous q advances using the
average of the old and new BPM. No positional noise, score snapping, truth-based
jumps, future timing, backward smoothing, or endpoint clamping is used.

The untimed chord plan, synchronized initial position/time, approximate starting
BPM, causal chord compatibility, and optional melodic potentials are the inputs.
Chord emissions and elapsed-time tempering match the other benchmark followers.
Unknown chords, gaps, and positions beyond the plan receive uniform 1/K evidence.
Their positions remain beyond the plan and are evaluated as out of score.

Sequential importance weights are normalized in log space. Systematic resampling
occurs only when ESS drops below a fixed fraction of the particle count. The RNG
seed is explicit; a prefix run uses exactly the same random draws as that prefix
of a full run. The default decision selects the bar with most particle weight
and reports its conditional mean position. BPM is the posterior particle mean.

These are declared exploratory baseline choices, not a reproduction or novelty
claim. Finite-particle depletion, sampling variability, bounded-tempo support,
and the supplied unfolded plan limit the tracker. Silence propagates uncertainty
and has Monte Carlo variability; it is not forced to match a perfect clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleConfig:
    n_particles: int = 1024
    seed: int = 0
    tempo_ratio_bounds: tuple = (.5, 1.8)
    initial_tempo_ratio_std: float = .15
    tempo_diffusion_per_sqrt_second: float = .01
    max_prediction_step_seconds: float = .25
    resample_ess_fraction: float = .5
    emission_power_per_second: float = 1.
    probability_floor_mix: float = .02
    output_rule: str = "bar"
    boundary_policy: str = "continuous_neutral_overflow"
    melodic_power_per_second: float = 0.


FilterConfig = ParticleConfig


def _reflect(values, lower, upper):
    """Reflect arbitrarily large excursions without accumulating boundary mass."""
    width = upper - lower
    folded = np.remainder(values - lower, 2. * width)
    return lower + np.where(folded <= width, folded, 2. * width - folded)


def _systematic_indices(weights, rng):
    points = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.  # protect against floating-point cumulative-sum error
    return np.searchsorted(cumulative, points, side="right")


def _bar_indices(position, bars, score_end):
    result = np.searchsorted(bars, position, side="right") - 1
    return np.where(position >= score_end, len(bars), result)


def _initial_tempi(rng, n, bpm, spread, bounds):
    """Exact rejection sampling of the declared truncated-normal prior."""
    ratios = rng.normal(1., spread, n)
    rejected = (ratios < bounds[0]) | (ratios > bounds[1])
    while np.any(rejected):
        ratios[rejected] = rng.normal(1., spread, int(rejected.sum()))
        rejected = (ratios < bounds[0]) | (ratios > bounds[1])
    return ratios * bpm


def follow_positions(times, chord_probabilities, available, chord_starts_beats,
                     chord_ends_beats, chord_ids, bar_starts_beats, initial_bpm, *,
                     initial_position_beats=0., start_time=None, config=None,
                     melodic_logscore=None):
    """Estimate current position from a causal continuous-state particle filter.

    The call signature matches the shared benchmark follower. Probability rows
    have shape (T,K); false availability skips observation updates. All-zero rows
    are permitted only when unavailable. IDs are chord-column indices or -1.
    Bar starts exclude the endpoint. Config is ParticleConfig, dict, or None.

    Diffusion is a fractional BPM standard deviation per sqrt(second): .01 means
    about 1.2 BPM over one second at 120 BPM before boundary reflection. Prediction
    gaps are split into at most .25-second steps. Emission log compatibility is
    multiplied by elapsed seconds * emission_power_per_second. A probability
    floor of .02 is mixed uniformly; exact zeros remaining with floor=0 receive
    machine-tiny probability, consistently with the other benchmark followers.

    Outputs are computed before any frame's resampling. ``position_confidence``
    is mass in the selected bar, not a calibrated correctness probability.
    ``effective_sample_size`` is measured before resampling. Final particles and
    weights are post-resampling diagnostic state, not retrospective predictions.

    Optional melodic_logscore(frame_index, time_seconds, position_beats,
    tempo_bpm) returns a finite vector with one log evidence potential per
    particle, or None/scalar zero for an unavailable frame. Position and tempo arguments
    are read-only vectors. The callback is evaluated at the current frame after
    propagation, independently of harmonic availability; it must use only
    observations available at that time. The filter cannot enforce a callback's
    internal causality. Prefix checks must also cover the chosen matcher.

    Zero means neutral evidence relative to a declared null/background model;
    positive and negative values are allowed. Scores must have a fixed scale
    across frames and songs. An arbitrary negative distance with zero for a
    missing reference is unsuitable: it systematically favors missing-reference
    positions. The provider must return zero for unavailable candidates. This
    filter also forces beyond-plan candidates to zero. No score normalization
    over chart positions or conversion of distances to likelihoods is invented
    here. A likelihood-ratio interpretation remains the provider's assumption,
    not a claim of calibrated likelihoods.

    The extra update is elapsed_seconds * melodic_power_per_second * logscore,
    before the original decision and resampling. All-zero or constant effective
    potentials, after out-of-score masking, skip that update exactly. Power zero
    never calls the provider. Harmonic-only
    uses power zero; melody-only sets emission_power_per_second=0; fused keeps
    both powers fixed. No automatic reliability gate is added.
    """
    config = ParticleConfig() if config is None else config
    if isinstance(config, dict):
        config = ParticleConfig(**config)
    if not isinstance(config, ParticleConfig):
        raise ValueError("config must be ParticleConfig, dict, or None")
    if melodic_logscore is not None and not callable(melodic_logscore):
        raise ValueError("melodic_logscore must be callable or None")
    if not np.isfinite(config.melodic_power_per_second) or config.melodic_power_per_second < 0:
        raise ValueError("melodic_power_per_second must be finite and nonnegative")
    if not np.isfinite(config.n_particles) or config.n_particles < 2 or int(config.n_particles) != config.n_particles:
        raise ValueError("n_particles must be an integer of at least two")
    if not isinstance(config.seed, (int, np.integer)) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bounds = np.asarray(config.tempo_ratio_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < 1 < bounds[1]:
        raise ValueError("tempo_ratio_bounds must contain positive lower < 1 < upper")
    settings = [config.initial_tempo_ratio_std, config.tempo_diffusion_per_sqrt_second,
                config.max_prediction_step_seconds, config.resample_ess_fraction,
                config.emission_power_per_second, config.probability_floor_mix]
    if not np.isfinite(settings).all() or settings[0] <= 0 or settings[1] < 0 or settings[2] <= 0 or not 0 < settings[3] <= 1 or settings[4] < 0 or not 0 <= settings[5] <= 1:
        raise ValueError("invalid particle dynamics, resampling, or observation settings")
    if config.output_rule not in ("bar", "mean"):
        raise ValueError("output_rule must be bar or mean")
    if config.boundary_policy != "continuous_neutral_overflow":
        raise ValueError("only continuous_neutral_overflow is supported")
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
    if bars.ndim != 1 or not len(bars) or not np.isfinite(bars).all() or np.any(np.diff(bars) <= 0) or bars[-1] >= ends[-1] or bars[0] > initial_position_beats:
        raise ValueError("bar starts must be ordered, begin by initial position, and exclude score end")
    if initial_position_beats >= ends[-1]:
        raise ValueError("initial position must precede score end")
    ids = ids.astype(int)
    n = int(config.n_particles)
    rng = np.random.default_rng(config.seed)
    tempo = _initial_tempi(rng, n, initial_bpm, config.initial_tempo_ratio_std, bounds)
    position = np.full(n, initial_position_beats, dtype=float)
    weights = np.full(n, 1. / n)
    lower, upper = initial_bpm * bounds
    q_output = np.empty(len(times))
    tempo_output = np.empty(len(times))
    confidence = np.empty(len(times))
    effective_n = np.empty(len(times))
    resampled = np.zeros(len(times), dtype=bool)
    overflow_mass = np.empty(len(times))
    previous_time = start_time
    n_classes = probability.shape[1]
    melodic_callback_calls = 0
    melodic_update_frames = 0
    for frame, now in enumerate(times):
        elapsed = now - previous_time
        if elapsed:
            substeps = max(1, int(np.ceil(elapsed / config.max_prediction_step_seconds)))
            dt = elapsed / substeps
            for _ in range(substeps):
                if config.tempo_diffusion_per_sqrt_second:
                    next_tempo = _reflect(tempo + config.tempo_diffusion_per_sqrt_second * tempo * np.sqrt(dt) * rng.standard_normal(n), lower, upper)
                else:
                    next_tempo = tempo
                position += (tempo + next_tempo) * dt / 120.
                tempo = next_tempo
        if available[frame] and elapsed and config.emission_power_per_second:
            interval = np.searchsorted(starts, position, side="right") - 1
            valid = (interval >= 0) & (position < ends[np.maximum(interval, 0)])
            chord = np.full(n, -1, dtype=int)
            chord[valid] = ids[interval[valid]]
            smooth = (1. - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
            potential = np.full(n, 1. / n_classes)
            known = chord >= 0
            potential[known] = smooth[chord[known]]
            log_weight = np.log(np.maximum(weights, np.finfo(float).tiny))
            log_weight += np.log(np.maximum(potential, np.finfo(float).tiny)) * elapsed * config.emission_power_per_second
            weights = np.exp(log_weight - np.max(log_weight))
            weights /= weights.sum()
        if melodic_logscore is not None and elapsed and config.melodic_power_per_second:
            candidate_position, candidate_tempo = position.view(), tempo.view()
            candidate_position.flags.writeable = False
            candidate_tempo.flags.writeable = False
            scores = melodic_logscore(frame, float(now), candidate_position, candidate_tempo)
            melodic_callback_calls += 1
            if scores is not None:
                scores = np.asarray(scores, dtype=float)
                if scores.shape == () and scores == 0.:
                    scores = np.zeros(n)
                if scores.shape != (n,) or not np.isfinite(scores).all():
                    raise ValueError("melodic_logscore must return a finite particle-length vector, zero, or None")
                scores = np.where(position >= ends[-1], 0., scores)
                if np.any(scores != scores[0]):
                    # A common log factor cancels. Skipping constant scores also
                    # preserves the original floating-point and RNG trajectory.
                    log_weight = np.log(np.maximum(weights, np.finfo(float).tiny))
                    log_weight += (scores - scores.max()) * elapsed * config.melodic_power_per_second
                    weights = np.exp(log_weight - np.max(log_weight))
                    weights /= weights.sum()
                    melodic_update_frames += 1
        particle_bars = _bar_indices(position, bars, ends[-1])
        bar_mass = np.bincount(particle_bars, weights=weights, minlength=len(bars)+1)
        if config.output_rule == "bar":
            selected_bar = int(bar_mass.argmax())
            selected = particle_bars == selected_bar
            q_output[frame] = (weights[selected] @ position[selected]) / bar_mass[selected_bar]
        else:
            q_output[frame] = weights @ position
            selected_bar = int(_bar_indices(q_output[frame], bars, ends[-1]))
        tempo_output[frame] = weights @ tempo
        confidence[frame] = bar_mass[selected_bar]
        overflow_mass[frame] = bar_mass[-1]
        effective_n[frame] = 1. / (weights @ weights)
        if effective_n[frame] < n * config.resample_ess_fraction:
            indices = _systematic_indices(weights, rng)
            position, tempo = position[indices].copy(), tempo[indices].copy()
            weights.fill(1. / n)
            resampled[frame] = True
        previous_time = now
    return {"position_beats": q_output, "tempo_bpm": tempo_output,
            "mean_tempo_bpm": tempo_output.copy(), "bar_index": _bar_indices(q_output, bars, ends[-1]),
            "position_confidence": confidence, "out_of_score": q_output >= ends[-1],
            "posterior_out_of_score_mass": overflow_mass, "effective_sample_size": effective_n,
            "resampled": resampled, "resampling_count": int(resampled.sum()),
            "final_particle_positions": position, "final_particle_tempo_bpm": tempo,
            "final_particle_weights": weights, "config": asdict(config),
            "melodic_callback_calls": melodic_callback_calls,
            "melodic_update_frames": melodic_update_frames,
            "implementation": "melody_observation_bootstrap_particle_filter"}


def checks():
    """Analytical/synthetic checks; no real annotation selects parameters."""
    times = np.arange(0., 60., .2)
    starts = np.arange(0., 160., 4.)
    ends, ids = starts + 4., np.arange(len(starts)) % 7
    uniform = np.ones((len(times), 7)) / 7.
    available = np.ones(len(times), bool)
    args = (times, uniform, np.zeros(len(times), bool), starts, ends, ids, starts, 120.)
    silence = follow_positions(*args, config={"output_rule":"mean"})
    clock = times * 2.
    assert np.max(np.abs(silence["position_beats"] - clock)) < 3.
    assert np.all(np.diff(silence["position_beats"]) > 0)
    assert not np.any(silence["resampled"])
    assert np.allclose(silence["effective_sample_size"], 1024.)
    same = follow_positions(times, uniform, available, starts, ends, ids, starts, 120., config={"output_rule":"mean"})
    assert np.allclose(same["position_beats"], silence["position_beats"], atol=1e-11, rtol=0)
    # Sufficiently distinctive synthetic chords provide a tempo-change signal.
    truth_q = np.where(times <= 12., times * 2., 24. + (times - 12.) * 2.4)
    truth_ids = np.floor(truth_q / 4.).astype(int) % 7
    probs = np.full((len(times), 7), .001)
    probs[np.arange(len(times)), truth_ids] = .994
    config = {"tempo_diffusion_per_sqrt_second":.03}
    dynamic_args = (times, probs, available, starts, ends, ids, starts, 120.)
    dynamic = follow_positions(*dynamic_args, config=config)
    tracking_error = np.mean(np.abs(dynamic["position_beats"][-50:] - truth_q[-50:]))
    clock_error = np.mean(np.abs(clock[-50:] - truth_q[-50:]))
    assert tracking_error < clock_error
    assert dynamic["resampling_count"] > 0
    assert np.all(dynamic["effective_sample_size"] >= 1.-1e-9)
    assert np.all(dynamic["effective_sample_size"] <= 1024.+1e-8)
    assert np.all(dynamic["effective_sample_size"][dynamic["resampled"]] < 512.)
    # Prefixes preserve the seed's entire random-draw history, including resampling.
    cut = 130
    prefix = follow_positions(times[:cut], probs[:cut], available[:cut], starts, ends, ids, starts, 120., config=config)
    for key in ("position_beats", "tempo_bpm", "bar_index", "effective_sample_size", "resampled"):
        assert np.array_equal(prefix[key], dynamic[key][:cut]), key
    repeated = follow_positions(*dynamic_args, config=config)
    assert np.array_equal(repeated["position_beats"], dynamic["position_beats"])
    # Exact-zero chord likelihoods remain numerically defined; invalid all-zero
    # available input rows are rejected explicitly instead of silently repaired.
    onehot = np.eye(7)[truth_ids]
    zero = follow_positions(times, onehot, available, starts, ends, ids, starts, 120., config={"probability_floor_mix":0.})
    assert np.isfinite(zero["position_beats"]).all()
    try:
        follow_positions(times, np.zeros_like(uniform), available, starts, ends, ids, starts, 120.)
    except ValueError:
        pass
    else:
        raise AssertionError("All-zero available rows must be rejected")
    # A short untimed score must not trap silent particles at its endpoint.
    overflow = follow_positions(times, uniform, np.zeros(len(times), bool), starts[:2], ends[:2], ids[:2], starts[:2], 120., config={"output_rule":"mean"})
    assert np.array_equal(overflow["position_beats"], silence["position_beats"])
    assert overflow["out_of_score"][-1] and overflow["bar_index"][-1] == 2
    assert np.all((dynamic["final_particle_tempo_bpm"] >= 60.) & (dynamic["final_particle_tempo_bpm"] <= 216.))
    reflected = _reflect(np.array([-100., 50., 60., 120., 216., 250., 1000.]), 60., 216.)
    assert np.all((reflected >= 60.) & (reflected <= 216.))
    assert np.array_equal(_systematic_indices(np.full(16,1./16),np.random.default_rng(0)),np.arange(16))
    return {"prefix_invariance":True, "seed_repeatability":True, "silence_no_resampling":True,
            "silence_max_error_beats":float(np.max(np.abs(silence["position_beats"]-clock))),
            "tempo_change_last_10s_mae_beats":float(tracking_error),
            "clock_last_10s_mae_beats":float(clock_error),
            "tempo_change_resamplings":dynamic["resampling_count"],
            "zero_likelihood_finite":True, "neutral_overflow":True,
            "tempo_bounds":True, "systematic_resampling":True}


def melody_checks():
    """Reproduction and causal-hook checks; synthetic observations only.

    Import the preserved original only for a direct equality check. Production
    inference has no dependency on that module, its globals, or any annotation.
    """
    import importlib.util
    from pathlib import Path
    import sys

    source = Path(__file__).resolve().parent.parent / "improvement_study" / "particle_follower.py"
    module_name = "_melody_check_original_particle_follower"
    spec = importlib.util.spec_from_file_location(module_name, source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = original
    spec.loader.exec_module(original)
    times = np.arange(0., 24., .2)
    starts = np.arange(0., 160., 4.)
    ids = np.arange(len(starts)) % 7
    rng = np.random.default_rng(315)
    probability = rng.dirichlet(np.ones(7), len(times))
    available = rng.random(len(times)) > .2
    args = (times, probability, available, starts, starts + 4., ids, starts, 120.)
    p02 = {"initial_tempo_ratio_std": .3, "tempo_diffusion_per_sqrt_second": .003}
    baseline = original.follow_positions(*args, config=p02)
    numeric_keys = [key for key, value in baseline.items() if isinstance(value, np.ndarray)]

    def assert_equal(actual, expected):
        for key in numeric_keys:
            assert np.array_equal(actual[key], expected[key]), key
        assert actual["resampling_count"] == expected["resampling_count"]

    disabled = follow_positions(*args, config=p02)
    assert_equal(disabled, baseline)

    def should_not_run(*_args):
        raise AssertionError("Power zero must not call the provider")

    assert_equal(follow_positions(*args, config=p02, melodic_logscore=should_not_run), baseline)
    melody_config = {**p02, "melodic_power_per_second": 2.}
    neutral = follow_positions(*args, config=melody_config,
                               melodic_logscore=lambda frame, now, q, bpm: np.zeros_like(q))
    assert_equal(neutral, baseline)
    assert neutral["melodic_callback_calls"] == len(times) - 1
    assert neutral["melodic_update_frames"] == 0
    missing = follow_positions(*args, config=melody_config, melodic_logscore=lambda *_args: None)
    assert_equal(missing, baseline)
    scalar_zero = follow_positions(*args, config=melody_config, melodic_logscore=lambda *_args: 0.)
    assert_equal(scalar_zero, baseline)

    def make_provider(events):
        def score(frame, now, q, bpm):
            assert now == times[frame]
            assert not q.flags.writeable and not bpm.flags.writeable
            event = np.searchsorted(events, now, side="right") - 1
            if event < 0:
                return None
            # Synthetic phrase identity supplies the event's known score index;
            # only already observed event times enter the candidate comparison.
            expected_q = event + (now - events[event]) * bpm / 60.
            return 2. * np.exp(-.5 * ((q - expected_q) / .6) ** 2) - .25
        return score

    events = np.arange(0., 30., .46)
    fused = follow_positions(*args, config=melody_config, melodic_logscore=make_provider(events))
    assert fused["melodic_update_frames"] > 0
    assert not np.array_equal(fused["position_beats"], baseline["position_beats"])
    cut = 73
    prefix_args = (times[:cut], probability[:cut], available[:cut], *args[3:])
    prefix = follow_positions(*prefix_args, config=melody_config,
                              melodic_logscore=make_provider(events[events <= times[cut - 1]]))
    for key in ("position_beats", "tempo_bpm", "bar_index", "effective_sample_size", "resampled"):
        assert np.array_equal(prefix[key], fused[key][:cut]), key

    # Melody is an independent observation stream, not gated by chord activity.
    no_harmony_args = (times, probability, np.zeros(len(times), bool), *args[3:])
    melody_only = follow_positions(*no_harmony_args,
                                  config={**melody_config, "emission_power_per_second": 0.},
                                  melodic_logscore=make_provider(events))
    assert melody_only["melodic_update_frames"] == len(times) - 1
    assert melody_only["resampling_count"] > 0

    # Beyond-score rewards are neutralized, never folded into the final bar.
    short_args = (times, probability, available, starts[:2], starts[:2] + 4., ids[:2], starts[:2], 120.)
    short_baseline = original.follow_positions(*short_args, config=p02)
    overflow = follow_positions(*short_args, config=melody_config,
                                melodic_logscore=lambda frame, now, q, bpm: np.where(q >= 8., 7., 0.))
    assert_equal(overflow, short_baseline)
    assert overflow["melodic_update_frames"] == 0

    for provider in (lambda frame, now, q, bpm: 1.,
                     lambda frame, now, q, bpm: np.full_like(q, np.nan)):
        try:
            follow_positions(*args, config=melody_config, melodic_logscore=provider)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid provider output must be rejected")
    return {"p02_exact_reproduction": True, "disabled_callback_not_called": True,
            "neutral_exact_reproduction": True, "missing_exact_reproduction": True,
            "scalar_zero_exact_reproduction": True,
            "melodic_prefix_invariance": True, "independent_harmonic_availability": True,
            "readonly_candidate_views": True, "neutral_overflow": True,
            "invalid_callback_output_rejected": True,
            "melody_only_resamplings": melody_only["resampling_count"]}


if __name__ == "__main__":
    import json
    print(json.dumps({"inherited_particle_checks": checks(), "melodic_hook_checks": melody_checks()}, indent=2))
