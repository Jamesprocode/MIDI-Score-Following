"""Causal, annotation-free MIDI onset compatibility for a position/tempo grid.

This is a deliberately simple benchmark heuristic, not a published beat tracker
or a calibrated observation model. For candidate position q at frame time t and
candidate tempo b, an observed onset u has beat phase q - b*(t-u)/60. Its soft
compatibility is K(phase) = cos(pi*phase)**8 + r*sin(pi*phase)**8. The default
r=.35 makes quarter-note peaks stronger than half-beat peaks, while allowing
syncopation. The finite Fourier expansion evaluates this kernel efficiently.

Recent onset compatibilities are averaged, using weight 1 for bass streams and
.25 for other pitched streams. Exact simultaneous onsets within one instrument
are counted once. Drum/percussion streams are ignored by default. These are
fixed declared choices, not parameters selected from test timing annotations.
Only onset timestamps and stream names are read; pitches, note offsets, score
chords, meter, and performance annotations do not enter this observation.

The output is a bounded relative log *potential*, not a normalized likelihood:
subtract the best state score and multiply by a [0,1] evidence-amount/freshness
factor. The caller should add elapsed_seconds * strength * log_likelihood to
the filtering log posterior. Overlapping windows reuse events; that per-second
tempering is essential, and does not make the observations independent. A
conservative initial strength is 1/window_seconds. Other fixed strengths are
explicit sensitivity experiments, not fitted likelihood calibrations.

This cue is invariant to adding whole beats to q. It provides no bar/downbeat
identity and cannot resolve all half/double-tempo ambiguities. Offbeat-only
playing can favor a shifted beat phase; a bass beat mixed with horn offbeats is
less ambiguous. Silence becomes exactly neutral after two approximate beats.
An abrupt tempo change can take the window length to replace old evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np


@dataclass
class RhythmEvidence:
    """Prepared causal evidence; arrays use frame, tempo, harmonic ordering.

    ``available`` and ``reliability`` have shape (T,). Reliability is only a
    deterministic evidence-amount/freshness gate, not a probability of correct
    tracking. ``beat_coherence`` / ``subdivision_coherence`` have shape (T,M)
    and diagnose periodic concentration at each supplied candidate tempo.
    ``moments`` has shape (T,M,4), avoiding a large T x M x Q stored tensor.
    """

    times: np.ndarray
    tempo_bpms: np.ndarray
    moments: np.ndarray
    available: np.ndarray
    reliability: np.ndarray
    event_count: np.ndarray
    onset_age_seconds: np.ndarray
    beat_coherence: np.ndarray
    subdivision_coherence: np.ndarray
    metadata: dict
    _coefficients: np.ndarray = field(repr=False)
    _constant: float = field(repr=False)
    _cached_grid: np.ndarray | None = field(default=None, repr=False)
    _cached_basis: np.ndarray | None = field(default=None, repr=False)
    _cached_inverse: np.ndarray | None = field(default=None, repr=False)

    def _phase_basis(self, q_grid):
        q = np.asarray(q_grid, dtype=float)
        if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
            raise ValueError("q_grid must be a nonempty finite vector of beat positions")
        # The filter normally reuses its grid, occasionally extending its end.
        if self._cached_grid is None or not np.array_equal(q, self._cached_grid):
            phase, inverse = np.unique(np.remainder(q, 1.), return_inverse=True)
            self._cached_grid = q.copy()
            self._cached_basis = np.exp(2j * np.pi * np.arange(1, 5)[:, None] * phase)
            self._cached_inverse = inverse
        return self._cached_basis, self._cached_inverse

    def compatibility(self, frame_index, q_grid):
        """Return (M,Q) weighted mean kernel scores in [0,1], before gating.

        A frame with no recent onsets has a constant zero matrix. Otherwise this
        may still be an unavailable frame (too few events or too much silence);
        filtering should use ``log_likelihood``, which applies the gate.
        """
        if not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < len(self.times):
            raise IndexError("frame_index must index a prepared frame")
        basis, inverse = self._phase_basis(q_grid)
        if not self.event_count[frame_index]:
            return np.zeros((len(self.tempo_bpms), len(inverse)))
        # Exact four-term Fourier expansion, apart from floating-point roundoff.
        score = self._constant + np.real(
            (self.moments[frame_index] * self._coefficients) @ basis)
        return np.clip(score, 0., 1.)[:, inverse]

    def log_likelihood(self, frame_index, q_grid):
        """Return (M,Q) relative log potentials in [-1,0].

        q is the candidate position *at times[frame_index]*. The same maximum
        is subtracted across all tempos and positions, so tempo evidence is
        retained. Entirely unavailable frames return exactly zero.
        """
        score = self.compatibility(frame_index, q_grid)
        if not self.available[frame_index]:
            return np.zeros_like(score)
        return self.reliability[frame_index] * (score - np.max(score))


def _weighted_onsets(notes_by_instrument, instrument_weights):
    if not isinstance(notes_by_instrument, Mapping):
        raise ValueError("notes_by_instrument must map stream names to note arrays")
    overrides = {} if instrument_weights is None else dict(instrument_weights)
    if any(not np.isfinite(w) or w < 0 for w in overrides.values()):
        raise ValueError("instrument weights must be finite and nonnegative")
    onsets, weights, audit = [], [], {}
    for name, notes in notes_by_instrument.items():
        lower = str(name).lower()
        default = 0. if ("drum" in lower or "percussion" in lower) else (1. if "bass" in lower else .25)
        weight = float(overrides.get(name, default))
        notes = np.asarray(notes)
        if notes.size == 0:
            event_times = np.empty(0)
        else:
            if notes.ndim != 2 or notes.shape[1] < 1:
                raise ValueError(f"notes for {name!r} must have onset times in column zero")
            # Do not inspect offset columns: even future note-offs cannot affect
            # any earlier output. Input may represent notes still sounding.
            event_times = np.asarray(notes[:, 0], dtype=float)
            if not np.isfinite(event_times).all():
                raise ValueError(f"nonfinite onset time in {name!r}")
            event_times = np.unique(event_times)
        audit[str(name)] = {"weight": weight, "distinct_onsets": len(event_times)}
        if weight > 0 and len(event_times):
            onsets.append(event_times)
            weights.append(np.full(len(event_times), weight))
    if not onsets:
        return np.empty(0), np.empty(0), audit
    onsets, weights = np.concatenate(onsets), np.concatenate(weights)
    event_times, inverse = np.unique(onsets, return_inverse=True)
    return event_times, np.bincount(inverse, weights=weights), audit


def prepare_rhythm(notes_by_instrument, times, tempo_bpms, approx_bpm, *,
                   window_beats=8., window_seconds=None, instrument_weights=None,
                   offbeat_weight=.35, minimum_onsets=3, minimum_span_beats=1.,
                   silence_beats=2.):
    """Prepare causal onset moments for later dynamic-grid filtering.

    Notes are arrays with onset seconds in column zero, conventionally
    (onset, offset, MIDI pitch). Frames are strictly increasing absolute seconds;
    tempi are a positive vector of candidate BPMs in the caller's grid order.
    approx_bpm only converts fixed beat-based window/freshness settings to
    seconds. It supplies no observed phase or true played tempo.

    Window membership is (t-W,t], never a future onset. With at least three
    distinct onset times spanning one approximate beat, reliability is
    min(1,count/4) times freshness. Freshness is one through one approximate beat
    of silence, then decreases linearly to zero at silence_beats (default two).
    All thresholds use supplied approximate BPM, never performance annotations.
    No meter-specific accents are assumed: q and BPM use quarter-note beats.
    """
    times = np.asarray(times, dtype=float)
    tempi = np.asarray(tempo_bpms, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a nonempty finite strictly increasing vector")
    if tempi.ndim != 1 or not len(tempi) or not np.isfinite(tempi).all() or np.any(tempi <= 0):
        raise ValueError("tempo_bpms must be a nonempty positive finite vector")
    if not np.isfinite(approx_bpm) or approx_bpm <= 0:
        raise ValueError("approx_bpm must be finite and positive")
    if not np.isfinite(window_beats) or window_beats <= 0:
        raise ValueError("window_beats must be finite and positive")
    window = window_beats * 60. / approx_bpm if window_seconds is None else float(window_seconds)
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not np.isfinite(offbeat_weight) or not 0 <= offbeat_weight <= 1:
        raise ValueError("offbeat_weight must lie in [0,1]")
    if int(minimum_onsets) != minimum_onsets or minimum_onsets < 2:
        raise ValueError("minimum_onsets must be an integer of at least two")
    if not np.isfinite(minimum_span_beats) or minimum_span_beats < 0:
        raise ValueError("minimum_span_beats must be finite and nonnegative")
    if not np.isfinite(silence_beats) or silence_beats <= 1:
        raise ValueError("silence_beats must be finite and exceed one")

    event_times, weights, audit = _weighted_onsets(notes_by_instrument, instrument_weights)
    # Each difference of prefix sums accesses events only through the frame's
    # right endpoint. Preparing all frames together does not use future events.
    right = np.searchsorted(event_times, times, side="right")
    left = np.searchsorted(event_times, times - window, side="right")
    count = right - left
    prefix_mass = np.concatenate(([0.], np.cumsum(weights)))
    mass = prefix_mass[right] - prefix_mass[left]
    moments = np.zeros((len(times), len(tempi), 4), dtype=complex)
    # Looping over the small tempo dimension limits temporary memory.
    harmonics = np.arange(1, 5)
    for j, bpm in enumerate(tempi):
        omega = 2. * np.pi * bpm / 60. * harmonics
        event_terms = weights[:, None] * np.exp(1j * event_times[:, None] * omega)
        prefix = np.concatenate((np.zeros((1, 4), complex), np.cumsum(event_terms, axis=0)), axis=0)
        sums = (prefix[right] - prefix[left]) * np.exp(-1j * times[:, None] * omega)
        moments[:, j, :] = np.divide(sums, mass[:, None], out=np.zeros_like(sums), where=mass[:, None] > 0)

    span = np.zeros(len(times))
    age = np.full(len(times), np.inf)
    nonempty = count > 0
    span[nonempty] = event_times[right[nonempty]-1] - event_times[left[nonempty]]
    age[nonempty] = np.maximum(0., times[nonempty] - event_times[right[nonempty]-1])
    age_beats = age * approx_bpm / 60.
    freshness = np.clip((silence_beats - age_beats) / (silence_beats - 1.), 0., 1.)
    enough = (count >= minimum_onsets) & (span * approx_bpm / 60. >= minimum_span_beats - 1e-12)
    reliability = enough * np.minimum(1., count / 4.) * freshness
    available = reliability > 0
    coefficients = np.array([56.*(1.-offbeat_weight), 28.*(1.+offbeat_weight),
                             8.*(1.-offbeat_weight), 1.*(1.+offbeat_weight)]) / 128.
    metadata = {
        "method": "causal_weighted_onset_phase_kernel", "version": 1,
        "window_seconds": window, "requested_window_beats": float(window_beats),
        "approx_bpm": float(approx_bpm), "offbeat_weight": float(offbeat_weight),
        "kernel": "cos(pi*phase)^8 + offbeat_weight*sin(pi*phase)^8",
        "minimum_onsets": int(minimum_onsets), "minimum_span_beats": float(minimum_span_beats),
        "silence_beats": float(silence_beats), "streams": audit,
        "uses_note_offsets": False, "uses_annotations": False,
        "claims_downbeat_detection": False,
        "recommended_conservative_strength_per_second": 1. / window,
        "overlap": "Multiply log potential by elapsed seconds and fixed declared strength; windows overlap.",
    }
    return RhythmEvidence(times.copy(), tempi.copy(), moments, available, reliability, count,
                          age, np.abs(moments[:, :, 0]), np.abs(moments[:, :, 1]), metadata,
                          coefficients, 35.*(1.+offbeat_weight)/128.)


def checks():
    """Synthetic checks of causality, phase origin, syncopation, and silence."""
    def notes(onsets):
        onsets = np.asarray(onsets, dtype=float)
        return np.column_stack((onsets, onsets + .1, np.full(len(onsets), 48.)))

    times = np.arange(0., 8.01, .1)
    tempi = np.array([90., 120., 150.])
    q = np.arange(0., 4., .125)
    bass = notes(np.arange(0., 5.01, .5))
    evidence = prepare_rhythm({"bass": bass}, times, tempi, 120.)
    frame = 32  # now=3.2, so a 120 BPM beat grid has q phase .4.
    actual_q = np.array([.4, .9, .65])
    s = evidence.compatibility(frame, actual_q)
    assert s[1, 0] > .999999 and abs(s[1, 1] - .35) < 1e-10
    assert s[1, 0] > s[1, 2] and s[1, 0] > s[0, 0] and s[1, 0] > s[2, 0]
    # Verify the efficient finite Fourier expression against the direct kernel.
    recent = bass[(bass[:, 0] > times[frame] - 4.) & (bass[:, 0] <= times[frame]), 0]
    for j, bpm in enumerate(tempi):
        phase = q[None, :] - bpm * (times[frame] - recent[:, None]) / 60.
        direct = np.mean(np.cos(np.pi * phase)**8 + .35*np.sin(np.pi * phase)**8, axis=0)
        assert np.allclose(direct, evidence.compatibility(frame, q)[j], atol=2e-13)

    # Every truncated input prefix agrees with preparation of the full stream;
    # note-offs can be arbitrary, including not-yet-known infinity/NaN values.
    for end in (0, 12, 32, 67):
        truncated = bass[bass[:, 0] <= times[end]].copy()
        truncated[:, 1] = np.nan
        prefix = prepare_rhythm({"bass": truncated}, times[:end+1], tempi, 120.)
        assert np.allclose(prefix.moments, evidence.moments[:end+1], atol=2e-13)
        assert np.array_equal(prefix.available, evidence.available[:end+1])
        assert np.allclose(prefix.log_likelihood(end, q), evidence.log_likelihood(end, q), atol=2e-13)

    # Shift the recording clock without supplying any reference beat phase.
    shifted = bass.copy()
    shifted[:, :2] += 17.3
    shifted_evidence = prepare_rhythm({"bass": shifted}, times + 17.3, tempi, 120.)
    assert np.allclose(shifted_evidence.moments, evidence.moments, atol=2e-12)

    # Bass beats plus horn offbeats still favor the bass beat phase. An
    # offbeat-only stream remains explicitly ambiguous and favors its own phase.
    horns = notes(np.arange(.25, 5., .5))
    mixed = prepare_rhythm({"bass": bass, "horns": horns}, times, tempi, 120.)
    assert mixed.compatibility(frame, actual_q)[1, 0] > mixed.compatibility(frame, actual_q)[1, 1]
    offbeat = prepare_rhythm({"horns": horns}, times, tempi, 120.)
    assert offbeat.compatibility(frame, actual_q)[1, 1] > offbeat.compatibility(frame, actual_q)[1, 0]
    # Adding notes to an existing same-instrument chord does not add onset votes.
    doubled = prepare_rhythm({"bass": np.repeat(bass, 5, axis=0)}, times, tempi, 120.)
    assert np.array_equal(doubled.moments, evidence.moments)

    # Silence, a lone onset, and silence after playing provide neutral updates.
    for streams in ({}, {"bass": notes([0.])}):
        empty = prepare_rhythm(streams, times, tempi, 120.)
        assert not np.any(empty.available)
        assert np.array_equal(empty.log_likelihood(frame, q), np.zeros((3, len(q))))
    assert not evidence.available[60]  # last onset 5.0; two beats silent by 6.0.
    assert not np.any(evidence.log_likelihood(60, q))
    potential = evidence.log_likelihood(frame, q)
    assert np.all(np.isfinite(potential)) and potential.max() == 0 and potential.min() >= -1
    # Whole-beat position identity and dynamically extended grids are preserved.
    assert np.allclose(evidence.compatibility(frame, q), evidence.compatibility(frame, q + 400.), atol=2e-12)
    extended = evidence.compatibility(frame, np.concatenate((q, q + 4.)))
    assert np.allclose(extended[:, :len(q)], extended[:, len(q):])
    return {"prefix_invariance": True, "future_offsets_ignored": True,
            "relative_phase_origin": True, "bass_with_offbeats": True,
            "offbeat_only_ambiguity_exposed": True, "silence_neutral": True,
            "same_instrument_chord_coalescing": True, "bounded_potential": True,
            "dynamic_grid": True, "direct_kernel_equivalence": True}


if __name__ == "__main__":
    import json
    print(json.dumps(checks(), indent=2))
