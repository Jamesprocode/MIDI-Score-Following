"""Maezawa (Yamaha, 2023), harmonic observation component only.

Source: ``Automatic Accompaniment from a Lead-Sheet by Jointly Tracking at
Chord and Downbeat Level``, two-page late-breaking demo, p. 2, section 2.2.
The printed formula is p_h(x | s=i) proportional to
exp(x.T @ T[c_i, r_i] + x.T @ m), with 88-dimensional piano-roll x,
chord template T, and a reference-melody indicator m.

Implemented faithfully: that dot-product log score, and the 88-key pitch axis.
Unspecified in the paper: numerical chord templates, piano-roll amplitudes,
note release/pedal treatment, and harmonic history aggregation. Our declared
default is binary current key activity, with explicit chord tones weighted 1
in every octave and other notes 0. The caller supplies the 12-PC chord masks;
``expand_chroma_templates`` repeats them over MIDI 21..108 without normalization.
Absent reference melody is represented by m=0, never by the observed solo.

``window_seconds`` is an optional, explicitly non-paper history adaptation:
binary union of pitches sounding at any point in (t-window, t]. It never
uses annotated boundaries or future notes. The native default is None.

This module does not implement Yamaha's downbeat TCN, HHSMM, tempo feedback,
or likelihood multiplication by downbeat-position probabilities. It supports
a harmonic diagnostic, not a reproduction of Yamaha's complete follower.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

LOWEST_MIDI = 21
HIGHEST_MIDI = 108
PIANO_PITCHES = np.arange(LOWEST_MIDI, HIGHEST_MIDI + 1)
METHOD_METADATA = {
    "paper": "Maezawa (2023), Automatic Accompaniment from a Lead-Sheet by Jointly Tracking at Chord and Downbeat Level",
    "source_section": "2.2, page 2",
    "implemented_equation": "log p_h(x|s=i) = x^T T_i + x^T m_i + constant",
    "source_scope": "Harmonic observation only; no downbeat model or temporal filter",
    "piano_midi_range_inclusive": [LOWEST_MIDI, HIGHEST_MIDI],
    "paper_template_values_available": False,
    "default_activity": "Binary instantaneous note activity: onset <= t < offset",
    "default_template": "Caller-supplied explicit chord membership: 1 on chord pitches, 0 elsewhere; repeated by octave; unnormalized",
    "default_reference_melody": "Zero vector because the pilot has no reference melody",
    "optional_history": "Trailing binary pitch-occupancy union; adaptation, not specified by the paper",
    "instrument_fusion": "Union of active pitches across selected MIDI streams; adaptation from the paper's single piano input",
    "outside_piano_range": "Excluded, never octave-folded; report with note_range_audit",
    "probability_caveat": "Softmax is a convenience over the supplied candidate set, not a calibrated position posterior",
}


def _notes_array(notes):
    array = np.asarray(notes, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("notes must have shape (N, 3): onset seconds, offset seconds, MIDI pitch")
    if not np.isfinite(array).all() or np.any(array[:, 1] < array[:, 0]):
        raise ValueError("notes must be finite and offsets must not precede onsets")
    if np.any(array[:, 2] != np.floor(array[:, 2])):
        raise ValueError("MIDI pitches must be integers")
    return array


def note_range_audit(notes):
    """Report notes excluded by the paper's 88-key representation."""
    array = _notes_array(notes)
    pitches = array[:, 2]
    outside = (pitches < LOWEST_MIDI) | (pitches > HIGHEST_MIDI)
    unique, counts = np.unique(pitches[outside].astype(int), return_counts=True)
    return {
        "note_count": len(array),
        "outside_piano_range_count": int(outside.sum()),
        "outside_piano_range_by_pitch": {str(int(p)): int(n) for p, n in zip(unique, counts)},
        "zero_duration_note_count": int(np.sum(array[:, 0] == array[:, 1])),
    }


def piano_roll88(notes, times, window_seconds=None):
    """Return a binary (len(times), 88) causal pitch-activity matrix.

    ``notes`` follows the pilot reader's (onset_s, offset_s, midi_pitch) rows.
    None selects instantaneous activity. A positive window selects the union
    of notes intersecting the trailing interval, including a note-on at t.
    Multiple simultaneous instances of one pitch still produce activity 1.
    Out-of-range pitches are ignored; ``note_range_audit`` reports them.
    """
    array = _notes_array(notes)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional array")
    if window_seconds is not None and (not np.isfinite(window_seconds) or window_seconds <= 0):
        raise ValueError("window_seconds must be None or strictly positive")
    values = np.zeros((len(times), 88), dtype=float)
    for onset, offset, pitch in array:
        if pitch < LOWEST_MIDI or pitch > HIGHEST_MIDI or offset <= onset:
            continue
        if window_seconds is None:
            active = (onset <= times) & (times < offset)
        else:
            active = (onset <= times) & (times - window_seconds < offset)
        values[active, int(pitch) - LOWEST_MIDI] = 1.0
    return values


def make_features(notes_by_instrument: Mapping, times, instruments: Sequence[str] | None = None,
                  window_seconds=None):
    """Union selected instrument notes into one 88-key roll, with no double count.

    All instruments are selected when ``instruments`` is None. Per-instrument
    features are available by calling ``piano_roll88`` on an individual array.
    """
    names = list(notes_by_instrument) if instruments is None else list(instruments)
    arrays = [_notes_array(notes_by_instrument[name]) for name in names]
    notes = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 3))
    return piano_roll88(notes, times, window_seconds=window_seconds)


def expand_chroma_templates(templates12):
    """Expand caller's (candidates, 12) weights to (candidates, 88).

    No template normalization, extra pitch penalties, smoothing, or learned
    values are introduced. A 1-D template is also accepted and returns 1-D.
    """
    templates = np.asarray(templates12, dtype=float)
    if templates.ndim not in (1, 2) or templates.shape[-1] != 12:
        raise ValueError("templates12 must have shape (12,) or (candidates, 12)")
    if not np.isfinite(templates).all():
        raise ValueError("templates must be finite")
    return templates[..., PIANO_PITCHES % 12]


def harmonic_log_scores(x88, templates88, melody88=None):
    """Compute exactly x^T T_i + x^T m_i, without exponentiating.

    ``x88`` is (frames, 88); ``templates88`` is (candidates, 88).
    ``melody88`` is None (zero), (88,) shared, or (candidates, 88).
    Melody templates must come from an available reference, not test labels.
    Zero input gives equal log scores, i.e. uninformative evidence; the caller
    decides whether its diagnostic abstains on such frames.
    """
    values = np.asarray(x88, dtype=float)
    templates = np.asarray(templates88, dtype=float)
    if values.ndim != 2 or values.shape[1] != 88:
        raise ValueError("x88 must have shape (frames, 88)")
    if templates.ndim != 2 or templates.shape[1] != 88 or len(templates) == 0:
        raise ValueError("templates88 must have nonempty shape (candidates, 88)")
    if not np.isfinite(values).all() or not np.isfinite(templates).all():
        raise ValueError("piano rolls and templates must be finite")
    if melody88 is not None:
        melody = np.asarray(melody88, dtype=float)
        if melody.shape not in ((88,), templates.shape) or not np.isfinite(melody).all():
            raise ValueError("melody88 must be finite and shaped (88,) or (candidates, 88)")
        templates = templates + melody
    return values @ templates.T


def harmonic_probabilities(x88, templates88, melody88=None):
    """Normalize exp(log scores) across the supplied candidates stably.

    This normalization is optional and preserves score ranking. It does not
    supply a temporal prior or turn the pilot scores into calibrated estimates.
    """
    logits = harmonic_log_scores(x88, templates88, melody88=melody88)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def checks():
    """Small analytical checks of equation, piano axis, boundaries, and causality."""
    notes = np.array([[0., 1., 60], [0., 1., 64], [0., 1., 67],
                      [0., 1., 21], [0., 1., 108], [0., 1., 20], [0., 1., 109]])
    times = np.array([0., .5, 1., 2.])
    roll = piano_roll88(notes, times)
    assert roll.shape == (4, 88)
    assert roll[0, 0] == roll[0, -1] == 1
    assert np.array_equal(roll.sum(axis=1), [5, 5, 0, 0])
    assert note_range_audit(notes)["outside_piano_range_count"] == 2
    # C-major has C4/E4/G4/C8 = 4 matches; A-minor has A0/C4/E4/C8 = 4.
    chroma = np.zeros((2, 12))
    chroma[0, [0, 4, 7]] = 1
    chroma[1, [9, 0, 4]] = 1
    templates = expand_chroma_templates(chroma)
    assert np.array_equal(harmonic_log_scores(roll, templates), [[4, 4], [4, 4], [0, 0], [0, 0]])
    melody = np.zeros((2, 88))
    melody[1, 67 - LOWEST_MIDI] = 1
    assert np.array_equal(harmonic_log_scores(roll, templates, melody)[0], [4, 5])
    probabilities = harmonic_probabilities(roll, templates, melody)
    assert np.allclose(probabilities.sum(axis=1), 1.)
    assert np.isclose(probabilities[0, 1] / probabilities[0, 0], np.e)
    assert np.allclose(probabilities[-1], [.5, .5])
    assert np.array_equal(make_features({"a": notes, "b": notes}, times), roll)
    future = np.vstack([notes, [3., 4., 62]])
    for window in (None, 2.):
        assert np.array_equal(piano_roll88(notes, times, window), piano_roll88(future, times, window))
    # Ended notes are retained only by the declared history adaptation.
    assert piano_roll88(notes, [1.5], 2.).sum() == 5
    assert piano_roll88(notes, [3.], 2.).sum() == 0
    # Future note-off time cannot affect activity before either candidate offset.
    assert np.array_equal(piano_roll88([[0., 5., 60]], times), piano_roll88([[0., 10., 60]], times))
    return {"status": "passed", "checks": ["88-key axis", "range audit", "active boundaries",
            "exact dot-product and melody", "stable exponential normalization", "silence",
            "instrument union", "future-note invariance", "history adaptation boundaries"]}


if __name__ == "__main__":
    import json
    print(json.dumps(checks(), indent=2))
