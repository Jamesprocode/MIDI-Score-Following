"""Pardo--Birmingham (2002) CHORD segment-labeling component, window adaptation.

Primary source: Algorithms for Chordal Analysis, Computer Music Journal 26(2),
27--49, doi:10.1162/014892602760137167. Tables 2 and 4; Figures 2--4; pp. 28--32.
https://interactiveaudiolab.github.io/assets/papers/pardo-birmingham-cmj02.pdf

The published rule is S = P - (M + N): P is matched note weight, N is unmatched
note weight, and M counts template pitch classes absent from the segment. Each
note's weight is the number of minimal note-on/off intervals it spans, NOT its
duration in seconds or its number of onsets. Simultaneous sounding notes of the
same pitch class each contribute one per interval.

Adaptations for this pilot are explicit:
* Segments are trailing fixed-second windows, clipped at the query time. They
  replace CHORD's search for a segmentation; no annotated chord boundary is used.
* Caller-supplied binary chart templates may expand/restrict the published set
  of six chord qualities times twelve roots (72 templates).
* All maximizers are retained by default. Root-weight tie resolution is optional.
  The prior-probability rule and future diminished-chord resolution are omitted.

This is neither the complete CHORD segmentation system nor the 2001 lead-sheet
follower (which adds metric reduction, chord-sequence alignment and corpus-based
pair-match scores). It estimates chord compatibility, not score position.

API: segment_profiles(notes, times, window) -> (T, 12);
     score(profiles, templates) -> (T, K);
     root_tie_mask(profiles, scores, roots) -> bool (T, K).
Pass the combined raw notes into segment_profiles for pooled input. Summing
profiles independently computed for each instrument misses the partition points
created by other instruments and does not give the published pooled weighting.
"""
from __future__ import annotations

import numpy as np


# Table 2, in descending Kostka--Payne corpus frequency. These frequencies are
# not fitted to the pilot and are not used in the primary no-tie-break scorer.
NATIVE_INTERVALS = {
    "major": (0, 4, 7),
    "dominant7": (0, 4, 7, 10),
    "minor": (0, 3, 7),
    "diminished7": (0, 3, 6, 9),
    "halfdiminished7": (0, 3, 6, 10),
    "diminished": (0, 3, 6),
}


def segment_profiles(notes, times, window):
    """Return raw minimal-segment counts in each causal trailing window.

    ``notes`` is an (N, 3) array of onset seconds, offset seconds, MIDI pitch.
    ``times`` is a 1-D array in seconds; ``window`` is a positive duration.
    Each positive-length overlap between a note-event interval and (t-window,t)
    contributes its active-note pitch-class counts once, even when only part of
    the interval lies inside the window. Window edges are artificial partition
    boundaries, an adaptation needed for fixed-second comparison. No future
    onset or offset changes the profile at t. A note starting exactly at t has
    no elapsed interval and contributes only at a later query.

    Zero-duration notes are ignored. Overlapping notes/unisons are not merged.
    No velocity weighting, instrument balancing or normalization is performed.
    """
    notes = np.asarray(notes, dtype=float)
    times = np.asarray(times, dtype=float)
    if notes.size == 0:
        notes = np.empty((0, 3), dtype=float)
    if notes.ndim != 2 or notes.shape[1] != 3:
        raise ValueError("notes must have shape (N, 3): onset, offset, MIDI pitch")
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional array")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    if not np.isfinite(notes).all() or np.any(notes[:, 1] < notes[:, 0]):
        raise ValueError("notes must be finite with offset >= onset")
    if np.any(notes[:, 2] != np.floor(notes[:, 2])) or np.any((notes[:, 2] < 0) | (notes[:, 2] > 127)):
        raise ValueError("MIDI pitches must be integers from 0 to 127")
    notes = notes[notes[:, 1] > notes[:, 0]]
    if len(notes) == 0:
        return np.zeros((len(times), 12), dtype=float)

    # The event stream may be loaded from disk, but queries only sum intervals
    # intersecting the observed past. Changes after t cannot alter that sum.
    boundaries = np.unique(notes[:, :2])
    deltas = np.zeros((len(boundaries), 12), dtype=np.int64)
    pcs = notes[:, 2].astype(int) % 12
    np.add.at(deltas, (np.searchsorted(boundaries, notes[:, 0]), pcs), 1)
    np.add.at(deltas, (np.searchsorted(boundaries, notes[:, 1]), pcs), -1)
    active = np.cumsum(deltas, axis=0)[:-1]
    cumulative = np.concatenate((np.zeros((1, 12), dtype=np.int64),
                                 np.cumsum(active, axis=0)), axis=0)
    left = np.searchsorted(boundaries, times - window, side="right") - 1
    right = np.searchsorted(boundaries, times, side="left")
    left = np.clip(left, 0, len(active))
    right = np.clip(right, 0, len(active))
    return (cumulative[right] - cumulative[left]).astype(float)


def _validated_arrays(profiles, templates):
    profiles = np.asarray(profiles, dtype=float)
    templates = np.asarray(templates, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != 12:
        raise ValueError("profiles must have shape (T, 12)")
    if templates.ndim != 2 or templates.shape[1] != 12 or len(templates) == 0:
        raise ValueError("templates must have nonempty shape (K, 12)")
    if not np.isfinite(profiles).all() or np.any(profiles < 0):
        raise ValueError("profiles must be finite and nonnegative")
    if not np.isfinite(templates).all() or not np.isin(templates, [0, 1]).all():
        raise ValueError("Pardo templates must be binary pitch-class membership")
    return profiles, templates


def score(profiles, templates):
    """Compute published P-(M+N) for every profile and binary template.

    Templates are (K, 12), profiles are (T, 12). These scores are not cosine
    similarities or calibrated probabilities. Preserve the raw segment counts:
    normalizing profiles changes the tradeoff against the unit missing-tone
    penalty and is an additional, unpublished adaptation. Silence produces
    negative scores; callers must use profile mass == 0 as an abstention mask.
    """
    profiles, templates = _validated_arrays(profiles, templates)
    positive = profiles @ templates.T
    negative = profiles.sum(axis=1, keepdims=True) - positive
    missing = (profiles == 0).astype(float) @ templates.T
    return positive - (missing + negative)


pardo_scores = score


def root_tie_mask(profiles, scores, roots):
    """Optional Figure-4 rule 1: among tied maxima, prefer larger root weight.

    Returns all surviving candidates, retaining unresolved ties. Silence returns
    an all-False row. This does not use the paper's other two tie rules, and is
    separate from the primary no-tie-break scores. Roots are K pitch classes.
    """
    profiles = np.asarray(profiles, dtype=float)
    scores = np.asarray(scores, dtype=float)
    roots = np.asarray(roots)
    if profiles.ndim != 2 or profiles.shape[1] != 12 or scores.ndim != 2 or scores.shape[0] != len(profiles):
        raise ValueError("profiles and scores must have matching frame dimensions")
    if roots.shape != (scores.shape[1],) or np.any(roots != np.floor(roots)) or np.any((roots < 0) | (roots > 11)):
        raise ValueError("roots must contain one integer pitch class per candidate")
    tied = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0, atol=1e-10)
    weights = np.where(tied, profiles[:, roots.astype(int)], -np.inf)
    result = tied & (weights == weights.max(axis=1, keepdims=True))
    return result & (profiles.sum(axis=1, keepdims=True) > 0)


def native_template_qualities(templates, roots):
    """Report native quality or None for each supplied template at its root.

    Used to disclose vocabulary expansion, not to reduce/rename jazz chords.
    Matching a pitch set does not establish equivalence of annotated spellings.
    """
    templates = np.asarray(templates, dtype=float)
    roots = np.asarray(roots)
    _validated_arrays(np.empty((0, 12)), templates)
    if roots.shape != (len(templates),) or np.any(roots != np.floor(roots)) or np.any((roots < 0) | (roots > 11)):
        raise ValueError("roots must contain one integer pitch class per template")
    output = []
    for template, root in zip(templates, roots.astype(int)):
        intervals = tuple(np.flatnonzero(np.roll(template, -root)))
        output.append(next((name for name, native in NATIVE_INTERVALS.items()
                            if intervals == native), None))
    return output


def checks():
    """Small scientific checks: paper arithmetic, partition weights and causality."""
    # Paper Figure 2: C4/E4 span three intervals; C5, D5, E5 one each.
    notes = np.array([[0, 3, 60], [0, 3, 64], [0, 1, 72],
                      [1, 2, 74], [2, 3, 76]], dtype=float)
    profile = segment_profiles(notes, np.array([3.0]), 3.0)
    assert np.array_equal(profile[0, [0, 2, 4]], [4, 1, 4])
    templates = np.zeros((2, 12))
    templates[0, [9, 0, 4]] = 1  # A minor: P=8, N=1, M=1, S=6.
    templates[1, [0, 4, 7]] = 1  # C major has the same score.
    scores = score(profile, templates)
    assert np.array_equal(scores, [[6, 6]])
    assert np.array_equal(root_tie_mask(profile, scores, [9, 0]), [[False, True]])
    assert native_template_qualities(templates, [9, 0]) == ["minor", "major"]

    # A sustained C is split by another voice's events, even on the same PC.
    unisons = np.array([[0, 4, 60], [1, 2, 72]], dtype=float)
    assert segment_profiles(unisons, [4], 4)[0, 0] == 4
    # Simultaneous onsets create one interval, not one interval per event.
    block = np.array([[0, 2, 60], [0, 2, 64], [0, 2, 67]], dtype=float)
    assert np.array_equal(segment_profiles(block, [1], .5)[0, [0, 4, 7]], [1, 1, 1])
    assert not segment_profiles(block, [0], 1).any()
    assert not segment_profiles(block, [3], 1).any()

    future = np.array([[.25, 8, 60], [1.2, 3, 64], [1.8, 2, 67],
                       [5, 6, 71], [0, 0, 55]], dtype=float)
    for now in [.1, .25, 1.2, 1.8, 2, 2.3, 5, 8, 10]:
        prefix = future[future[:, 0] < now].copy()
        prefix[:, 1] = np.minimum(prefix[:, 1], now)
        for window in [.5, 1, 2, 4]:
            assert np.array_equal(segment_profiles(future, [now], window),
                                  segment_profiles(prefix, [now], window)), "Future-event leakage"
    assert np.array_equal(score(np.zeros((1, 12)), templates), [[-3, -3]])
    assert not root_tie_mask(np.zeros((1, 12)), np.zeros((1, 2)), [9, 0]).any()


if __name__ == "__main__":
    checks()
    print("Pardo matcher checks passed")
