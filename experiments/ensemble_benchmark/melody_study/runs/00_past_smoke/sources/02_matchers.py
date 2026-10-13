"""Five non-neural comparisons of fixed nine-note melodic windows.

Every method returns similarity in [0,1], where one is an exact feature match.
These are declared benchmark adaptations of established matching families, not
paper reproductions or calibrated probabilities. Same numerical scores across
different methods do not imply equal evidence strength. See references.json.

Inputs are monophonic event windows: integer MIDI pitches and strictly increasing
onset seconds, with nine notes in each window. Simultaneous events must be reduced
to a monophonic sequence upstream. No note offsets are used: all eight IOIs are
completed as soon as the ninth onset has arrived. Reference selection, exclusion
of overlapping self-matches, past-only memory, score-position mapping and any
position prior belong to the caller, not to this stateless matcher.

pc_cosine: cosine of 12-bin pitch-class count histograms, ignoring order/rhythm.
pc_edit: unit insertion/deletion/substitution Levenshtein on pitch classes;
    similarity=1-distance/9. Neither PC method is transposition-invariant.
interval_edit: unit Levenshtein on eight signed semitone intervals;
    similarity=1-distance/8. Octave sizes are retained, not folded modulo12.
interval_dtw: DTW on (signed interval, IOI/median IOI), with a two-event band.
    Local cost=.5*min(|interval difference|/12,1)
              +.5*min(|log2(normalized IOI ratio)|/2,1).
    The minimum-SUM path is found; exact cost ties prefer shorter paths. Return
    1-(path cost/path length). This is post-normalized DTW, not minimum-average
    path optimization. No learned costs, tempo prior, or endpoint anchor is used.
interval_ngrams: six overlapping triples of signed intervals (four-note units),
    augmented by their two adjacent log2 IOI ratios, quantized to quarter units.
    Similarity is multiset Dice overlap: 2*intersection/(6+6). Repeated tokens
    cannot receive more credit than their multiplicity in either window.

All interval methods are invariant to constant pitch transposition. All methods
are invariant to positive global tempo scaling and recording-clock translation;
pc_cosine and the edit methods simply ignore timing. Changing rhythm can affect
DTW/n-grams. Pitch-class methods additionally ignore octave displacement of each
note; interval methods retain contour and octave leap size. No method establishes
bar/chorus identity by itself, or recognizes arbitrary nonrepeating improvisation.
"""
from __future__ import annotations

import numpy as np


WINDOW_NOTES = 9
METHODS = ("pc_cosine", "pc_edit", "interval_edit", "interval_dtw", "interval_ngrams")
DTW_BAND_EVENTS = 2
RHYTHM_RATIO_BINS_PER_OCTAVE = 4


def _inputs(query_pitches, query_onsets, ref_pitches, ref_onsets):
    q = np.asarray(query_pitches, dtype=float)
    qt = np.asarray(query_onsets, dtype=float)
    r = np.asarray(ref_pitches, dtype=float)
    rt = np.asarray(ref_onsets, dtype=float)
    if q.shape != (WINDOW_NOTES,) or qt.shape != q.shape:
        raise ValueError("query pitches and onsets must each have shape (9,)")
    if r.ndim != 2 or r.shape[1] != WINDOW_NOTES or rt.shape != r.shape:
        raise ValueError("reference pitches and onsets must each have shape (R,9), including (0,9)")
    if not all(np.isfinite(x).all() for x in (q, qt, r, rt)):
        raise ValueError("pitches and onsets must be finite")
    if np.any(q != np.rint(q)) or np.any(r != np.rint(r)) or np.any((q < 0) | (q > 127)) or np.any((r < 0) | (r > 127)):
        raise ValueError("pitches must be integer MIDI values in [0,127]")
    qioi, rioi = np.diff(qt), np.diff(rt, axis=1)
    if np.any(qioi <= 0) or np.any(rioi <= 0):
        raise ValueError("onsets must increase strictly; deduplicate simultaneous events upstream")
    return q.astype(np.int64), r.astype(np.int64), qioi, rioi


def _levenshtein(query, references):
    """Unit edit distance, vectorized over all reference rows."""
    count, width = references.shape
    previous = np.broadcast_to(np.arange(width+1), (count,width+1)).copy()
    for i, value in enumerate(query, 1):
        current = np.empty_like(previous)
        current[:, 0] = i
        for j in range(1, width+1):
            current[:, j] = np.minimum(
                np.minimum(previous[:, j]+1, current[:, j-1]+1),
                previous[:, j-1] + (references[:, j-1] != value))
        previous = current
    return previous[:, -1]


def _dtw(query_intervals, reference_intervals, qioi, rioi):
    count, width = reference_intervals.shape
    qlog = np.log2(qioi / np.median(qioi))
    rlog = np.log2(rioi / np.median(rioi, axis=1, keepdims=True))
    previous = np.full((count,width+1), np.inf)
    previous[:, 0] = 0.
    previous_length = np.zeros((count,width+1), dtype=np.int64)
    for i in range(1, len(query_intervals)+1):
        current = np.full_like(previous, np.inf)
        current_length = np.zeros_like(previous_length)
        for j in range(max(1,i-DTW_BAND_EVENTS), min(width,i+DTW_BAND_EVENTS)+1):
            costs = np.stack((previous[:, j-1], previous[:, j], current[:, j-1]), axis=1)
            lengths = np.stack((previous_length[:, j-1], previous_length[:, j], current_length[:, j-1]), axis=1)
            best = costs.min(axis=1)
            length = np.where(costs == best[:, None], lengths, 1000).min(axis=1)
            pitch_cost = np.minimum(np.abs(reference_intervals[:, j-1]-query_intervals[i-1])/12., 1.)
            rhythm_cost = np.minimum(np.abs(rlog[:, j-1]-qlog[i-1])/2., 1.)
            current[:, j] = best + .5*(pitch_cost+rhythm_cost)
            current_length[:, j] = length+1
        previous, previous_length = current, current_length
    return 1. - previous[:, -1]/previous_length[:, -1]


def _ngram_tokens(intervals, iois):
    """Last dimension represents a sequence; preserve any leading dimensions."""
    pitches = np.lib.stride_tricks.sliding_window_view(intervals, 3, axis=-1)
    ratios = np.diff(np.log2(iois), axis=-1)
    # Rounding well below bin precision makes exact-boundary quantization stable
    # under harmless floating-point differences from global time rescaling.
    ratios = np.rint(np.round(ratios*RHYTHM_RATIO_BINS_PER_OCTAVE, 10)).astype(np.int64)
    rhythm = np.lib.stride_tricks.sliding_window_view(ratios, 2, axis=-1)
    return np.concatenate((pitches, rhythm), axis=-1)


def _ngrams(query_intervals, reference_intervals, qioi, rioi):
    query = _ngram_tokens(query_intervals, qioi)
    references = _ngram_tokens(reference_intervals, rioi)
    unique, counts = np.unique(query, axis=0, return_counts=True)
    reference_counts = np.all(references[:, :, None, :] == unique[None, None, :, :], axis=-1).sum(axis=1)
    intersection = np.minimum(reference_counts, counts[None, :]).sum(axis=1)
    return 2. * intersection / (len(query)+references.shape[1])


def score_windows(query_pitches, query_onsets, ref_pitches, ref_onsets, method):
    """Return one [0,1] similarity for each reference nine-note window.

    Accepts numpy arrays or shape-compatible sequences. Empty references must
    have shape (0,9), and return shape (0,). Inputs are never modified. All work
    is vectorized across R; edit/DTW loops traverse only their small 8/9-note DP
    lattice. This function neither softmaxes scores nor selects a winning match.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    query, references, qioi, rioi = _inputs(query_pitches, query_onsets, ref_pitches, ref_onsets)
    if not len(references):
        return np.empty(0, dtype=float)
    if method == "pc_cosine":
        qhist = np.bincount(query % 12, minlength=12).astype(float)
        indices = references % 12 + 12*np.arange(len(references))[:, None]
        rhist = np.bincount(indices.ravel(), minlength=12*len(references)).reshape(-1,12)
        scores = (rhist @ qhist)/(np.linalg.norm(rhist, axis=1)*np.linalg.norm(qhist))
    elif method == "pc_edit":
        scores = 1. - _levenshtein(query % 12, references % 12)/WINDOW_NOTES
    else:
        qinterval = np.diff(query)
        rinterval = np.diff(references, axis=1)
        if method == "interval_edit":
            scores = 1. - _levenshtein(qinterval, rinterval)/(WINDOW_NOTES-1)
        elif method == "interval_dtw":
            scores = _dtw(qinterval, rinterval, qioi, rioi)
        else:
            scores = _ngrams(qinterval, rinterval, qioi, rioi)
    return np.clip(scores, 0., 1.)


def checks():
    """Synthetic feature/algorithm properties, without dataset tuning."""
    q = np.array([60,62,65,64,69,67,71,72,74])
    qt = np.r_[0., np.cumsum([.5,.25,.5,1.,.25,.5,.75,.25])]
    refs = np.stack((q, q+5, q+12, q[::-1], np.array([62,65,64,69,67,71,72,74,76])))
    rt = np.broadcast_to(qt, refs.shape).copy()
    for method in METHODS:
        score = score_windows(q, qt, refs, rt, method)
        assert np.isfinite(score).all() and np.all((score >= 0) & (score <= 1))
        assert abs(score[0]-1.) < 1e-12
        rescaled = score_windows(q, qt*1.7+10., refs, rt*3.+31., method)
        assert np.allclose(score, rescaled, atol=1e-12), method
        assert score_windows(q, qt, np.empty((0,9)), np.empty((0,9)), method).shape == (0,)
        if method.startswith("interval_"):
            assert np.allclose(score[:3], 1., atol=1e-12), method
            assert score[3] < score[0], method
        else:
            assert abs(score[2]-1.) < 1e-12 and score[1] < score[0], method
    cosine = score_windows(q, qt, refs, rt, "pc_cosine")
    edit = score_windows(q, qt, refs, rt, "pc_edit")
    assert abs(cosine[3]-1.) < 1e-12 and edit[3] < 1.  # same bag, different order
    # Shifted windows share eight ordered notes: delete first and insert last.
    assert abs(edit[4]-(1.-2./9.)) < 1e-12
    assert abs(score_windows(q, qt, refs, rt, "interval_edit")[4]-(1.-2./8.)) < 1e-12
    # A single octave displacement is ignored by PC but changes signed contour.
    octave = q.copy(); octave[4] += 12
    for method in ("pc_cosine", "pc_edit"):
        assert abs(score_windows(q, qt, octave[None,:], qt[None,:], method)[0]-1.) < 1e-12
    assert score_windows(q, qt, octave[None,:], qt[None,:], "interval_edit")[0] < 1.
    # Irregular rhythm matters to rhythmic matchers, but not pitch-only edit.
    even = np.arange(9)*.5
    assert score_windows(q, qt, q[None,:], even[None,:], "interval_edit")[0] == 1.
    assert score_windows(q, qt, q[None,:], even[None,:], "interval_dtw")[0] < 1.
    assert score_windows(q, qt, q[None,:], even[None,:], "interval_ngrams")[0] < 1.
    # Repeated n-grams receive only multiset-matched credit, never excess votes.
    repeated = np.arange(60,69)
    assert score_windows(repeated, even, repeated[None,:], even[None,:], "interval_ngrams")[0] == 1.
    try:
        score_windows(q, np.zeros(9), refs, rt, "pc_edit")
    except ValueError:
        pass
    else:
        raise AssertionError("Nonpositive IOIs must be rejected")
    return {"exact_match":True,"tempo_scaling_and_clock_shift":True,
            "interval_transposition_invariance":True,"pitchclass_octave_invariance":True,
            "insert_delete_alignment":True,"ordering_distinction":True,
            "rhythm_sensitivity":True,"multiset_counting":True,"bounds_and_empty":True}


if __name__ == "__main__":
    import json
    import time
    print(json.dumps(checks(), indent=2))
    rng = np.random.default_rng(0)
    pitches = rng.integers(48,85,(10000,9))
    onsets = np.cumsum(rng.uniform(.1,1.,(10000,9)),axis=1)
    timings = {}
    for name in METHODS:
        before = time.perf_counter()
        score_windows(pitches[0],onsets[0],pitches,onsets,name)
        timings[name] = time.perf_counter()-before
    print(json.dumps({"reference_windows":10000,"seconds":timings},indent=2))
