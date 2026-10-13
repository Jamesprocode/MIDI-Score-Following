"""Causal melodic memory and sparse position evidence, separate from evaluation.

The anchor follower is an independent causal harmonic tracker. Past observed
phrases are tagged with its PAST position estimates, never reference timestamps.
An untimed performance plan supplies repeated measure IDs and planned solo roles.
Queries contain nine onsets, so their elapsed duration varies; this is not a
fixed short time window. Span, internal-gap, freshness, and separation limits
are expressed in nominal beats, converted using the supplied catalog BPM.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

import matchers

PARAMETERS = {"window_notes": 9, "maximum_span_nominal_beats": 32.,
              "maximum_internal_gap_nominal_beats": 8., "freshness_nominal_beats": 2.,
              "self_match_separation_nominal_beats": 2., "minimum_similarity": .75,
              "median_background_min_references": 8,
              "max_distinct_reference_hits": 8, "position_sigma_beats": .75,
              "reference_stride": 2,
              "query_roles": "catalog lead horns; exclude planned soloist at anchor-estimated position",
              "reference_roles": "same causal role mask as query for past memory",
              "anchor": "independent p02 harmonic-only causal estimates; zero-order prediction to note time"}


def untimed_plan(data):
    beats = data["beats_per_bar"]
    first = data["first_annotation_bar"]
    sections = []
    for row in data["sections"]:
        sections.append({"start": (float(row["start_bar"])-first+float(row["start_fraction"]))*beats,
                         "end": (float(row["end_bar"])-first+float(row["end_fraction"]))*beats,
                         "soloist": row.get("soloist", "NA"), "section": row.get("section", "")})
    return {"beats_per_bar": beats, "measure_ids": np.array([str(r["measure"]) for r in data["bars"]]),
            "sections": sections, "start_time": data["start_time"], "initial_bpm": data["initial_bpm"]}


def role_matches(instrument, role):
    instrument, role = str(instrument).strip().lower(), str(role).strip().lower()
    # A named alto solo must not remove tenor accompaniment, or vice versa.
    # Family matching is reserved for an explicitly unspecified sax role.
    return role == instrument or (role in ("sax", "saxophone") and "sax" in instrument)


def similarity_threshold(similarities):
    """Declared heuristic gate, not a calibrated probability or significance test.

    Sparse memory permits a match above the fixed minimum. With at least eight
    eligible windows, require contrast above their median as well. Strictly
    greater comparison abstains on large all-tied reference sets, even at one.
    The eight-window count is before recurring-position consolidation; windows
    need not be independent or distinct chart locations.
    """
    threshold = PARAMETERS["minimum_similarity"]
    if len(similarities) >= PARAMETERS["median_background_min_references"]:
        threshold = max(threshold, float(np.median(similarities)))
    return threshold


def estimated_at(onsets, times, anchor, plan):
    idx = np.searchsorted(times, onsets, side="right")-1
    q = np.zeros(len(onsets))
    valid = idx >= 0
    q[valid] = anchor["position_beats"][idx[valid]] + (
        onsets[valid]-times[idx[valid]])*anchor["tempo_bpm"][idx[valid]]/60.
    q[~valid] = (onsets[~valid]-plan["start_time"])*plan["initial_bpm"]/60.
    return q


def prepare_stream(notes, instrument, times, anchor, plan, mask_solo=True):
    # At simultaneous horn note-ons select the highest pitch. No future offsets,
    # duration filtering, or notes arriving after the event enter this choice.
    notes = np.asarray(notes)
    notes = notes[notes[:, 0] >= plan["start_time"]]
    if not len(notes):
        return np.empty((0, 3))
    onsets = np.unique(notes[:, 0])
    pitches = np.array([notes[notes[:, 0] == t, 2].max() for t in onsets])
    q = estimated_at(onsets, times, anchor, plan)
    keep = np.ones(len(q), bool)
    if mask_solo:
        for section in plan["sections"]:
            if role_matches(instrument, section["soloist"]):
                keep &= ~((q >= section["start"]) & (q < section["end"]))
    return np.column_stack((onsets[keep], pitches[keep], q[keep]))


def windows(stream, bpm, *, written=False):
    if len(stream) < matchers.WINDOW_NOTES:
        return np.empty((0, 9)), np.empty((0, 9)), np.empty(0), np.empty(0, dtype=int)
    times = np.lib.stride_tricks.sliding_window_view(stream[:, 0], 9)
    pitches = np.lib.stride_tricks.sliding_window_view(stream[:, 1], 9)
    ends = np.arange(8, len(stream))
    nominal_beat = 1. if written else 60./bpm
    valid = ((times[:, -1]-times[:, 0] <= PARAMETERS["maximum_span_nominal_beats"]*nominal_beat) &
             (np.diff(times, axis=1).max(axis=1) <= PARAMETERS["maximum_internal_gap_nominal_beats"]*nominal_beat))
    return pitches[valid], times[valid], stream[ends[valid], 2], ends[valid]


def expand_location(q, plan):
    beats = plan["beats_per_bar"]
    bar = int(np.floor(q/beats))
    measures = plan["measure_ids"]
    if not 0 <= bar < len(measures):
        return np.empty(0)
    return np.flatnonzero(measures == measures[bar])*beats + (q-bar*beats)


@dataclass
class PhraseEvidence:
    records: list
    available: np.ndarray
    scoring_seconds: float
    query_updates: int
    reference_comparisons: int

    def __call__(self, frame, now, positions, tempi):
        result = np.zeros(len(positions))
        sigma = PARAMETERS["position_sigma_beats"]
        for end_time, centers, strengths in self.records[frame]:
            # A completed phrase votes for its END. Advance to the present using
            # each candidate tempo, without awaiting another note.
            phrase_position = positions - (now-end_time)*tempi/60.
            for center, strength in zip(centers, strengths):
                value = strength*np.exp(-.5*((phrase_position-center)/sigma)**2)
                result = np.maximum(result, value)
        return result


def prepare_all(streams, times, anchor, plan, *, written_notes=None, mask_solo=True):
    """Return all five evidence objects from one shared query/reference protocol.

    `streams` contains only selected lead-horn tracks. `written_notes`, when
    supplied, is Nx3 [onset score beats, MIDI pitch, aligned plan q]. The written
    source may be known in full, but queries always end at the current event.
    """
    methods = matchers.METHODS
    records = {m: [[] for _ in times] for m in methods}
    total = {m: 0. for m in methods}
    updates = 0
    comparisons = 0
    nominal = 60./plan["initial_bpm"]
    for instrument, notes in streams.items():
        stream = prepare_stream(notes, instrument, times, anchor, plan, mask_solo=mask_solo)
        qp, qt, qq, qend = windows(stream, plan["initial_bpm"])
        if written_notes is None:
            rp, rt, rq, rend = windows(stream, plan["initial_bpm"])
            selection = np.arange(0, len(rq), PARAMETERS["reference_stride"])
            rp, rt, rq = rp[selection], rt[selection], rq[selection]
        else:
            rp, rt, rq, rend = windows(written_notes, plan["initial_bpm"], written=True)
        if not len(qt) or not len(rt):
            continue
        current = -1
        cached = {}
        for frame, now in enumerate(times):
            index = int(np.searchsorted(qt[:, -1], now, side="right")-1)
            if index < 0 or now-qt[index, -1] > PARAMETERS["freshness_nominal_beats"]*nominal:
                continue
            # Exclusion follows the anchor's CURRENT predicted planned role as
            # well as the note-time mask, so old notes do not persist into a solo.
            if mask_solo and any(s["start"] <= anchor["position_beats"][frame] < s["end"] and
                                 role_matches(instrument, s["soloist"]) for s in plan["sections"]):
                continue
            if index != current:
                current = index
                eligible = np.ones(len(rq), bool) if written_notes is not None else (
                    rt[:, -1] <= qt[index, 0]-PARAMETERS["self_match_separation_nominal_beats"]*nominal)
                # q anchors must describe a valid plan location; no truth-based
                # repair is made when the independent anchor has lost its place.
                eligible &= (rq >= 0) & (rq < len(plan["measure_ids"])*plan["beats_per_bar"])
                cached = {}
                if not eligible.any():
                    continue
                ep, et, eq = rp[eligible], rt[eligible], rq[eligible]
                updates += 1
                comparisons += len(eq)
                for method in methods:
                    before = time.perf_counter()
                    similarities = matchers.score_windows(qp[index], qt[index], ep, et, method)
                    total[method] += time.perf_counter()-before
                    threshold = similarity_threshold(similarities)
                    hits = np.flatnonzero(similarities > threshold+1e-10)
                    hit_positions, hit_strengths, seen = [], [], set()
                    for hit in hits[np.argsort(-similarities[hits], kind="stable")]:
                        centers = expand_location(eq[hit], plan)
                        if not len(centers):
                            continue
                        # Consolidate duplicate observations at approximately
                        # the same recurring chart position before top-k.
                        key = tuple(np.rint(centers*2).astype(int))
                        if key in seen:
                            continue
                        seen.add(key)
                        strength = float((similarities[hit]-threshold)/max(1.-threshold, 1e-10))
                        hit_positions.extend(centers.tolist())
                        hit_strengths.extend([strength]*len(centers))
                        if len(seen) == PARAMETERS["max_distinct_reference_hits"]:
                            break
                    if hit_positions:
                        cached[method] = (float(qt[index, -1]), np.array(hit_positions), np.array(hit_strengths))
            for method, value in cached.items():
                records[method][frame].append(value)
    return {m: PhraseEvidence(records[m], np.array([bool(x) for x in records[m]]), total[m], updates, comparisons)
            for m in methods}


def checks():
    # An explicit two-chorus phrase repeat; no overlap or future cache leakage.
    times = np.arange(0., 24., .2)
    anchor = {"position_beats": times*2, "tempo_bpm": np.full(len(times), 120.)}
    plan = {"beats_per_bar": 4., "measure_ids": np.array([str(i%4) for i in range(12)]),
            "sections": [], "start_time": 0., "initial_bpm": 120.}
    onsets = np.arange(.5, 23., .5)
    pitch = np.resize(np.array([60, 62, 65, 64, 67, 69, 71, 72, 70]), len(onsets))
    streams = {"trumpet": np.column_stack((onsets, onsets+.3, pitch))}
    full = prepare_all(streams, times, anchor, plan)
    stop = 70
    short_streams = {"trumpet": streams["trumpet"][onsets <= times[stop-1]]}
    short_anchor = {k: v[:stop] for k,v in anchor.items()}
    short = prepare_all(short_streams, times[:stop], short_anchor, plan)
    for method in matchers.METHODS:
        assert np.array_equal(full[method].available[:stop], short[method].available)
        for i in range(stop):
            position = np.arange(0., 48., .5)
            assert np.array_equal(full[method](i, times[i], position, np.full(len(position),120.)),
                                  short[method](i, times[i], position, np.full(len(position),120.)))
        # Earliest eligible pair: the first nine-note reference ends at 4.5 s;
        # a query can start one second later and needs another four seconds.
        # Availability before 9.5 s is impossible, independently of match scores.
        earliest_reference_end = onsets[matchers.WINDOW_NOTES-1]
        first_window_span = onsets[matchers.WINDOW_NOTES-1] - onsets[0]
        earliest_pair = (earliest_reference_end +
                         PARAMETERS["self_match_separation_nominal_beats"]*60./120. +
                         first_window_span)
        assert not full[method].available[times < earliest_pair].any()
        assert not full[method].available[times-onsets[-1] > 60./120.*PARAMETERS["freshness_nominal_beats"]].any()
    assert full["pc_edit"].available.any()
    assert np.array_equal(expand_location(5.,plan), np.array([5.,21.,37.]))
    assert not len(expand_location(-.1, plan))
    assert not len(expand_location(48., plan))
    assert np.array_equal(expand_location(4.,plan), np.array([4.,20.,36.]))

    # Specific soloists remain separate; generic sax roles retain family intent.
    assert role_matches("saxophone_alto", "saxophone_alto")
    assert not role_matches("saxophone_tenor", "saxophone_alto")
    assert not role_matches("saxophone_alto", "saxophone_tenor")
    assert role_matches("saxophone_tenor", "saxophone")
    assert role_matches("trumpet", "TRUMPET")
    alto_solo = {**plan, "sections": [{"start": 0., "end": 48., "soloist": "saxophone_alto"}]}
    assert not len(prepare_stream(streams["trumpet"], "saxophone_alto", times, anchor, alto_solo))
    assert len(prepare_stream(streams["trumpet"], "saxophone_tenor", times, anchor, alto_solo)) == len(onsets)
    trumpet_solo = {**plan, "sections": [{"start": 32., "end": 40., "soloist": "trumpet"}]}
    masked_stream = prepare_stream(streams["trumpet"], "trumpet", times, anchor, trumpet_solo)
    assert not np.any((masked_stream[:, 2] >= 32.) & (masked_stream[:, 2] < 40.))
    masked = prepare_all(streams, times, anchor, trumpet_solo)
    for method in matchers.METHODS:
        assert not masked[method].available[(times >= 16.) & (times < 20.)].any()

    # A static written melody may be known in full; only query notes are causal.
    written = np.column_stack((onsets[:18]*2., pitch[:18], onsets[:18]*2.))
    full_written = prepare_all(streams, times, anchor, plan, written_notes=written)
    short_written = prepare_all(short_streams, times[:stop], short_anchor, plan, written_notes=written)
    for method in matchers.METHODS:
        assert np.array_equal(full_written[method].available[:stop], short_written[method].available)
        assert not full_written[method].available[times < onsets[8]].any()
        for i in range(stop):
            position = np.arange(0., 48., .5)
            tempi = np.full(len(position), 120.)
            assert np.array_equal(full_written[method](i, times[i], position, tempi),
                                  short_written[method](i, times[i], position, tempi))
    assert full_written["pc_edit"].available.any()

    # Future note-offs are irrelevant; no duration-derived filtering is used.
    unknown_offsets = streams["trumpet"].copy()
    unknown_offsets[:, 1] = np.nan
    assert np.array_equal(prepare_stream(unknown_offsets, "trumpet", times, anchor, plan),
                          prepare_stream(streams["trumpet"], "trumpet", times, anchor, plan))
    assert similarity_threshold(np.array([1.])) == .75
    assert similarity_threshold(np.full(7, .9)) == .75
    assert similarity_threshold(np.full(8, .9)) == .9
    return {"past_memory_prefix": True, "written_query_prefix": True,
            "nonoverlap_warmup": True, "earliest_possible_reference_pair_seconds": float(earliest_pair),
            "repeat_occurrences_preserved": True, "specific_soloist_mask": True,
            "current_role_cache_mask": True, "future_noteoffs_ignored": True,
            "sparse_reference_threshold": True, "freshness_expiration": True}


if __name__ == "__main__":
    print(checks())
