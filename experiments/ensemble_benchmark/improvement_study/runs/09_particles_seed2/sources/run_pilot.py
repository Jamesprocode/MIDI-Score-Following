"""Small causal chord-compatibility diagnostic; this is not a position tracker.

Requires NumPy. Reads original MIDI and official annotations without modifying them.
Run with the bundled Python, or any Python with NumPy, from any directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SONGS = ["00-a_night_in_tunisia", "07-blue_bossa", "14-freddie_freeloader"]
WINDOWS = [0.5, 1.0, 2.0, 4.0]
STEP = 0.1
PRIMARY_WINDOW = 2.0  # Declared before running; not selected by test accuracy.
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_class(name):
    return (PC[name[0]] + name[1:].count("#") - name[1:].count("b")) % 12


def truth(value):
    return value is True or value == "True"


def vlq(data, pos):
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 127)
        if byte < 128:
            return value, pos


def read_midi(path):
    """Read note events and the actual tempo map; never infer meter from this export."""
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntracks, ppq = struct.unpack(">HHH", data[8:14])
    assert fmt in (0, 1) and ppq < 32768
    pos = 8 + header_len
    notes, tempos, track_counts = [], [], []
    for track in range(ntracks):
        assert data[pos:pos + 4] == b"MTrk"
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        raw = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        cursor, tick, status, count = 0, 0, None, 0
        active = defaultdict(deque)
        while cursor < len(raw):
            delta, cursor = vlq(raw, cursor)
            tick += delta
            if raw[cursor] >= 128:
                status = raw[cursor]
                cursor += 1
            assert status is not None
            if status == 255:
                kind = raw[cursor]
                cursor += 1
                length, cursor = vlq(raw, cursor)
                payload = raw[cursor:cursor + length]
                cursor += length
                if kind == 81:
                    tempos.append((tick, int.from_bytes(payload, "big")))
                status = None
            elif status in (240, 247):
                length, cursor = vlq(raw, cursor)
                cursor += length
                status = None
            else:
                kind, channel = status >> 4, status & 15
                length = 1 if kind in (12, 13) else 2
                payload = raw[cursor:cursor + length]
                cursor += length
                assert len(payload) == length and all(x < 128 for x in payload)
                if kind == 9 and payload[1] > 0:
                    active[(channel, payload[0])].append(tick)
                elif kind == 8 or (kind == 9 and payload[1] == 0):
                    key = (channel, payload[0])
                    assert active[key], f"Unmatched note-off: {path}"
                    onset = active[key].popleft()
                    notes.append((onset, tick, payload[0]))
                    count += 1
        assert not any(active.values()), f"Unclosed note: {path}"
        track_counts.append(count)
    assert pos == len(data)
    tempos.sort()

    def seconds(tick):
        previous, micros, elapsed = 0, 500000, 0.0
        for change, new_micros in tempos:
            if change > tick:
                break
            elapsed += (change - previous) * micros / ppq / 1e6
            previous, micros = change, new_micros
        return elapsed + (tick - previous) * micros / ppq / 1e6

    array = np.array([(seconds(a), seconds(b), p) for a, b, p in notes], dtype=float)
    array = array[np.argsort(array[:, 0])]
    assert np.all(array[:, 1] >= array[:, 0])
    info = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "format": fmt, "tracks": ntracks, "note_counts_by_track": track_counts,
            "ppq": ppq, "tempo_events": tempos, "note_count": len(array),
            "first_note_seconds": float(array[0, 0]),
            "last_note_off_seconds": float(array[:, 1].max())}
    return array, info


def chord_template(chord, soft=False):
    """Use explicit annotated chord spelling, not the key-relative root_interval."""
    triads = {"maj": {0, 4, 7}, "min": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8}}
    degrees = set(triads[chord["triad"]])
    if truth(chord["is_sus2"]):
        degrees -= {3, 4}
        degrees.add(2)
    if truth(chord["is_sus4"]):
        degrees -= {3, 4}
        degrees.add(5)
    sevenths = {"none": None, "dom": 10, "min": 10, "maj": 11, "dim": 9}
    seventh = sevenths[chord["seventh"]]
    if seventh is not None:
        degrees.add(seventh)
    for field, interval in [("has_6", 9), ("has_9", 2), ("has_11", 5),
                            ("has_13", 9), ("has_add9", 2)]:
        if truth(chord[field]):
            degrees.add(interval)
    for field, natural, altered in [("alt_b5", 7, 6), ("alt_sharp5", 7, 8),
                                     ("alt_b9", 2, 1), ("alt_sharp9", 2, 3),
                                     ("alt_sharp11", 5, 6), ("alt_b11", 5, 4)]:
        if truth(chord[field]):
            degrees.discard(natural)
            degrees.add(altered)
    root = pitch_class(chord["root"])
    vector = np.full(12, 0.05 if soft else 0.0)
    if soft:
        # Fixed exploratory heuristic, not a published chord-scale model.
        # Explicit tones=1; unaltered 9/13 and minor/diminished 11=0.25; others=.05.
        optional = {2, 9}
        if chord["triad"] in ("min", "dim"):
            optional.add(5)
        if truth(chord["alt_b9"]) or truth(chord["alt_sharp9"]):
            optional.discard(2)
        if truth(chord["alt_sharp11"]) or truth(chord["alt_b11"]):
            optional.discard(5)
        for degree in optional:
            vector[(root + degree) % 12] = 0.25
    for degree in degrees:
        vector[(root + degree) % 12] = 1.0
    if truth(chord["is_slash"]):
        vector[pitch_class(chord["bass"])] = 1.0
    return vector


def features(notes, times, window, mode):
    """Trailing fixed-second windows. Ground-truth chord boundaries are never used.

    Duration accumulation sees only the part of a note already elapsed at time t.
    Reading its eventual offset from disk does not expose it in this feature.
    """
    output = np.zeros((len(times), 12))
    for onset, offset, pitch in notes:
        if mode == "onsets":
            weight = ((times >= onset) & (times - window < onset)).astype(float)
        else:
            weight = np.maximum(0.0, np.minimum(times, offset) - np.maximum(times - window, onset))
        output[:, int(pitch) % 12] += weight
    return output


def normalize_rows(values):
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def combine(streams, condition):
    if condition == "bass":
        return streams["bass"]
    if condition == "horns":
        return sum(value for key, value in streams.items() if key != "bass")
    if condition == "pooled":
        return sum(streams.values())
    # Each currently active instrument contributes equal total mass.
    return sum(normalize_rows(value) for value in streams.values())


def cosine_scores(values, templates):
    numerator = values @ templates.T
    denominator = np.linalg.norm(values, axis=1, keepdims=True) * np.linalg.norm(templates, axis=1)[None, :]
    return numerator / np.maximum(denominator, 1e-12)


def statistics(scores, values, truth_idx, roots, changed_recently, selection=None):
    if selection is None:
        selection = np.ones(len(scores), dtype=bool)
    scores, values, truth_idx = scores[selection], values[selection], truth_idx[selection]
    recent = changed_recently[selection]
    available = values.sum(axis=1) > 0
    winner = scores.argmax(axis=1)  # Deterministic alphabetical label tie break.
    correct = (winner == truth_idx) & available
    tied = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0, atol=1e-10)
    ties = tied.sum(axis=1)
    expected = tied[np.arange(len(scores)), truth_idx] / ties * available

    def average(v, mask=None):
        if mask is not None:
            v = v[mask]
        return float(v.mean()) if len(v) else None

    return {
        "frames": len(scores), "coverage": average(available),
        "chord_accuracy_all_frames": average(correct),
        "chord_accuracy_active_frames": average(correct, available),
        "root_accuracy_all_frames": average((roots[winner] == roots[truth_idx]) & available),
        "tied_top_score_rate_active": average(ties > 1, available),
        "tie_randomized_expected_accuracy_all_frames": average(expected),
        "unique_correct_all_frames": average(correct & (ties == 1)),
        "accuracy_first_0p5s_after_change": average(correct, recent),
        "accuracy_after_0p5s": average(correct, ~recent),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checks():
    # This task merits checks: future leakage or root-transposition bugs invalidate results.
    notes = np.array([[0.25, 5.0, 60], [1.8, 2.2, 64], [8.0, 9.0, 67]])
    for mode in ("onsets", "elapsed_duration"):
        now = 2.0
        full = features(notes, np.array([now]), 1.0, mode)
        prefix = notes[notes[:, 0] <= now].copy()
        prefix[:, 1] = np.minimum(prefix[:, 1], now)
        past_only = features(prefix, np.array([now]), 1.0, mode)
        assert np.allclose(full, past_only), "Future event/duration leakage"
    duration = features(notes, np.array([2.0]), 1.0, "elapsed_duration")
    assert np.isclose(duration[0, 0], 1.0) and np.isclose(duration[0, 4], 0.2)
    assert pitch_class("Eb") == 3 and pitch_class("Db") == 1 and pitch_class("F#") == 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi-root", type=Path, default=HERE.parents[1] / "async" / "songs")
    args = parser.parse_args()
    checks()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    all_metrics, section_metrics, predictions, audit, song_details = [], [], [], [], []
    started = time.perf_counter()
    for song in SONGS:
        annotation = json.loads((HERE / "data" / "annotations" / f"{song}.json").read_text())
        chords = [x for x in annotation["chords"] if float(x["end_time"]) > float(x["start_time"])
                  and not truth(x["is_no_chord"])]
        assert all(float(a["end_time"]) <= float(b["start_time"]) + 1e-8 for a, b in zip(chords, chords[1:]))
        starts = np.array([float(c["start_time"]) for c in chords])
        ends = np.array([float(c["end_time"]) for c in chords])
        # Fixed global grid, not synchronized to label changes. Padding is excluded by evaluation mask.
        times = np.arange(0.0, max(ends), STEP)
        idx = np.searchsorted(starts, times, side="right") - 1
        valid = (idx >= 0) & (times < ends[np.maximum(idx, 0)])
        times, idx = times[valid], idx[valid]
        label_definitions = {chord["chord"]: chord for chord in chords}
        labels = sorted(label_definitions)
        definitions = [label_definitions[label] for label in labels]
        roots = np.array([pitch_class(c["root"]) for c in definitions])
        truth_idx = np.array([labels.index(chords[i]["chord"]) for i in idx])
        # Adjacent rows may repeat a label; only actual label changes count here.
        change_times = np.array([starts[i] for i in range(len(chords))
                                 if i == 0 or chords[i]["chord"] != chords[i-1]["chord"]])
        change_idx = np.searchsorted(change_times, times, side="right") - 1
        recent = times - change_times[change_idx] < 0.5
        sec_starts = np.array([float(s["start_time"]) for s in annotation["sections"]])
        sec_idx = np.searchsorted(sec_starts, times, side="right") - 1
        section_labels = np.array([annotation["sections"][max(i, 0)]["top_level_section"] for i in sec_idx])
        template_sets = {"binary_cosine": np.stack([chord_template(d) for d in definitions]),
                         "soft_extension_cosine": np.stack([chord_template(d, True) for d in definitions])}
        duplicate_masks = [[labels[i], labels[j]] for i in range(len(labels)) for j in range(i)
                           if np.array_equal(template_sets["binary_cosine"][i], template_sets["binary_cosine"][j])]
        freq = np.bincount(truth_idx, minlength=len(labels)) / len(truth_idx)
        song_details.append({"song": song, "labels": labels, "number_of_chords": len(labels),
                             "evaluation_start_s": float(times[0]), "evaluation_end_s": float(times[-1]),
                             "evaluation_frames_per_take": len(times), "uniform_guess_accuracy": 1 / len(labels),
                             "most_frequent_label": labels[int(freq.argmax())],
                             "largest_label_fraction_descriptive_not_trained_baseline": float(freq.max()),
                             "identical_binary_template_pairs": duplicate_masks,
                             "templates": {method: {label: vector.tolist() for label, vector in zip(labels, templates)}
                                           for method, templates in template_sets.items()}})
        for take in ("preferred", "alternate"):
            midi_files = sorted((args.midi_root / song / take).glob("*.mid"))
            assert len(midi_files) == 3, (song, take, midi_files)
            notes_by_instrument = {}
            for midi in midi_files:
                instrument = midi.name.split(".")[0]
                notes, info = read_midi(midi)
                notes_by_instrument[instrument] = notes
                audit.append({"song": song, "take": take, "instrument": instrument, **info})
            for feature in ("onsets", "elapsed_duration"):
                for window in WINDOWS:
                    streams = {key: features(notes, times, window, feature) for key, notes in notes_by_instrument.items()}
                    for condition in ("bass", "horns", "pooled", "equal_instruments"):
                        values = combine(streams, condition)
                        for method, templates in template_sets.items():
                            scores = cosine_scores(values, templates)
                            meta = {"song": song, "take": take, "feature": feature, "window_seconds": window,
                                    "input": condition, "method": method}
                            stats = statistics(scores, values, truth_idx, roots, recent)
                            all_metrics.append({**meta, **stats})
                            if window == PRIMARY_WINDOW and feature == "onsets":
                                for section in sorted(set(section_labels)):
                                    section_metrics.append({**meta, "section": section,
                                                            **statistics(scores, values, truth_idx, roots, recent, section_labels == section)})
                                winner = scores.argmax(axis=1)
                                sorted_scores = np.sort(scores, axis=1)
                                available = values.sum(axis=1) > 0
                                for i, t in enumerate(times):
                                    predictions.append({**meta, "time_seconds": round(float(t), 6),
                                                        "section": section_labels[i], "true_chord": labels[truth_idx[i]],
                                                        "predicted_chord": labels[winner[i]] if available[i] else "ABSTAIN",
                                                        "available": int(available[i]), "top_score": float(sorted_scores[i, -1]),
                                                        "top_minus_second": float(sorted_scores[i, -1] - sorted_scores[i, -2])})
    write_csv(out / "metrics.csv", all_metrics)
    write_csv(out / "by_section.csv", section_metrics)
    write_csv(out / "predictions_primary.csv", predictions)
    (out / "midi_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "task": "chart-constrained current chord compatibility; not bar-position tracking",
        "step_seconds": STEP, "primary_window_seconds": PRIMARY_WINDOW,
        "primary_feature": "onsets", "windows_seconds": WINDOWS,
        "training": "none", "parameter_selection": "fixed exploratory choices, no optimization on these songs",
        "annotations": "official demo JSONs; preferred annotation grid also applies to alternate async takes per release",
        "alignment": "release zero-time convention, no time-shift fitted against labels; no independent audio verification",
        "evaluation": "all valid labeled frames including unavailable input (abstention counted incorrect)",
        "ties": "alphabetical deterministic argmax; tie rate and randomized-tie expectation also reported",
        "ground_truth_use": "chart label vocabulary/templates and evaluation only; no timed boundaries used by features/matcher",
        "split": "three deliberately selected exploratory songs; both takes; not a held-out benchmark",
        "songs": song_details, "metric_rows": len(all_metrics), "prediction_rows": len(predictions),
        "runtime_seconds_entire_offline_diagnostic": time.perf_counter() - started,
        "checks_passed": ["future-note and future-duration invariance", "duration overlap arithmetic", "absolute root parsing",
                           "MIDI note pairing", "nonoverlapping chord labels", "expected three instrument files per take"]
    }, indent=2) + "\n")
    print(json.dumps({"results": str(out), "metric_rows": len(all_metrics), "midi_files": len(audit),
                      "evaluation_minutes_both_takes": sum(s["evaluation_frames_per_take"] for s in song_details)*STEP*2/60}, indent=2))


if __name__ == "__main__":
    main()
