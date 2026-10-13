"""Descriptive, reproducible tune analysis for the 75 preferred JazzSAMBA songs.

Requires NumPy. Reads baseline data/results; writes only analysis artifacts.
Extra per-song position CSVs can be supplied with repeated --metrics PATH.
Reference-derived tempo/accuracy fields are explanatory outcomes, never proposed
online model inputs. Rank associations are exploratory and have no p-values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BENCH))
import common

HARMONIC_METHODS = ("cosine_onsets", "cosine_duration", "pardo", "yamaha_active", "empirical_profile")
OUTCOMES = ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats")
FEATURE_DEFINITIONS = {}


def register(name, group, definition):
    FEATURE_DEFINITIONS[name] = {"group": group, "definition": definition}
    return name


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = [float(value) for value in values if finite(value) is not None]
    return float(np.mean(values)) if values else None


def median(values):
    values = [float(value) for value in values if finite(value) is not None]
    return float(np.median(values)) if values else None


def pct(value):
    return "—" if finite(value) is None else f"{100 * float(value):.1f}%"


def fmt(value):
    return "—" if finite(value) is None else f"{float(value):.3f}"


def ranks(values):
    """Average ranks for ties, without a SciPy dependency."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2.
        start = end
    return output


def rank_association(x, y):
    pairs = [(finite(a), finite(b)) for a, b in zip(x, y)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 8:
        return len(pairs), None
    a, b = np.asarray(pairs).T
    a, b = ranks(a), ranks(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return len(pairs), None
    return len(pairs), float(np.corrcoef(a, b)[0, 1])


def repeated_ngram_fraction(sequence, length):
    """Fraction of valid start positions whose exact n-gram occurs elsewhere."""
    grams = [tuple(sequence[i:i + length]) for i in range(len(sequence) - length + 1)]
    if not grams:
        return None
    counts = Counter(grams)
    return sum(counts[gram] > 1 for gram in grams) / len(grams)


def bar_pattern(keys, starts, ends, begin, end):
    """Canonical reduced harmony inside a bar, coalescing redundant boundaries."""
    tokens = []
    for key, a, b in zip(keys, starts, ends):
        if a >= end or b <= begin:
            continue
        token = (common.label_string(key), round(max(a, begin) - begin, 6), round(min(b, end) - begin, 6))
        if tokens and tokens[-1][0] == token[0] and abs(tokens[-1][2] - token[1]) < 1e-6:
            tokens[-1] = (token[0], tokens[-1][1], token[2])
        else:
            tokens.append(token)
    return tuple(tokens)


def note_features(notes, times, start, end, prefix):
    notes = np.asarray(notes, float).reshape(-1, 3)
    positive = notes[notes[:, 1] > notes[:, 0]]
    onsets = np.sort(positive[:, 0])
    offsets = np.sort(positive[:, 1])
    active = np.searchsorted(onsets, times, side="right") - np.searchsorted(offsets, times, side="right")
    recent = np.searchsorted(onsets, times, side="right") - np.searchsorted(onsets, times - 1., side="right")
    within = positive[(positive[:, 0] >= start) & (positive[:, 0] < end)]
    durations = within[:, 1] - within[:, 0]
    overlap = positive[(positive[:, 0] < end) & (positive[:, 1] > start)]
    events = [(max(start, a), 1) for a, b, _ in overlap] + [(min(end, b), -1) for a, b, _ in overlap]
    active_count = maximum = 0
    for _, delta in sorted(events):
        active_count += delta
        maximum = max(maximum, active_count)
    definitions = {
        "note_count": "Number of note-on events during musical annotation coverage; simultaneous notes count separately.",
        "onsets_per_second": "Note-on count during musical annotation coverage divided by its elapsed seconds; simultaneous notes count separately.",
        "active_coverage": "Fraction of 200-ms grid instants with at least one sounding note, preserving instrument multiplicity.",
        "onset_coverage_1s": "Fraction of grid instants with a note-on in the causal trailing one-second window.",
        "maximum_polyphony": "Maximum overlapping note instances during musical annotation coverage; separate instruments/unisons count separately.",
        "median_note_seconds": "Median complete exported duration of notes starting inside musical annotation coverage; diagnostic only.",
        "p95_note_seconds": "95th percentile exported duration of those notes; diagnostic only.",
        "note_fraction_over_10s": "Fraction of those notes with exported duration exceeding ten seconds; diagnostic only.",
    }
    for suffix, definition in definitions.items():
        register(f"{prefix}_{suffix}", "MIDI content summary", definition)
    row = {
        f"{prefix}_note_count": len(within),
        f"{prefix}_onsets_per_second": len(within) / (end - start),
        f"{prefix}_active_coverage": float((active > 0).mean()),
        f"{prefix}_onset_coverage_1s": float((recent > 0).mean()),
        f"{prefix}_maximum_polyphony": maximum,
        f"{prefix}_median_note_seconds": float(np.median(durations)) if len(durations) else None,
        f"{prefix}_p95_note_seconds": float(np.percentile(durations, 95)) if len(durations) else None,
        f"{prefix}_note_fraction_over_10s": float((durations > 10).mean()) if len(durations) else None,
    }
    return row, {"active": active > 0, "recent": recent > 0}


def harmonic_clarity(notes, times, chords, start, end, prefix):
    """Reference-conditioned diagnostic; never an inference feature.

    Onsets use their exact timestamps, with [start,end) chord boundaries.
    Each complete positive-duration MIDI note counts once, including simultaneous
    notes and notes outside the 88-key scorer range. N.C./unannotated instants
    are excluded from harmonic denominators and explicitly counted.
    """
    notes = np.asarray(notes, float).reshape(-1, 3)
    notes = notes[notes[:, 1] > notes[:, 0]]
    chord_starts = np.array([float(chord["start_time"]) for chord in chords])
    chord_ends = np.array([float(chord["end_time"]) for chord in chords])
    keys = [common.quality_id(chord, "triad") for chord in chords]
    roots = np.array([-1 if key is None else key[0] for key in keys], dtype=int)
    templates = np.array([np.zeros(12) if key is None else common.chord_template(*key) for key in keys])
    if len(chords) == 0:
        raise ValueError("Harmonic-clarity diagnostics require reference chord intervals")
    if np.any(chord_starts[1:] < chord_ends[:-1] - 1e-8):
        raise ValueError("Reference chord intervals overlap")

    def reference_at(instants):
        indices = np.searchsorted(chord_starts, instants, side="right") - 1
        safe = np.maximum(indices, 0)
        valid = (indices >= 0) & (instants < chord_ends[safe]) & (roots[safe] >= 0)
        return safe, valid

    onsets = notes[(notes[:, 0] >= start) & (notes[:, 0] < end)]
    onset_indices, onset_valid = reference_at(onsets[:, 0])
    onset_pc = onsets[:, 2].astype(int) % 12
    onset_count = int(onset_valid.sum())
    root_onsets = int(((onset_pc == roots[onset_indices]) & onset_valid).sum())
    tone_onsets = int(((templates[onset_indices, onset_pc] > 0) & onset_valid).sum())
    frame_indices, frame_valid = reference_at(times)
    active_pc = np.zeros((len(times), 12), bool)
    for pc in range(12):
        selected = notes[notes[:, 2].astype(int) % 12 == pc]
        starts, ends = np.sort(selected[:, 0]), np.sort(selected[:, 1])
        active_pc[:, pc] = (np.searchsorted(starts, times, side="right") - np.searchsorted(ends, times, side="right")) > 0
    active_valid = frame_valid & active_pc.any(axis=1)
    root_active = active_pc[np.arange(len(times)), np.maximum(roots[frame_indices], 0)] & active_valid
    active_count = int(active_valid.sum())
    definitions = {
        "reference_harmony_onset_count": "Number of positive-duration MIDI note-ons inside musical coverage with an active pitched reference chord. Each simultaneous note counts separately; N.C./annotation gaps excluded.",
        "reference_harmony_excluded_onset_count": "Note-ons inside musical coverage excluded because the timestamp is N.C. or lacks a pitched chord annotation.",
        "reference_root_onset_fraction": "Fraction of eligible note-ons whose pitch class equals the annotated chord root at onset. Exact half-open timestamp intervals; N.C./gaps excluded; slash bass ignored. All exported pitches count, including outside88keys. Evaluation-only.",
        "reference_base_tone_onset_fraction": "Fraction of eligible note-ons inside the annotated base chord template (major/minor/diminished/augmented/sus2/sus4/power/root as canonicalized). Sevenths, extensions, alterations and slash bass ignored. Same event denominator as root fraction. Evaluation-only.",
        "reference_harmony_active_frames": "Number of 200-ms grid instants with a pitched reference chord and at least one active note in this role. N.C./gaps and silence excluded.",
        "reference_root_active_fraction": "Fraction of eligible active-role frames containing at least one note at the annotated root pitch class. Denominator excludes silence, N.C. and gaps; all exported pitches count. Note state uses onset<=time<offset. Evaluation-only.",
    }
    values = {"reference_harmony_onset_count": onset_count,
        "reference_harmony_excluded_onset_count": int(len(onsets) - onset_count),
        "reference_root_onset_fraction": root_onsets / onset_count if onset_count else None,
        "reference_base_tone_onset_fraction": tone_onsets / onset_count if onset_count else None,
        "reference_harmony_active_frames": active_count,
        "reference_root_active_fraction": int(root_active.sum()) / active_count if active_count else None}
    return {register(f"{prefix}_{name}", "reference harmonic clarity: evaluation-only", definitions[name]): value for name, value in values.items()}


def feature_row(entry, data, catalog, flagged_files):
    times = data["times"][::2]
    duration = data["end_time"] - data["start_time"]
    row = {"performance_id": entry["performance_id"], "song": entry["song"], "protocol": entry["protocol"],
           "take": "preferred", "title": catalog["title"], "genre": catalog.get("genre", ""),
           "form": catalog.get("form", ""), "meter": entry["time_signature"], "catalog_bpm": data["initial_bpm"],
           "instrument_names": ";".join(sorted(data["notes_by_instrument"]))}
    for name, group, definition, value in [
        ("duration_seconds", "reference structure", "Elapsed musical annotation coverage, excluding leading padding.", duration),
        ("bars", "chart structure", "Number of unfolded musical bars, including all planned chorus/repeat occurrences.", len(data["bars"])),
        ("beats_per_bar", "chart structure", "Quarter-note beats per bar from catalog meter.", data["beats_per_bar"]),
        ("catalog_bpm", "catalog metadata", "Catalog nominal quarter-note BPM supplied for initialization.", data["initial_bpm"]),
        ("pitched_instruments", "MIDI content summary", "Number of locally available pitched instrument streams; excludes drums.", len(data["notes_by_instrument"])),
    ]:
        row[register(name, group, definition)] = value
    durations = np.array([float(b["end_time"]) - float(b["start_time"]) for b in data["bars"]])
    tempi = 60. * data["beats_per_bar"] / durations
    med_bpm = float(np.median(tempi))
    for name, definition, value in [
        ("reference_bar_bpm_median", "Median bar-equivalent BPM: 60 times catalog quarter beats per bar divided by annotated bar seconds. Evaluation-only.", med_bpm),
        ("reference_bar_bpm_cv", "Population SD/mean of bar-equivalent BPM; sensitive to pickups, endings and interpolation. Evaluation-only.", float(tempi.std() / tempi.mean())),
        ("reference_catalog_bpm_abs_fraction", "Absolute fractional difference between reference median bar-equivalent BPM and catalog BPM. Evaluation-only.", abs(med_bpm / data["initial_bpm"] - 1)),
        ("reference_catalog_bpm_signed_fraction", "Signed fractional difference: reference median bar-equivalent BPM/catalog BPM minus one. Evaluation-only.", med_bpm / data["initial_bpm"] - 1),
        ("reference_final_clock_mismatch_beats", "Absolute difference between total score beats and duration times catalog BPM/60. Directly related to clock drift; evaluation-only.", abs(data["score_end_beats"] - duration * data["initial_bpm"] / 60)),
    ]:
        row[register(name, "reference timing: evaluation-only", definition)] = value
    for scheme in ("triad", "seventh"):
        keys = [common.quality_id(c, scheme) for c in data["chords"]]
        sequence = []
        masses = Counter()
        for key, begin, end in zip(keys, data["chord_starts_beats"], data["chord_ends_beats"]):
            token = common.label_string(key)
            if not sequence or sequence[-1] != token:
                sequence.append(token)
            if key is not None:
                masses[token] += end - begin
        row[register(f"{scheme}_vocabulary", "chart harmony", "Number of distinct pitched reduced chart labels.")] = len(masses)
        total = sum(masses.values())
        shares = np.array(list(masses.values())) / total
        row[register(f"{scheme}_chord_entropy_bits", "chart harmony", "Shannon entropy of chord labels weighted by musical beats in the unfolded plan.")] = float(-(shares * np.log2(shares)).sum())
        row[register(f"{scheme}_largest_chord_beat_share", "chart harmony", "Largest chord-class share of pitched musical score beats.")] = float(shares.max())
        changes = sum(a != b and a != "N.C." and b != "N.C." for a, b in zip(sequence, sequence[1:]))
        row[register(f"{scheme}_changes_per_minute", "chart harmony plus reference duration", "Actual changes between successive reduced pitched labels per minute of reference duration; adjacent identical labels merged.")] = changes / duration * 60
        row[f"{scheme}_collapsed_chord_events"] = len(sequence)
        for n in (1, 2, 4, 8):
            row[register(f"{scheme}_repeated_{n}gram_fraction", "chart ambiguity", f"Fraction of start positions whose exact {n}-event chord-label sequence appears elsewhere in the unfolded plan; adjacent identical labels merged, N.C. retained.")] = repeated_ngram_fraction(sequence, n)
        bar_tokens = []
        for begin in data["bar_starts_beats"]:
            end = begin + data["beats_per_bar"]
            tokens = bar_pattern(keys, data["chord_starts_beats"], data["chord_ends_beats"], begin, end)
            bar_tokens.append(tokens)
        row[register(f"{scheme}_unique_bar_fraction", "chart ambiguity", "Distinct exact within-bar reduced harmonic patterns divided by unfolded bars; includes chord durations, merging adjacent identical reduced labels.")] = len(set(bar_tokens)) / len(bar_tokens)
        row[register(f"{scheme}_repeated_4bar_fraction", "chart ambiguity", "Fraction of four-bar start positions whose complete timed-in-score harmonic pattern occurs elsewhere.")] = repeated_ngram_fraction(bar_tokens, 4)
    quality = flagged_files.get((entry["protocol"], entry["song"], "preferred"), [])
    row[register("quality_flagged_pitched_files", "MIDI quality audit", "Number of fixed-audit flagged pitched files: polyphony>4, any note>10s, or outside88keys; no filtering.")] = len(quality)
    row["quality_flagged_instruments"] = ";".join(sorted({x["instrument"] for x in quality}))
    availability = {}
    for role in ("pooled", "bass", "horns"):
        arrays = [n for instrument, n in data["notes_by_instrument"].items()
                  if role == "pooled" or (role == "bass" and instrument == "bass") or (role == "horns" and instrument != "bass")]
        notes = np.concatenate(arrays) if arrays else np.empty((0, 3))
        values, availability[role] = note_features(notes, times, data["start_time"], data["end_time"], role)
        row.update(values)
        if role in ("bass", "horns"):
            row.update(harmonic_clarity(notes, times, data["chords"], data["start_time"], data["end_time"], role))
    return row, availability


def position_rows(path, source):
    rows = read_csv(path)
    needed = {"performance_id", "song", "protocol", "take", "scheme", "initial_tempo_factor", "method", *OUTCOMES}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"{path} must use the baseline per-song position metric schema: missing {needed - set(rows[0] if rows else {})}")
    selected, seen = [], set()
    for row in rows:
        if row["take"] != "preferred" or float(row["initial_tempo_factor"]) != 1.:
            continue
        key = (row["protocol"], row["song"], row["scheme"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate nominal preferred method condition in {path}: {key}. Give variants distinct method names.")
        seen.add(key)
        result = {**row, "source": source, "metrics_path": str(path)}
        for name in OUTCOMES:
            value = finite(row[name])
            if value is None:
                raise ValueError(f"Nonfinite {name} in {path}: {key}")
            result[name] = value
        selected.append(result)
    return selected


def section_rows(entry, data, availability, metrics, warnings):
    """Use saved baseline trajectories, verifying totals against saved metrics."""
    output = []
    times = data["times"][::2]
    for metric in metrics:
        if metric["song"] != entry["song"] or metric["protocol"] != entry["protocol"]:
            continue
        if metric["source"] == "baseline":
            trace = BENCH / "results" / "position_traces" / entry["protocol"] / entry["song"] / f"{metric['scheme']}_{metric['method']}.npz"
        else:
            trace = Path(metric["metrics_path"]).parent / "traces" / entry["protocol"] / entry["song"] / "preferred" / f"{metric['method']}.npz"
        if not trace.exists():
            continue
        with np.load(trace) as stored:
            t = stored["time"]
            if t.shape != times.shape or not np.allclose(t, times, atol=1e-8, rtol=0):
                warnings.append(f"Trace time mismatch, omitted sections: {trace}")
                continue
            correct = stored["estimated_bar"] == stored["reference_bar"]
            error = np.abs(stored["estimated_beats"] - stored["reference_beats"])
        if not np.isclose(correct.mean(), metric["bar_accuracy"], atol=1e-10) or not np.isclose((error <= 2).mean(), metric["within2beat_accuracy"], atol=1e-10):
            raise ValueError(f"Saved trajectory disagrees with baseline metric: {trace}")
        assigned = np.zeros(len(t), int)
        for section in data["sections"]:
            start, end = float(section["start_time"]), float(section["end_time"])
            mask = (t >= start) & (t < end)
            if not mask.any():
                continue
            assigned += mask
            solo = common.truth(section.get("is_solo", False))
            role = section.get("soloist", "NA") if solo else "NA"
            kind = "solo" if solo else section.get("top_level_section", section["section"])
            output.append({"source": metric["source"], "protocol": entry["protocol"], "song": entry["song"],
                "scheme": metric["scheme"], "method": metric["method"], "section": section["section"],
                "section_kind": kind, "solo_role": role, "frames": int(mask.sum()),
                "bar_correct_frames": int(correct[mask].sum()), "within2beat_frames": int((error[mask] <= 2).sum()),
                "bar_accuracy": float(correct[mask].mean()), "within2beat_accuracy": float((error[mask] <= 2).mean()),
                "mean_abs_beats": float(error[mask].mean()),
                "mean_fraction_through_song": float(((t[mask] - data["start_time"]) / (data["end_time"] - data["start_time"])).mean()),
                "pooled_active_coverage": float(availability["pooled"]["active"][mask].mean()),
                "bass_active_coverage": float(availability["bass"]["active"][mask].mean()),
                "horns_onset_coverage_1s": float(availability["horns"]["recent"][mask].mean())})
        if np.any(assigned > 1):
            raise ValueError(f"Overlapping structural section annotations: {entry['song']}")
    return output


def section_summary(rows):
    # Pool repeated sections within a song first; then average songs equally.
    per_song = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("source", "protocol", "scheme", "method", "section_kind", "solo_role", "song"))
        per_song[key].append(row)
    groups = defaultdict(list)
    for key, values in per_song.items():
        frames = sum(row["frames"] for row in values)
        groups[key[:-1]].append({"frames": frames,
            "bar_accuracy": sum(row["bar_correct_frames"] for row in values) / frames,
            "within2beat_accuracy": sum(row["within2beat_frames"] for row in values) / frames,
            **{name: sum(row[name] * row["frames"] for row in values) / frames for name in
               ("mean_fraction_through_song", "pooled_active_coverage", "bass_active_coverage", "horns_onset_coverage_1s")}})
    return [{**dict(zip(("source", "protocol", "scheme", "method", "section_kind", "solo_role"), key)),
             "songs": len(values), "frames": sum(row["frames"] for row in values),
             **{name: mean(row[name] for row in values) for name in values[0] if name != "frames"}}
            for key, values in sorted(groups.items())]


def analyze_metrics(metrics, features):
    lookup = {(row["protocol"], row["song"]): row for row in features}
    baseline = {(row["protocol"], row["song"], row["scheme"], row["method"]): row
                for row in metrics if row["source"] == "baseline"}
    comparison = []
    groups = defaultdict(list)
    for row in metrics:
        if (row["protocol"], row["song"]) not in lookup:
            raise ValueError(f"Extra metric song is outside preferred feature inventory: {row['song']}")
        value = dict(row)
        for method in ("clock", "uniform_filter", "empirical_profile"):
            reference = baseline.get((row["protocol"], row["song"], row["scheme"], method))
            if reference:
                value[f"baseline_{method}_bar_accuracy"] = reference["bar_accuracy"]
                value[f"delta_bar_vs_baseline_{method}"] = row["bar_accuracy"] - reference["bar_accuracy"]
        comparison.append(value)
        groups[(row["source"], row["protocol"], row["scheme"], row["method"])].append(value)
        feature = lookup[(row["protocol"], row["song"])]
        for name in OUTCOMES:
            feature[f"outcome__{row['source']}__{row['scheme']}__{row['method']}__{name}"] = row[name]
    distributions, high_low, associations, group_features = [], [], [], []
    fields = ("source", "protocol", "scheme", "method")
    for key, rows in sorted(groups.items()):
        meta = dict(zip(fields, key))
        scores = np.array([row["bar_accuracy"] for row in rows])
        distribution = {**meta, "songs": len(rows), "bar_accuracy_mean": float(scores.mean()),
            "bar_accuracy_median": float(np.median(scores)), "bar_accuracy_p25": float(np.percentile(scores, 25)),
            "bar_accuracy_p75": float(np.percentile(scores, 75)), "bar_accuracy_min": float(scores.min()),
            "bar_accuracy_max": float(scores.max()), "songs_at_least_80pct": int((scores >= .8).sum()),
            "songs_at_least_90pct": int((scores >= .9).sum()), "songs_at_most_25pct": int((scores <= .25).sum()),
            **{name + "_song_mean": mean(row[name] for row in rows) for name in OUTCOMES if name != "bar_accuracy"}}
        for method in ("clock", "uniform_filter", "empirical_profile"):
            deltas = [row[f"delta_bar_vs_baseline_{method}"] for row in rows if f"delta_bar_vs_baseline_{method}" in row]
            distribution[f"mean_delta_bar_vs_baseline_{method}"] = mean(deltas)
            distribution[f"songs_beating_baseline_{method}"] = sum(value > 1e-10 for value in deltas)
        distributions.append(distribution)
        ordered = sorted(rows, key=lambda row: (-row["bar_accuracy"], row["song"]))
        for side, chosen in (("highest_5", ordered[:5]), ("lowest_5", list(reversed(ordered[-5:])))):
            for rank, row in enumerate(chosen, 1):
                f = lookup[(row["protocol"], row["song"])]
                high_low.append({**meta, "group": side, "rank": rank, "song": row["song"], "title": f["title"],
                    "bar_accuracy": row["bar_accuracy"], "within2beat_accuracy": row["within2beat_accuracy"],
                    **{name: f[name] for name in ("genre", "duration_seconds", "bars", "meter", "catalog_bpm",
                        "reference_bar_bpm_median", "reference_bar_bpm_cv", "reference_catalog_bpm_abs_fraction",
                        "triad_vocabulary", "triad_unique_bar_fraction", "triad_repeated_4bar_fraction",
                        "pooled_active_coverage", "quality_flagged_pitched_files")}})
        feature_rows = [lookup[(row["protocol"], row["song"])] for row in rows]
        for feature, definition in sorted(FEATURE_DEFINITIONS.items()):
            n, rho = rank_association([f.get(feature) for f in feature_rows], scores)
            associations.append({**meta, "feature": feature, "feature_group": definition["group"], "songs": n,
                                 "spearman_rho": rho, "status": "descriptive" if rho is not None else "constant_or_insufficient_data"})
            order = np.array(sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["song"])))
            top = np.zeros(len(scores), bool)
            bottom = np.zeros(len(scores), bool)
            top[order[:5]] = True
            bottom[order[-5:]] = True
            for name, mask in (("high_at_least_80pct", scores >= .8), ("low_at_most_25pct", scores <= .25),
                               ("highest_5", top), ("lowest_5", bottom)):
                values = [f.get(feature) for f, use in zip(feature_rows, mask) if use]
                group_features.append({**meta, "group": name, "feature": feature,
                                       "songs": int(mask.sum()), "feature_median": median(values)})
    return comparison, distributions, high_low, associations, group_features


def paired_changes(metrics, features, references, cross_run=False):
    """Matched song/scheme contrasts; optional unique cross-run references."""
    feature_lookup = {(r["protocol"], r["song"]): r for r in features}
    metric_lookup = {(r["source"], r["protocol"], r["song"], r["scheme"], r["method"]): r for r in metrics}
    changes, grouped = [], defaultdict(list)
    for row in metrics:
        for reference in references:
            if row["method"] == reference:
                continue
            ref = metric_lookup.get((row["source"], row["protocol"], row["song"], row["scheme"], reference))
            if ref is None and cross_run:
                candidates = [r for r in metrics if r["protocol"] == row["protocol"] and r["song"] == row["song"]
                              and r["scheme"] == row["scheme"] and r["method"] == reference]
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous cross-run reference {reference}: {row['protocol']}/{row['song']}/{row['scheme']}")
                ref = candidates[0] if candidates else None
            if ref is None:
                continue
            feature = feature_lookup[(row["protocol"], row["song"])]
            value = {name: row[name] for name in ("source", "protocol", "song", "scheme", "method")}
            value.update(title=feature["title"], reference_method=reference, reference_source=ref["source"],
                         comparison_scope="same_run" if row["source"] == ref["source"] else "cross_run")
            for outcome in OUTCOMES:
                value[outcome] = row[outcome]
                value["reference_" + outcome] = ref[outcome]
                value["delta_" + outcome] = row[outcome] - ref[outcome]
            changes.append(value)
            grouped[(row["source"], row["protocol"], row["scheme"], row["method"], reference, ref["source"])].append(value)
    summaries, associations, extremes = [], [], []
    for key, rows in sorted(grouped.items()):
        meta = dict(zip(("source", "protocol", "scheme", "method", "reference_method", "reference_source"), key))
        meta["comparison_scope"] = "same_run" if meta["source"] == meta["reference_source"] else "cross_run"
        delta = np.array([r["delta_bar_accuracy"] for r in rows])
        summaries.append({**meta, "songs": len(rows), "improved_songs": int((delta > 1e-10).sum()),
            "worsened_songs": int((delta < -1e-10).sum()), "unchanged_songs": int((abs(delta) <= 1e-10).sum()),
            "mean_delta_bar_accuracy": float(delta.mean()), "median_delta_bar_accuracy": float(np.median(delta)),
            **{"mean_delta_" + name: mean(r["delta_" + name] for r in rows) for name in OUTCOMES if name != "bar_accuracy"}})
        ordered = sorted(rows, key=lambda r: (-r["delta_bar_accuracy"], r["song"]))
        for group, selected in (("largest_5_gains", ordered[:5]), ("smallest_5_changes", list(reversed(ordered[-5:])))):
            extremes += [{**row, "change_group": group, "rank": i} for i, row in enumerate(selected, 1)]
        for name, definition in sorted(FEATURE_DEFINITIONS.items()):
            n, rho = rank_association([feature_lookup[(r["protocol"], r["song"])].get(name) for r in rows], delta)
            associations.append({**meta, "feature": name, "feature_group": definition["group"], "songs": n,
                "spearman_rho": rho, "status": "descriptive" if rho is not None else "constant_or_insufficient_data"})
    return changes, summaries, associations, extremes


def md_table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows]) + "\n"


def report(distributions, high_low, associations, sections, focus, warnings, sources, features, paired, group_features):
    text = ["# Tune characteristics and bar-tracking outcomes\n",
        "This is a descriptive analysis of the 75 preferred performances, with async and sync kept separate. "
        "Chord recognition and position following are different tasks: a stronger chord classifier need not give a better tracking trajectory. "
        "The fixed primary view uses triad targets, nominal initial tempo, and the saved baseline settings. "
        "No per-song oracle chooses a winning method. Additional runs remain separately identified.\n",
        "Reference-derived tempo and harmonic clarity, note-duration summaries, complete-song structure, and accuracy fields are explanatory measurements. "
        "They are **not causal inference inputs** and must not be supplied to an online selector as if available in advance.\n",
        ]
    focus_configs = []
    for source in sources:
        metadata_path = Path(source["path"]).parent / "run.json"
        if source["source"] == "baseline" or not metadata_path.exists():
            continue
        config = json.loads(metadata_path.read_text())
        for setting in config.get("settings", []):
            if setting.get("method") == focus:
                focus_configs.append((source["source"], setting))
    if focus_configs:
        text.append(f"## Focused configuration: {focus}\n")
        for source, setting in focus_configs:
            text.append(f"Recorded source: `{source}`. Engine: `{setting.get('engine', 'unspecified')}`. Harmonic input: `{setting.get('feature', 'unspecified')}`. {setting.get('reason', '')}.\n")
        if any("bass" in setting.get("feature", "") for _, setting in focus_configs):
            text.append("This configuration receives the dataset's **bass instrument label** and uses that stream for harmonic evidence. It is not role-free inference. Reference harmonic clarity is measured afterward; the tracker does not receive those diagnostic values.\n")
    text.append("## Baseline distributions\n")
    for protocol in ("async", "sync"):
        selected = [r for r in distributions if r["source"] == "baseline" and r["protocol"] == protocol and r["scheme"] == "triad"]
        text.append(f"### {protocol}\n")
        text.append(md_table(["Method", "Songs", "Mean exact bar", "Median", "Middle 50%", "≥80% songs", "≥90% songs", "Within 2 beats"],
            [[r["method"], r["songs"], pct(r["bar_accuracy_mean"]), pct(r["bar_accuracy_median"]),
              f"{pct(r['bar_accuracy_p25'])}–{pct(r['bar_accuracy_p75'])}", r["songs_at_least_80pct"],
              r["songs_at_least_90pct"], pct(r["within2beat_accuracy_song_mean"])] for r in selected]))
    text.append("Async uses a fixed timing plan and a synchronized known start, so near-perfect clock tracking is expected. "
                "Clock performance therefore does not establish robust score following. The uniform-evidence filter controls for the shared filter's own dynamics.\n")
    text.append("### Scale of reference timing variation\n")
    text.append(md_table(["Protocol", "Median BPM CV", "BPM CV range", "Median catalog mismatch", "Maximum catalog mismatch"],
        [[protocol, fmt(median(r["reference_bar_bpm_cv"] for r in features if r["protocol"] == protocol)),
          f"{min(r['reference_bar_bpm_cv'] for r in features if r['protocol'] == protocol):.6f}–{max(r['reference_bar_bpm_cv'] for r in features if r['protocol'] == protocol):.6f}",
          pct(median(r["reference_catalog_bpm_abs_fraction"] for r in features if r["protocol"] == protocol)),
          pct(max(r["reference_catalog_bpm_abs_fraction"] for r in features if r["protocol"] == protocol))]
         for protocol in ("async", "sync")]))
    text.append("The tiny async variation is consistent with timestamp rounding on a fixed grid and must not be interpreted as measured expressive tempo drift. "
                "Rank correlations can look large even when a feature's physical range is negligible; inspect these ranges and `feature_ranges.csv`.\n")
    extra_sources = sorted({r["source"] for r in distributions if r["source"] != "baseline"})
    for source in extra_sources:
        text.append(f"## Additional run: {source}\n")
        text.append("These rows are paired with baseline outcomes for the same songs. If the additional run covers a subset, its percentages are not directly comparable to the full-cohort baseline mean; paired deltas use only its evaluated songs.\n")
        for protocol in ("async", "sync"):
            rows = [r for r in distributions if r["source"] == source and r["protocol"] == protocol and r["scheme"] == "triad"]
            if not rows:
                continue
            text.append(f"### {protocol}\n")
            text.append(md_table(["Method", "Songs", "Mean bar", "Median", "≥80%", "Δ vs clock", "Δ vs uniform", "Δ vs empirical baseline"],
                [[r["method"], r["songs"], pct(r["bar_accuracy_mean"]), pct(r["bar_accuracy_median"]), r["songs_at_least_80pct"],
                  pct(r["mean_delta_bar_vs_baseline_clock"]), pct(r["mean_delta_bar_vs_baseline_uniform_filter"]),
                  pct(r["mean_delta_bar_vs_baseline_empirical_profile"])] for r in rows]))
    _, paired_summaries, gain_associations, extremes = paired
    paired_focus = [r for r in paired_summaries if r["method"] == focus and r["scheme"] == "triad"]
    if paired_focus:
        text += [f"## Paired changes for {focus}\n",
            "Every contrast uses the same song, protocol, target scheme, preferred take and nominal initial tempo. Reference sources are recorded explicitly; cross-run contrasts compare separately executed fixed configurations. Bar/within-two-beat differences are percentage points; negative beat-error differences are improvements. "
            "These methods were explored on this same cohort, so this is not a held-out estimate of a selected winner.\n",
            md_table(["Source", "Protocol", "Reference", "Reference source", "Songs", "Bar change (pp)", "Median change (pp)", "Improved/worse/same", "Mean absolute error change (beats)"],
                [[r["source"], r["protocol"], r["reference_method"], r["reference_source"], r["songs"], f"{100*r['mean_delta_bar_accuracy']:+.2f}",
                  f"{100*r['median_delta_bar_accuracy']:+.2f}", f"{r['improved_songs']}/{r['worsened_songs']}/{r['unchanged_songs']}",
                  f"{r['mean_delta_mean_abs_beats']:+.2f}"] for r in paired_focus])]
        if focus == "g06_bass_root":
            text.append("This method receives the supplied **bass instrument label** and uses only that role's active pitches for harmony, with extra root weight and a bar-posterior decision rule. "
                "It is not role-free inference. Compared with g00_reproduce, three choices change together. The g01_bar_decision contrast holds the decision rule fixed; "
                "the g04_bass_active contrast additionally holds bass-only input fixed and isolates extra root weighting. No reference tempo or outcome features in this analysis enter the tracker.\n")
        first_reference = paired_focus[0]["reference_method"]
        # Prefer the explicit reproduction control when supplied.
        if any(r["reference_method"] == "g00_reproduce" for r in paired_focus):
            first_reference = "g00_reproduce"
        for source, protocol in sorted({(r["source"], r["protocol"]) for r in paired_focus}):
            text.append(f"### {source} · {protocol} · changes versus {first_reference}\n")
            rows = [r for r in extremes if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad" and r["reference_method"] == first_reference]
            text.append(md_table(["Group", "Tune", "Reference bar", "Current bar", "Change (pp)"],
                [[r["change_group"], r["title"], pct(r["reference_bar_accuracy"]), pct(r["bar_accuracy"]), f"{100*r['delta_bar_accuracy']:+.2f}"] for r in rows]))
            rows = [r for r in gain_associations if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad" and r["reference_method"] == first_reference and r["spearman_rho"] is not None]
            rows = sorted(rows, key=lambda r: -abs(r["spearman_rho"]))[:8]
            text.append("The following exploratory associations concern **change** in accuracy, which is distinct from associations with final accuracy. They are post hoc descriptions, not causes.\n")
            text.append(md_table(["Feature", "Group", "Songs", "Spearman ρ with bar change"],
                [[r["feature"], r["feature_group"], r["songs"], fmt(r["spearman_rho"])] for r in rows]))
    text += [f"## Highest and lowest tunes for {focus}\n",
             "These are the five highest and five lowest observed outcomes for one named method, not claims that a tune is intrinsically easy or difficult. "
             "Ties are broken by song ID. If many tunes reach 100%, the five shown are an arbitrary deterministic subset of that tied group; their feature medians do not represent every perfect tune. Full tables for every method and both target schemes are in `high_low_tunes.csv`.\n"]
    available_focus = sorted({(r["source"], r["protocol"]) for r in high_low if r["method"] == focus and r["scheme"] == "triad"})
    for source, protocol in available_focus:
        for group in ("highest_5", "lowest_5"):
            rows = [r for r in high_low if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad" and r["group"] == group]
            text.append(f"### {source} · {protocol} · {group.replace('_', ' ')}\n")
            text.append(md_table(["Tune", "Exact bar", "Within 2 beats", "Bars", "Reference BPM CV", "Catalog mismatch", "4-bar repetition", "Flagged files"],
                [[r["title"], pct(r["bar_accuracy"]), pct(r["within2beat_accuracy"]), r["bars"], fmt(r["reference_bar_bpm_cv"]),
                  pct(r["reference_catalog_bpm_abs_fraction"]), pct(r["triad_repeated_4bar_fraction"]), r["quality_flagged_pitched_files"]] for r in rows]))
        text.append(f"### {source} · {protocol} · characteristics of these two groups\n")
        text.append("Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.\n")
        feature_names = ("duration_seconds", "bars", "catalog_bpm", "reference_bar_bpm_cv", "reference_catalog_bpm_abs_fraction",
                         "triad_vocabulary", "triad_changes_per_minute", "triad_unique_bar_fraction", "triad_repeated_4bar_fraction",
                         "bass_onsets_per_second", "bass_active_coverage", "bass_median_note_seconds", "horns_note_fraction_over_10s", "quality_flagged_pitched_files",
                         "bass_reference_root_onset_fraction", "bass_reference_base_tone_onset_fraction", "bass_reference_root_active_fraction",
                         "horns_reference_root_onset_fraction", "horns_reference_base_tone_onset_fraction")
        found = {(r["group"], r["feature"]): r for r in group_features if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad"}
        text.append(md_table(["Feature (median)", "Highest five", "Lowest five"],
            [[name, fmt(found.get(("highest_5", name), {}).get("feature_median")), fmt(found.get(("lowest_5", name), {}).get("feature_median"))] for name in feature_names]))
        genres = []
        lookup = {(r["protocol"], r["song"]): r for r in features}
        for group in ("highest_5", "lowest_5"):
            selected = [lookup[(protocol, r["song"])] for r in high_low if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad" and r["group"] == group]
            genres.append([group, "; ".join(f"{name}: {count}" for name, count in sorted(Counter(r["genre"] for r in selected).items())),
                           "; ".join(f"{name}: {count}" for name, count in sorted(Counter(r["meter"] for r in selected).items()))])
        text.append(md_table(["Group", "Catalog genres (counts)", "Meters (counts)"], genres))
    text.append("## Reference harmonic clarity\n")
    text.append("These within-protocol Spearman correlations compare each clarity measure with final bar accuracy. Onset fractions count each note event equally; active-root fractions weight non-silent 200-ms frames and ask whether any active pitch matches the root. "
                "They answer different questions, and extra polyphony or persistent transcribed notes can inflate active-root presence. All use timed reference harmony and are evaluation-only. Their correlations describe this inspected sample, not effects established by intervention.\n")
    clarity = ("bass_reference_root_onset_fraction", "bass_reference_base_tone_onset_fraction", "bass_reference_root_active_fraction",
               "horns_reference_root_onset_fraction", "horns_reference_base_tone_onset_fraction", "horns_reference_root_active_fraction")
    text.append(md_table(["Source", "Protocol", "Measure", "Songs", "Spearman ρ with bar accuracy"],
        [[r["source"], r["protocol"], r["feature"], r["songs"], fmt(r["spearman_rho"])] for r in associations
         if r["method"] == focus and r["scheme"] == "triad" and r["feature"] in clarity]))
    text.append("## Exploratory within-protocol rank associations\n")
    text.append("Spearman correlations use average ranks for ties. They are descriptive associations across songs, not causal effects or significance tests. "
                "The strongest absolute correlations below are selected from many features after inspecting this same sample; correlated features and outliers can dominate. "
                "Constant or insufficient-data features have no correlation value. In particular, reference/catalog mismatch and final clock mismatch are mathematically related to timing error.\n")
    for source, protocol in available_focus:
        rows = [r for r in associations if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad" and r["spearman_rho"] is not None]
        rows = sorted(rows, key=lambda r: -abs(r["spearman_rho"]))[:10]
        text.append(f"### {source} · {protocol} · {focus}\n")
        text.append(md_table(["Feature", "Group", "Songs", "Spearman ρ"], [[r["feature"], r["feature_group"], r["songs"], fmt(r["spearman_rho"])] for r in rows]))
    text.append("## Structural sections and solo roles\n")
    text.append("Section metrics come from saved trajectories whose whole-song metrics were checked against their CSV rows. "
                "Repeated sections of the same role are pooled within each song before songs are averaged. "
                "Section errors inherit previous tracking drift; a late solo's accuracy is not a local causal test of that soloist's evidence. "
                "Piano and drums can be labeled as soloists even though their notes are absent from harmonic input. "
                "These are instrument-role annotations, not tests of individual performer effects. "
                "Clock/uniform rows without saved traces are omitted here, not treated as missing whole-song outcomes.\n")
    for source, protocol in available_focus:
        rows = [r for r in sections if r["source"] == source and r["protocol"] == protocol and r["method"] == focus and r["scheme"] == "triad"]
        text.append(f"### {source} · {protocol} · {focus}\n")
        text.append(md_table(["Section", "Solo role", "Songs", "Bar accuracy", "Within 2 beats", "Mean song progress", "Active pooled", "Active bass"],
            [[r["section_kind"], r["solo_role"], r["songs"], pct(r["bar_accuracy"]), pct(r["within2beat_accuracy"]),
              pct(r["mean_fraction_through_song"]), pct(r["pooled_active_coverage"]), pct(r["bass_active_coverage"])] for r in rows]))
    text += ["## Confounders and limits\n",
        "- Async and sync use different tunes, protocols, available instruments, and performance conditions. Their difference is not an isolated synchrony effect.\n"
        "- Only 49 async and 26 sync preferred performances are available. All feature/outcome relationships here are exploratory. Genre and meter subgroups can be very small.\n"
        "- Tempo summaries use annotation bar durations and catalog meter, not independent beat onsets. Pickups, endings, and bar-level interpolation affect them.\n"
        "- The unfolded chart already includes the performed bar order. Repetition statistics describe ambiguity within that supplied plan, not unknown-form tracking.\n"
        "- Transcription-derived bass/horn notes contain artifacts. Quality flags are diagnostic, not verified labels of erroneous notes. Flagged/unflagged tunes also differ musically.\n"
        "- Chord-accuracy features are evaluation outcomes. Their association with position accuracy is descriptive and not proof that a classifier change causes tracking improvement.\n"
        "- Harmonic-clarity features compare exported pitches with timed reference chords. They depend on transcription quality, annotation alignment and canonical chord reduction; non-chord tones and anticipations are not automatically errors. These quantities are evaluation-only and never online inputs. Onset fractions weight note events; active-root fractions condition on non-silent frames, with denominator counts retained.\n"
        "- Independent MIDI/audio/annotation time-origin verification remains outstanding. Timing or transcription mismatch can affect both clarity features and evaluation outcomes.\n"
        "- Baseline profiles exclude the evaluated composition, but choices among windows, methods and improvements on these same songs do not constitute a held-out generalization test.\n"
        "- Whole-song densities and durations are unavailable at the beginning of a live performance. They are not an online gating feature specification.\n",
        "## Artifacts and reproduction\n",
        "`song_features.csv` has one row per preferred song, with baseline/additional outcomes attached. "
        "`feature_definitions.json` gives exact feature meanings and separates reference-derived quantities. "
        "`method_distributions.csv`, `high_low_tunes.csv`, `rank_associations.csv`, `high_low_feature_medians.csv`, "
        "`method_comparisons.csv`, `section_metrics.csv`, and `section_summary.csv` retain the inspectable numbers. "
        "`paired_method_changes.csv`, `paired_change_summary.csv`, `paired_change_associations.csv`, and `paired_change_extremes.csv` compare matched methods and preserve both source runs; `feature_ranges.csv` records physical feature ranges. "
        "`analysis_provenance.json` records paths, hashes, settings, and QA warnings.\n",
        "Run `python analyze_tunes.py`; add `--metrics /absolute/path/to/run/metrics.csv` for each additional run. "
        "Use `--focus-method METHOD` to choose which fixed method's high/low tunes and associations appear in this narrative. "
        "Use repeated `--reference-method METHOD` to add paired comparisons; the default is g00_reproduce. Add `--cross-run-references` to use an unambiguous reference from another supplied run when absent from the current run. "
        "Additional run traces are discovered at `run_dir/traces/{protocol}/{song}/preferred/{method}.npz`.\n"]
    if warnings:
        text.append("## QA warnings\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n")
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", default=[], help="Additional per-song position metrics CSV; may be repeated")
    parser.add_argument("--focus-method", default="empirical_profile")
    parser.add_argument("--reference-method", action="append", default=None, help="Same-run comparison method; repeatable; default g00_reproduce")
    parser.add_argument("--cross-run-references", action="store_true", help="Allow a unique reference from another input source when not present in the current source")
    parser.add_argument("--output", type=Path, default=HERE / "tune_analysis")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("Analysis outputs must stay inside improvement_study")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = BENCH / "data" / "manifest.json"
    baseline_path = BENCH / "results" / "position_metrics.csv"
    baseline_summary = json.loads((BENCH / "results" / "summary.json").read_text())
    if baseline_summary.get("position_complete") is not True:
        raise ValueError("Baseline position results are not marked complete")
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["performances"] if entry["take"] == "preferred"]
    catalogs = {(song["protocol"], song["song"]): song["catalog"] for song in manifest["songs"]}
    sources = [{"source": "baseline", "path": str(baseline_path), "sha256": digest(baseline_path)}]
    metrics = position_rows(baseline_path, "baseline")
    warnings = []
    for i, path in enumerate(args.metrics, 1):
        path = path.resolve()
        source = f"run{i}_{path.parent.name}"
        metadata = {p.name: {"path": str(p), "sha256": digest(p)} for p in (path.parent / "run.json", path.parent / "summary.json") if p.exists()}
        sources.append({"source": source, "path": str(path), "sha256": digest(path), "run_metadata": metadata})
        metrics += position_rows(path, source)
        run_summary = path.parent / "summary.json"
        if run_summary.exists():
            state = json.loads(run_summary.read_text())
            if state.get("complete") is False:
                warnings.append(f"{source}: summary marks this additional run incomplete; all shown results are only its currently saved songs.")
    flagged = defaultdict(list)
    flag_path = BENCH / "audit" / "quality_flags.csv"
    for row in read_csv(flag_path):
        parts = Path(row["path"]).parts
        flagged[(row["protocol"], parts[2], row["take"])].append(row)
    chord_rows = read_csv(BENCH / "results" / "chord_metrics.csv")
    chord_lookup = {(r["protocol"], r["song"], r["scheme"], r["method"]): r for r in chord_rows
                    if r["take"] == "preferred" and r["input"] == "pooled" and r["preprocess"] == "raw"
                    and float(r["window_seconds"]) == (0. if r["method"] == "yamaha_active" else 1.)}
    features, section_values, source_files = [], [], []
    for index, entry in enumerate(entries):
        data = common.load_performance(entry)
        row, available = feature_row(entry, data, catalogs[(entry["protocol"], entry["song"])], flagged)
        for scheme in ("triad", "seventh"):
            for method in HARMONIC_METHODS:
                result = chord_lookup.get((entry["protocol"], entry["song"], scheme, method))
                if result:
                    name = register(f"baseline_{scheme}_{method}_chord_accuracy", "evaluation outcome: not inference input",
                                    "Saved baseline pooled raw chord-classification accuracy at fixed one-second context (instantaneous for Yamaha).")
                    row[name] = float(result["chord_accuracy"])
        features.append(row)
        section_values += section_rows(entry, data, available, metrics, warnings)
        source_files.append({"performance_id": entry["performance_id"], "annotation_sha256": digest(Path(entry["annotation_path"])),
                             "midi_sha256": {r["path"]: r["sha256"] for r in data["midi_audit"]}})
        if index % 15 == 0 or index == len(entries) - 1:
            print(f"Feature/trace audit {index + 1}/{len(entries)}", flush=True)
    comparison, distributions, high_low, associations, group_features = analyze_metrics(metrics, features)
    references = args.reference_method or ["g00_reproduce"]
    paired = paired_changes(metrics, features, references, cross_run=args.cross_run_references)
    sections = section_summary(section_values)
    ranges = []
    for protocol in ("async", "sync"):
        for name, definition in FEATURE_DEFINITIONS.items():
            values = [finite(row.get(name)) for row in features if row["protocol"] == protocol]
            values = [value for value in values if value is not None]
            if values:
                ranges.append({"protocol": protocol, "feature": name, "feature_group": definition["group"], "songs": len(values),
                               "minimum": min(values), "median": median(values), "maximum": max(values)})
    for name, rows in [("song_features.csv", features), ("method_comparisons.csv", comparison),
        ("method_distributions.csv", distributions), ("high_low_tunes.csv", high_low),
        ("rank_associations.csv", associations), ("high_low_feature_medians.csv", group_features),
        ("section_metrics.csv", section_values), ("section_summary.csv", sections), ("feature_ranges.csv", ranges),
        ("paired_method_changes.csv", paired[0]), ("paired_change_summary.csv", paired[1]),
        ("paired_change_associations.csv", paired[2]), ("paired_change_extremes.csv", paired[3])]:
        write_csv(output / name, rows)
    save_json(output / "feature_definitions.json", FEATURE_DEFINITIONS)
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(), "preferred_songs": len(features),
        "protocol_counts": dict(Counter(row["protocol"] for row in features)), "metric_sources": sources,
        "manifest_sha256": digest(manifest_path), "quality_audit_sha256": digest(flag_path),
        "analysis_script_sha256": digest(Path(__file__)), "common_loader_sha256": digest(BENCH / "common.py"),
        "source_files": source_files, "focus_method": args.focus_method,
        "paired_reference_methods": references,
        "cross_run_references_allowed": args.cross_run_references,
        "nominal_tempo_preferred_only": True, "high_threshold": .8, "low_threshold": .25,
        "rank_associations": "Spearman, average tied ranks, within protocol; minimum8 valid songs; no p-values or causal claims",
        "sections": "Saved traces verified against per-song metrics; pool frames per song/role first, then equal song means",
        "reference_features": "Explanatory/evaluation-only. Not supplied to any tracker or gate.",
        "warnings": sorted(set(warnings)), "position_metric_rows": len(metrics), "section_metric_rows": len(section_values)}
    save_json(output / "analysis_provenance.json", provenance)
    (output / "analysis.md").write_text(report(distributions, high_low, associations, sections, args.focus_method, warnings, sources, features, paired, group_features))
    print(json.dumps({"output": str(output), "songs": len(features), "metric_rows": len(metrics),
                      "section_rows": len(section_values), "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
