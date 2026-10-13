"""Build three MIDI playback and causal harmonic-following browser payloads.

Run using a Python with NumPy. Original MIDI, annotations, manifests and saved
experiments are read-only. Only the data/audit JSON files declared in PERFORMANCES
beside this script are written. The follower is the
existing p02 exploratory baseline, not a full
reproduction of Maezawa's system. It gets a known start, catalog tempo, an untimed
chord plan, and current bass notes, horn notes, or their combined binary union.
All evidence choices share every filter/template setting. Annotation timestamps are
evaluation only.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BENCHMARK = HERE.parent / "ensemble_benchmark"
STUDY = BENCHMARK / "improvement_study"
sys.path[:0] = [str(STUDY), str(BENCHMARK)]
import run_study
import particle_follower

PERFORMANCES = (
    {"id": "async/07-blue_bossa/preferred", "key": "blue_bossa", "title": "Blue Bossa",
     "data": "data.json", "audit": "data_audit.json"},
    {"id": "sync/67-on_green_dolphin_street/preferred", "key": "green_dolphin",
     "title": "On Green Dolphin Street", "data": "green_dolphin.json",
     "audit": "green_dolphin_audit.json"},
    {"id": "sync/70-samba_de_orfeu/preferred", "key": "samba_de_orfeu",
     "title": "Samba de Orfeu", "data": "samba_de_orfeu.json",
     "audit": "samba_de_orfeu_audit.json"},
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def playback_notes(path, return_details=False):
    """Preserve MIDI velocities as well as source note timing and all tails.

    The benchmark reader intentionally discards velocity. This small extension
    follows the same event/tempo handling and is checked against that reader.
    """
    raw_file = path.read_bytes()
    assert raw_file[:4] == b"MThd"
    header_len = struct.unpack(">I", raw_file[4:8])[0]
    fmt, n_tracks, ppq = struct.unpack(">HHH", raw_file[8:14])
    assert fmt in (0, 1) and ppq < 32768
    pos = 8 + header_len
    tick_notes, tempos = [], []
    channels = defaultdict(int)
    programs = defaultdict(set)
    vlq = run_study.benchmark.old.vlq
    for _ in range(n_tracks):
        assert raw_file[pos:pos + 4] == b"MTrk"
        size = struct.unpack(">I", raw_file[pos + 4:pos + 8])[0]
        raw = raw_file[pos + 8:pos + 8 + size]
        pos += 8 + size
        cursor, tick, status = 0, 0, None
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
                    active[(channel, payload[0])].append((tick, payload[1]))
                    channels[channel] += 1
                elif kind == 12:
                    programs[channel].add(payload[0])
                elif kind == 8 or (kind == 9 and payload[1] == 0):
                    key = (channel, payload[0])
                    assert active[key], f"Unmatched note-off: {path}"
                    onset, velocity = active[key].popleft()
                    tick_notes.append((onset, tick, payload[0], velocity))
        assert not any(active.values()), f"Unclosed note: {path}"
    assert pos == len(raw_file)
    tempos.sort()

    def seconds(tick):
        previous, micros, elapsed = 0, 500000, 0.
        for change, new_micros in tempos:
            if change > tick:
                break
            elapsed += (change - previous) * micros / ppq / 1e6
            previous, micros = change, new_micros
        return elapsed + (tick - previous) * micros / ppq / 1e6

    notes = [[seconds(a), seconds(b), int(p), int(v)] for a, b, p, v in tick_notes]
    notes.sort(key=lambda n: (n[0], n[1], n[2], n[3]))
    original, _ = run_study.benchmark.old.read_midi(path)
    # Equal-onset order is unspecified in the original; compare canonical rows.
    assert sorted(map(tuple, original.tolist())) == sorted(tuple(n[:3]) for n in notes)
    details = {"midiChannels": sorted(channels),
               "noteCountByMidiChannel": {str(k): v for k, v in sorted(channels.items())},
               "programsByMidiChannel": {str(k): sorted(v) for k, v in sorted(programs.items())},
               "midiPitchCounts": {str(p): sum(n[2] == p for n in notes)
                                   for p in sorted({n[2] for n in notes})},
               "channelNumbering": "Zero-based; channel 9 is General MIDI percussion."}
    return (notes, details) if return_details else notes


def numeric_row(row):
    """Keep full official rows while making timeline coordinates numeric."""
    output = {}
    for key, value in row.items():
        if key in {"bar", "measure", "start_bar", "end_bar", "section_index",
                   "solo_index", "musician_id", "position", "play_index"}:
            output[key] = int(value)
        elif key in {"start_time", "end_time", "start_fraction", "end_fraction"}:
            output[key] = float(value)
        elif value in ("True", "False"):
            output[key] = value == "True"
        else:
            output[key] = value
    if "start_time" in output:
        output["startTime"] = output["start_time"]
        output["endTime"] = output["end_time"]
    return output


def selected_features(streams, times, instruments, targets):
    """Union of active keys from explicitly named instruments, never a relabel.

    Each roll is binary. Max pooling means a unison shared by two horn tracks
    counts once, while two octaves remain separate piano keys. The same template
    assigns weight 2 to the root and 1 to the other triad tones for either input.
    This matches active_bass_root exactly when instruments is ('bass',).
    """
    assert instruments and all(name in streams for name in instruments)
    active = np.maximum.reduce([
        run_study.benchmark.yamaha.piano_roll88(streams[name], times)
        for name in instruments
    ])
    templates = targets["templates"].copy()
    templates[np.arange(len(templates)), targets["roots"]] += 1.
    templates88 = run_study.benchmark.yamaha.expand_chroma_templates(templates)
    return active @ templates88.T, active.sum(axis=1) > 0, 2. * active.sum(axis=1)


def subset_metrics(prediction, data, selected, mask):
    """The original metrics restricted to an explicitly reported frame mask."""
    subset = selected[mask]
    if not len(subset):
        return {"frames": 0, "bar_accuracy": None, "within2beat_accuracy": None,
                "mean_abs_beats": None, "median_abs_beats": None}
    output = {key: prediction[key][mask] for key in ("position_beats", "bar_index")}
    return run_study.benchmark.position_metrics(output, data, subset)


def build(spec):
    # Preserve the pre-existing bass sample objects byte-for-byte after JSON
    # canonicalization, not only their displayed positions.
    old_path = HERE / spec["data"]
    old_payload = json.loads(old_path.read_text()) if old_path.exists() else None
    legacy_samples = old_payload["samples"] if old_payload is not None else None
    manifest_path = BENCHMARK / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = copy.deepcopy(next(e for e in manifest["performances"]
                               if e["performance_id"] == spec["id"]))
    protocol, song, take = spec["id"].split("/")
    annotation_path = BENCHMARK / "data/annotations" / protocol / f"{song}.json"
    midi_dir = PROJECT / "Dataset" / "SAMBA dataset" / f"{protocol}_midi" / "songs" / song / take
    entry["annotation_path"] = str(annotation_path)
    for item in entry["midi_files"]:
        item["path"] = str(midi_dir / item["name"])
        assert digest(Path(item["path"])) == item["sha256"], item["name"]
    annotation = json.loads(annotation_path.read_text())
    data = run_study.benchmark.common.load_performance(entry)
    setting = next(s for s in run_study.variants("continuous_tempo")
                   if s["method"] == "p02_particles_slow")
    selected = np.arange(0, len(data["times"]), 2)
    times = data["times"][selected]
    assert np.allclose(np.diff(times), .2, rtol=0, atol=1e-10)
    case = run_study.get_features(data, {"triad"})["triad"]
    scores, available, mass = [a[selected] for a in case["features"]["active_bass_root"]]
    probabilities, reliability = run_study.observation(scores, available, mass, setting)
    args = (times, probabilities, available, data["chord_starts_beats"],
            data["chord_ends_beats"], case["targets"]["ids"],
            data["bar_starts_beats"], data["initial_bpm"])
    prediction = particle_follower.follow_positions(
        *args, start_time=data["start_time"], config=setting["config"])
    saved_path = STUDY / "runs/05_continuous_tempo/traces" / protocol / song / take / "p02_particles_slow.npz"
    comparisons = {}
    if saved_path.exists():
        with np.load(saved_path) as saved:
            for key, actual in (("time", times), ("estimated_beats", prediction["position_beats"]),
                                ("estimated_bpm", prediction["tempo_bpm"]),
                                ("reference_beats", data["truth_position_beats"][selected])):
                comparisons[key] = bool(np.array_equal(saved[key], actual))
                assert comparisons[key], f"Saved p02 mismatch: {key}"

    cut = min(600, len(times) // 2)
    prefix = particle_follower.follow_positions(
        times[:cut], probabilities[:cut], available[:cut], *args[3:],
        start_time=data["start_time"], config=setting["config"])
    for key in ("position_beats", "tempo_bpm", "bar_index", "effective_sample_size"):
        assert np.array_equal(prefix[key], prediction[key][:cut]), key
    # Also test feature causality after removing all unobserved note onsets and
    # replacing future note-offs by an 'open through the cut' marker.
    cutoff = times[cut - 1]
    prefix_data = dict(data)
    prefix_data["times"] = data["times"][:selected[cut - 1] + 1]
    prefix_streams = {}
    for instrument, notes in data["notes_by_instrument"].items():
        kept = notes[notes[:, 0] <= cutoff].copy()
        kept[:, 1] = np.minimum(kept[:, 1], cutoff + 1e-6)
        prefix_streams[instrument] = kept
    prefix_data["notes_by_instrument"] = prefix_streams
    truncated = run_study.get_features(prefix_data, {"triad"})["triad"]
    for original, short in zip(case["features"]["active_bass_root"],
                               truncated["features"]["active_bass_root"]):
        assert np.array_equal(original[:len(short)], short)

    horn_instruments = sorted(name for name in data["notes_by_instrument"]
                              if name.startswith("saxophone") or name in {"trumpet", "trombone", "clarinet", "flute"})
    assert horn_instruments, "No explicitly identified horn streams"
    variants, variant_audits = {}, {}
    input_choices = {"bass": ["bass"], "horn": horn_instruments,
                     "both": ["bass", *horn_instruments]}
    for variant_id, instruments in input_choices.items():
        full_features = selected_features(data["notes_by_instrument"], data["times"], instruments, case["targets"])
        if variant_id == "both":
            # Verify pooling at the note level against both an explicit union
            # of component rolls and a roll of the concatenated source notes.
            bass_roll = run_study.benchmark.yamaha.piano_roll88(data["notes_by_instrument"]["bass"], data["times"])
            horn_roll = np.maximum.reduce([
                run_study.benchmark.yamaha.piano_roll88(data["notes_by_instrument"][name], data["times"])
                for name in horn_instruments
            ])
            union_roll = np.maximum(bass_roll, horn_roll)
            combined_notes = np.concatenate([data["notes_by_instrument"][name] for name in instruments])
            assert np.array_equal(union_roll, run_study.benchmark.yamaha.piano_roll88(combined_notes, data["times"]))
            root_templates = case["targets"]["templates"].copy()
            root_templates[np.arange(len(root_templates)), case["targets"]["roots"]] += 1.
            expected_scores = union_roll @ run_study.benchmark.yamaha.expand_chroma_templates(root_templates).T
            assert np.array_equal(full_features[0], expected_scores)
            assert np.array_equal(full_features[1], union_roll.sum(axis=1) > 0)
            assert np.array_equal(full_features[2], 2. * union_roll.sum(axis=1))
        variant_scores, variant_available, variant_mass = [a[selected] for a in full_features]
        variant_probabilities, _ = run_study.observation(variant_scores, variant_available, variant_mass, setting)
        variant_prediction = particle_follower.follow_positions(
            times, variant_probabilities, variant_available, *args[3:],
            start_time=data["start_time"], config=setting["config"])
        if variant_id == "bass":
            for expected, actual in zip((scores, available, mass, probabilities),
                                        (variant_scores, variant_available, variant_mass, variant_probabilities)):
                assert np.array_equal(expected, actual), "Bass features changed"
            for key in ("position_beats", "tempo_bpm", "bar_index", "position_confidence"):
                assert np.array_equal(prediction[key], variant_prediction[key]), key
        variant_prefix = particle_follower.follow_positions(
            times[:cut], variant_probabilities[:cut], variant_available[:cut], *args[3:],
            start_time=data["start_time"], config=setting["config"])
        for key in ("position_beats", "tempo_bpm", "bar_index", "effective_sample_size"):
            assert np.array_equal(variant_prefix[key], variant_prediction[key][:cut]), key
        truncated_features = selected_features(prefix_streams, prefix_data["times"], instruments, case["targets"])
        for original, short in zip(full_features, truncated_features):
            assert np.array_equal(original[:len(short)], short), "Future notes changed an earlier feature"
        # Removing or injecting arbitrarily dense notes in every excluded source
        # track must not alter either variant's input features.
        excluded = sorted({item["instrument"] for item in entry["midi_files"]} - set(instruments))
        only_selected = {name: data["notes_by_instrument"][name] for name in instruments}
        altered = dict(data["notes_by_instrument"])
        dense_notes = np.array([[0., data["end_time"] + 10., pitch] for pitch in range(21, 109)])
        for name in [*excluded, "unselected_piano_probe"]:
            altered[name] = dense_notes.copy()
        for alternative in (only_selected, altered):
            changed = selected_features(alternative, data["times"], instruments, case["targets"])
            for original, other in zip(full_features, changed):
                assert np.array_equal(original, other), "An excluded track affected evidence"
        variants[variant_id] = {"instruments": instruments, "scores": variant_scores,
                                "available": variant_available, "probabilities": variant_probabilities,
                                "prediction": variant_prediction}
        variant_audits[variant_id] = {
            "instruments": instruments, "excludedInstruments": excluded,
            "prefixInferenceIdentical": True, "featuresUnaffectedByFutureNotes": True,
            "featuresUnaffectedByExcludedTrackRemovalOrMutation": True,
            "prefixLastSampleSeconds": float(cutoff),
            "evidenceCoverage": float(variant_available.mean()),
            "availableFrames": int(variant_available.sum()), "totalFrames": len(times),
            "trackerConfig": variant_prediction["config"],
            "metrics": run_study.benchmark.position_metrics(variant_prediction, data, selected),
        }
        if variant_id == "both":
            variant_audits[variant_id]["featuresMatchBassHornBinaryUnionAndConcatenatedNotes"] = True
    assert all(v["prediction"]["config"] == variants["bass"]["prediction"]["config"]
               for v in variants.values()), "Tracking settings must be identical across input choices"

    tracks = []
    track_labels = {"bass": "Bass", "saxophone_alto": "Alto saxophone", "trumpet": "Trumpet",
                    "drums": "Drums", "piano": "Piano", "saxophone_tenor": "Tenor saxophone"}
    sources = {"annotation": {"path": str(annotation_path), "sha256": digest(annotation_path)},
               "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
               "midi": [], "implementation": []}
    for item in entry["midi_files"]:
        path = Path(item["path"])
        notes, midi_details = playback_notes(path, return_details=True)
        source = {"instrument": item["instrument"], "path": str(path), "sha256": digest(path)}
        sources["midi"].append(source)
        instrument = item["instrument"]
        is_percussion = instrument.startswith("drums") or midi_details["midiChannels"] == [9]
        tracks.append({"id": instrument, "label": track_labels.get(instrument, instrument.replace("_", " ").capitalize()),
                       "notes": notes, "noteCount": len(notes), "sourcePath": str(path),
                       "isPercussion": is_percussion, **midi_details})
    for path in (Path(__file__), STUDY / "run_study.py", STUDY / "particle_follower.py",
                 BENCHMARK / "common.py", BENCHMARK / "run_benchmark.py",
                 HERE.parent / "harmony_pilot/yamaha_matcher.py",
                 HERE.parent / "harmony_pilot/run_pilot.py"):
        sources["implementation"].append({"path": str(path), "sha256": digest(path)})
    midi_end = max(n[1] for track in tracks for n in track["notes"])
    bars = []
    first_bar = data["first_annotation_bar"]
    for row in annotation["bars"]:
        bar = numeric_row(row)
        bar["padding"] = bar["measure"] == 0
        bar["startBeat"] = (bar["bar"] - first_bar) * data["beats_per_bar"]
        bar["endBeat"] = bar["startBeat"] + data["beats_per_bar"]
        bar["referenceBpm"] = (data["beats_per_bar"] * 60. / (bar["end_time"] - bar["start_time"])) if not bar["padding"] and bar["end_time"] > bar["start_time"] else None
        bars.append(bar)
    chords = []
    for row in annotation["chords"]:
        chord = numeric_row(row)
        chord["label"] = row["chord"]
        chord["triadLabel"] = run_study.benchmark.common.label_string(
            run_study.benchmark.common.quality_id(row, "triad"))
        chord["startBeat"] = (chord["start_bar"] - first_bar + chord["start_fraction"]) * data["beats_per_bar"]
        chord["endBeat"] = (chord["end_bar"] - first_bar + chord["end_fraction"]) * data["beats_per_bar"]
        chord["pointAnnotation"] = chord["end_time"] == chord["start_time"]
        chords.append(chord)
    sections = []
    for row in annotation["sections"]:
        section = numeric_row(row)
        section["label"] = row["section"]
        section["startBeat"] = (section["start_bar"] - first_bar + section["start_fraction"]) * data["beats_per_bar"]
        section["endBeat"] = (section["end_bar"] - first_bar + section["end_fraction"]) * data["beats_per_bar"]
        sections.append(section)
    truth = data["truth_position_beats"][selected]
    steady = (times - data["start_time"]) * data["initial_bpm"] / 60.
    steady_matches = bool(np.allclose(truth, steady, rtol=0, atol=1e-9))
    musical_bars = [b for b in bars if not b["padding"] and b["end_time"] > b["start_time"]]
    bar_tempi = np.array([b["referenceBpm"] for b in musical_bars])
    reference_bpm = bar_tempi[data["truth_bar_index"][selected]]
    tempo_summary = {"minimumBpm": float(bar_tempi.min()), "maximumBpm": float(bar_tempi.max()),
                     "medianBpm": float(np.median(bar_tempi)),
                     "overallBpm": float(data["score_end_beats"] * 60. / (data["end_time"] - data["start_time"])),
                     "catalogBpm": data["initial_bpm"], "fixedCatalogClockMatches": steady_matches,
                     "description": "Tempo averaged within each annotated bar; not independently annotated beat-to-beat tempo."}
    if protocol == "async":
        assert steady_matches, "Expected the asynchronous reference to match its supplied fixed tempo"
    samples = []
    for index, t in enumerate(times):
        samples.append({"time": round(float(t), 10),
                        "truthPositionBeats": float(truth[index]),
                        "estimatedPositionBeats": float(prediction["position_beats"][index]),
                        "estimatedBpm": float(prediction["tempo_bpm"][index]),
                        "referenceBpm": float(reference_bpm[index]),
                        "truthBar": int(data["truth_bar_index"][selected[index]]) + first_bar,
                        "estimatedBar": int(prediction["bar_index"][index]) + first_bar,
                        "harmonyAvailable": bool(available[index]),
                        "positionConfidence": float(prediction["position_confidence"][index]),
                        "harmonicScores": scores[index].tolist(),
                        "harmonicProbabilities": probabilities[index].tolist(),
                        "activeBassNotes": [int(p) for a, b, p in data["notes_by_instrument"]["bass"] if a <= t < b],
                        "errorBeats": float(prediction["position_beats"][index] - truth[index])})
    metrics = run_study.benchmark.position_metrics(prediction, data, selected)
    if legacy_samples is not None:
        assert samples == legacy_samples, "Legacy bass samples must remain exactly unchanged"
    followers = {}
    for variant_id, variant in variants.items():
        variant_prediction = variant["prediction"]
        variant_available = variant["available"]
        variant_samples = []
        for index, t in enumerate(times):
            active_notes = sorted({int(p) for name in variant["instruments"]
                                   for a, b, p in data["notes_by_instrument"][name]
                                   if a <= t < b and 21 <= p <= 108})
            sample = dict(samples[index])
            if variant_id != "bass":
                sample.pop("activeBassNotes")
            sample.update({
                "estimatedPositionBeats": float(variant_prediction["position_beats"][index]),
                "estimatedBpm": float(variant_prediction["tempo_bpm"][index]),
                "estimatedBar": int(variant_prediction["bar_index"][index]) + first_bar,
                "harmonyAvailable": bool(variant_available[index]),
                "evidenceAvailable": bool(variant_available[index]),
                "positionConfidence": float(variant_prediction["position_confidence"][index]),
                "harmonicScores": variant["scores"][index].tolist(),
                "harmonicProbabilities": variant["probabilities"][index].tolist(),
                "activeEvidenceNotes": active_notes,
                "errorBeats": float(variant_prediction["position_beats"][index] - truth[index]),
            })
            variant_samples.append(sample)
        followers[variant_id] = {
            "label": {"bass": "Bass only", "horn": "Horns only", "both": "Bass + horns"}[variant_id],
            "instruments": variant["instruments"], "samples": variant_samples,
            "metrics": variant_audits[variant_id]["metrics"],
            "metricsDenominator": "All 200 ms evaluated frames, including frames without active selected notes.",
            "metricsByEvidence": {
                "available": subset_metrics(variant_prediction, data, selected, variant_available),
                "absent": subset_metrics(variant_prediction, data, selected, ~variant_available),
            },
            "evidenceCoverage": float(variant_available.mean()),
            "availableFrames": int(variant_available.sum()), "totalFrames": len(times),
            "trackerConfig": variant_prediction["config"],
            "description": f"Current {' + '.join(variant['instruments'])} notes only. The same root-weighted triad templates, score normalization, known start, catalog tempo, particle configuration and seed are used for all inputs. Notes are pooled before harmonic scoring, without averaging independently estimated positions or learning adaptive weights. No beat detector or melodic-order matching.",
            "pooling": "Binary union across the selected tracks at each MIDI key; repeated unisons count once and octaves remain separate.",
        }
    if old_payload is not None and "followers" in old_payload:
        for variant_id, previous in old_payload["followers"].items():
            assert followers[variant_id]["samples"] == previous["samples"], f"Existing {variant_id} trace changed"
            assert followers[variant_id]["metrics"] == previous["metrics"], f"Existing {variant_id} metrics changed"
    track_names = ", ".join(t["label"].lower() for t in tracks)
    missing = [name for name in ("piano", "drums") if not any(t["id"] == name for t in tracks)]
    missing_description = (f" {' and '.join(missing).capitalize()} MIDI is absent from the selected derived-MIDI tracks." if missing else "")
    truth_description = (f"Official bar annotations; this asynchronous take has a fixed {data['initial_bpm']:g} BPM grid."
                         if steady_matches else
                         "Official bar annotations from the synchronous performance preserve its changing bar durations.")
    truth_description += " Bar 0 is leading padding. Position within each bar is interpolated, not independently beat-annotated."
    sample_description = f"Causal estimates every 200 ms starting at {times[0]:g} seconds, displayed by holding the most recent update. Ground truth is undefined after {data['end_time']:g} seconds; original MIDI note tails are preserved."
    payload = {
        "metadata": {
            "title": spec["title"], "key": entry["key_signature"], "recordingKey": spec["key"], "performanceId": entry["performance_id"],
            "protocol": "Asynchronous" if protocol == "async" else "Synchronous", "take": "Preferred", "bpm": data["initial_bpm"],
            "meter": entry["time_signature"], "beatsPerBar": data["beats_per_bar"], "updateSeconds": .2,
            "musicalStartSeconds": data["start_time"], "annotationEndSeconds": data["end_time"],
            "midiEndSeconds": midi_end, "durationSeconds": max(midi_end, data["end_time"]),
            "scoreEndBeats": data["score_end_beats"], "musicalBarCount": len(data["bars"]),
            "firstAnnotationBar": first_bar, "firstChartMeasure": int(data["bars"][0]["measure"]),
            "noteFields": ["onsetSeconds", "offsetSeconds", "midiPitch", "velocity"],
            "harmonicLabels": case["targets"]["labels"],
            "groundTruthDescription": truth_description,
            "trackerDescription": f"Three independent p02 runs use bass-only, horn-only, or their combined active notes with identical root-weighted triad templates, known musical start, supplied {data['initial_bpm']:g} BPM and seed. Combined input is a binary union of MIDI keys before harmonic scoring. No averaging of independent position estimates, beat detector, melodic-order branch, learned weighting, or ground-truth timestamps after initialization.",
            "playbackDescription": f"Original transcribed MIDI note timings and velocities for {track_names}. Browser sound is synthesized." + missing_description,
            "sampleDescription": sample_description, "tempoSummary": tempo_summary,
            "trackerSetting": setting, "trackerConfig": prediction["config"],
            "sourcePaths": {"annotation": str(annotation_path), "midiDirectory": str(midi_dir)},
            "metrics": metrics,
        },
        "tracks": tracks, "bars": bars, "chords": chords, "sections": sections,
        "samples": samples, "followers": followers, "sources": sources,
        "additionalAnnotations": {k: [numeric_row(r) for r in v]
                                  for k, v in annotation.items() if k not in {"bars", "chords", "sections"}},
    }
    audit = {
        "status": "passed", "performanceId": entry["performance_id"],
        "sourceMidiHashesMatchManifest": True, "sourceFilesUnmodified": True,
        "playbackNotesMatchBenchmarkReader": True, "velocitiesPreserved": True,
        "samplingIntervalSeconds": .2, "sampleCount": len(times),
        "firstSampleSeconds": float(times[0]), "lastSampleSeconds": float(times[-1]),
        "annotationMusicalStartSeconds": data["start_time"], "annotationEndSeconds": data["end_time"],
        "midiEndSeconds": midi_end, "postAnnotationTailSeconds": midi_end - data["end_time"],
        "steady160BpmMatchesAllAnnotationBars": bool(np.allclose(bar_tempi, 160., rtol=0, atol=1e-9)),
        "fixedCatalogTempoMatchesAnnotations": steady_matches, "tempoSummary": tempo_summary,
        "followerPrefixIdentical": True, "featurePrefixIdenticalAfterRemovingFutureNotes": True,
        "prefixLastSampleSeconds": float(cutoff), "savedTracePath": str(saved_path),
        "savedTraceIdentical": comparisons, "noteCounts": {t["id"]: t["noteCount"] for t in tracks},
        "midiTrackDetails": {t["id"]: {k: t[k] for k in ("midiChannels", "noteCountByMidiChannel", "programsByMidiChannel", "midiPitchCounts", "isPercussion")} for t in tracks},
        "trackerConfig": prediction["config"], "metrics": metrics, "sources": sources,
        "followers": variant_audits,
        "legacyBassSamplesUnchanged": legacy_samples is None or samples == legacy_samples,
        "existingFollowerTracesUnchanged": old_payload is None or "followers" not in old_payload or all(followers[k]["samples"] == v["samples"] for k, v in old_payload["followers"].items()),
        "identicalFollowerSettingsAcrossInputs": all(v["prediction"]["config"] == variants["bass"]["prediction"]["config"] for v in variants.values()),
        "limitations": ["Transcribed MIDI is not verified against performed notes.",
                        "This is a precomputed causal replay, not measured live end-to-end latency.",
                        ("Known start and supplied tempo make the steady clock reference unusually strong for this asynchronous take." if steady_matches else "Reference tempo is derived from annotated bar durations, not independent beat annotations."),
                        missing_description.strip(),
                        "Official chart measure numbers repeat; chronological annotation bar numbers identify occurrences."]
    }
    for name, value in ((spec["data"], payload), (spec["audit"], audit)):
        (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for source in [sources["annotation"], sources["manifest"], *sources["midi"], *sources["implementation"]]:
        assert digest(Path(source["path"])) == source["sha256"]
    print(json.dumps({"status": "passed", "title": spec["title"], "samples": len(samples), "notes": audit["noteCounts"],
                      "savedTraceIdentical": comparisons, "tempoSummary": tempo_summary,
                      "midiChannels": {t["id"]: t["midiChannels"] for t in tracks}, "metrics": metrics,
                      "followers": {k: {"instruments": v["instruments"], "metrics": v["metrics"], "evidenceCoverage": v["evidenceCoverage"]} for k, v in followers.items()}}, indent=2))


def main():
    for spec in PERFORMANCES:
        build(spec)


if __name__ == "__main__":
    main()
