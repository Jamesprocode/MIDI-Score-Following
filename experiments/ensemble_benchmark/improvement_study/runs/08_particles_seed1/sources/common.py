"""Shared chord vocabulary and data loading for the ensemble benchmark.

Vocabulary reductions are declared benchmark choices, not published Yamaha or
Pardo chord templates. Triad labels retain the annotation's base triad/suspension
and ignore sevenths, extensions, slash bass and alteration flags. Thus a parsed
half-diminished chord (triad=dim) remains dim, but a major chord with an altered
fifth remains maj if the source labels its base triad maj. The seventh scheme
retains the parsed seventh as well. Neither scheme infers harmony from note data.

Position-plan coordinates use chronological bar index plus bar fraction. Actual
bar timestamps produce evaluation targets only. Do not pass truth_position_beats,
truth_bar_index or the timed annotation rows to an inference method.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
PC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
BASE_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
    "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "power": (0, 7), "root": (0,),
}
SEVENTH_INTERVALS = {"dom": 10, "min": 10, "maj": 11, "dim": 9}
_SEVENTH_NAMES = {
    ("maj", "dom"): "dom7", ("maj", "min"): "dom7",
    ("maj", "maj"): "maj7", ("min", "min"): "min7",
    ("min", "dom"): "min7", ("min", "maj"): "minmaj7",
    ("dim", "min"): "hdim7", ("dim", "dom"): "hdim7",
    ("dim", "dim"): "dim7", ("dim", "maj"): "dimmaj7",
    ("aug", "dom"): "aug7", ("aug", "min"): "aug7",
    ("aug", "maj"): "augmaj7",
}
QUALITY_INTERVALS = dict(BASE_INTERVALS)
for _base, _intervals in BASE_INTERVALS.items():
    for _seventh, _interval in SEVENTH_INTERVALS.items():
        _name = _SEVENTH_NAMES.get((_base, _seventh), f"{_base}_{_seventh}7")
        if _base in ("sus2", "sus4", "power", "root") and _seventh in ("dom", "min"):
            _name = f"{_base}_7" if _base.startswith("sus") else f"{_base}7"
        QUALITY_INTERVALS[_name] = tuple(sorted(set(_intervals) | {_interval}))

CANONICALIZATION_METADATA = {
    "triad": "Source base triad, explicit sus2/sus4, power or root; ignore sevenths/extensions/alteration flags/slash bass",
    "seventh": "Triad scheme plus parsed seventh; ignore higher extensions/alteration flags/slash bass",
    "root": "Absolute pitch class from spelled root; never key-relative root_interval",
    "no_chord": "None, represented by N.C.; not a pitched chord class",
    "triad_none_audit": "All 45 triad=none rows in the 75-song source inventory are N.C. with root NA. Root-only fallback exists for future valid-root none rows, not observed here.",
    "unknown_status": "Raise ValueError instead of guessing",
}


def truth(value):
    if value is True or value == "True":
        return True
    if value is False or value in ("False", None, "", "NA"):
        return False
    raise ValueError(f"Unsupported annotation boolean {value!r}")


def pitch_class(name):
    """Spelled root to absolute chroma, including Fb, Cb and Bbb."""
    value = str(name).strip().replace("♭", "b").replace("♯", "#")
    match = re.fullmatch(r"([A-Ga-g])([#b]*)", value)
    if match is None:
        raise ValueError(f"Unsupported pitch spelling {name!r}")
    natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return (natural[match[1].upper()] + match[2].count("#") - match[2].count("b")) % 12


def quality_id(chord, scheme="triad"):
    """Return (root_pc, canonical_quality), or None for explicit no-chord."""
    if scheme not in ("triad", "seventh"):
        raise ValueError("scheme must be 'triad' or 'seventh'")
    if truth(chord.get("is_no_chord", False)):
        return None
    root = pitch_class(chord["root"])
    base = str(chord.get("triad", "none"))
    sus2, sus4 = truth(chord.get("is_sus2", False)), truth(chord.get("is_sus4", False))
    if sus2 and sus4:
        raise ValueError("Chord has both sus2 and sus4 flags")
    if sus2:
        base = "sus2"
    elif sus4:
        base = "sus4"
    elif base == "sus":
        raise ValueError("Suspended chord lacks an explicit sus2/sus4 flag")
    elif base == "none":
        base = "root"
    if base not in BASE_INTERVALS:
        raise ValueError(f"Unsupported triad {base!r}")
    seventh = str(chord.get("seventh", "none"))
    if scheme == "triad" or seventh == "none":
        return root, base
    if seventh not in SEVENTH_INTERVALS:
        raise ValueError(f"Unsupported seventh {seventh!r}")
    quality = _SEVENTH_NAMES.get((base, seventh), f"{base}_{seventh}7")
    if base in ("sus2", "sus4", "power", "root") and seventh in ("dom", "min"):
        quality = f"{base}_7" if base.startswith("sus") else f"{base}7"
    return root, quality


def chord_template(root_pc, quality):
    """Binary 12-PC template for the declared canonical chord vocabulary."""
    if int(root_pc) != root_pc or not 0 <= int(root_pc) < 12:
        raise ValueError("root_pc must be an integer in 0..11")
    if quality not in QUALITY_INTERVALS:
        raise ValueError(f"Unsupported canonical quality {quality!r}")
    vector = np.zeros(12, dtype=float)
    vector[(int(root_pc) + np.asarray(QUALITY_INTERVALS[quality], dtype=int)) % 12] = 1.
    return vector


def label_string(chord_id, quality=None):
    """Stable label from an ID tuple, None, or explicit root_pc and quality."""
    if quality is not None:
        chord_id = (chord_id, quality)
    if chord_id is None:
        return "N.C."
    root_pc, name = chord_id
    if int(root_pc) != root_pc or not 0 <= int(root_pc) < 12 or name not in QUALITY_INTERVALS:
        raise ValueError(f"Invalid chord ID {chord_id!r}")
    return f"{PC_NAMES[int(root_pc)]}:{name}"


@lru_cache(maxsize=1)
def _pilot_reader():
    path = HERE.parent / "harmony_pilot" / "run_pilot.py"
    spec = importlib.util.spec_from_file_location("_ensemble_pilot_midi_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_midi


def load_performance(entry, step_seconds=.1):
    """Load notes and an untimed score plan; return evaluation targets separately.

    The entry is from data/manifest.json['performances']. ``initial_bpm`` is
    supplied catalog BPM, never the AMT MIDI export's tempo or a whole-recording
    fitted tempo. The reused MIDI reader uses its tempo map only to turn ticks
    into event seconds. Drums are excluded from pitched harmony evidence.

    Bar 0 / measure 0 leading padding is removed. q=0 is the first musical bar;
    q is measured in quarter-note beats. Score intervals are computed directly
    from source bar/fraction coordinates. ``times`` follows an absolute 100-ms
    grid by default, restricted to musical bar coverage. ``start_time`` is the
    known first musical boundary supplied for synchronized initialization.

    ``truth_position_beats`` uses interpolation between actual bar timestamps
    and is evaluation-only; it is not an independently annotated beat series.
    """
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    annotation = json.loads(Path(entry["annotation_path"]).read_text())
    all_bars = sorted(annotation["bars"], key=lambda r: (float(r["start_time"]), int(r["bar"])))
    positive_bars = [dict(r) for r in all_bars if float(r["end_time"]) > float(r["start_time"])]
    musical_start_index = next((i for i, r in enumerate(positive_bars) if int(r["measure"]) != 0), None)
    if musical_start_index is None:
        raise ValueError("No musical bars with nonzero chart measure numbers")
    bars = positive_bars[musical_start_index:]
    first_bar = int(bars[0]["bar"])
    if any(int(r["bar"]) != first_bar + i for i, r in enumerate(bars)):
        raise ValueError("Musical bar indices must be consecutive for this benchmark")
    meter = tuple(int(x) for x in str(entry["time_signature"]).split("/"))
    if len(meter) != 2 or min(meter) <= 0:
        raise ValueError("Expected a positive meter numerator/denominator")
    beats_per_bar = meter[0] * 4. / meter[1]
    bar_starts_beats = np.arange(len(bars), dtype=float) * beats_per_bar
    start_time, end_time = float(bars[0]["start_time"]), float(bars[-1]["end_time"])
    first_grid_index = int(np.ceil(start_time / step_seconds - 1e-9))
    times = np.arange(first_grid_index * step_seconds, end_time, step_seconds)
    actual_starts = np.array([float(r["start_time"]) for r in bars])
    actual_ends = np.array([float(r["end_time"]) for r in bars])
    truth_bar_index = np.searchsorted(actual_starts, times, side="right") - 1
    valid = (truth_bar_index >= 0) & (times < actual_ends[np.maximum(truth_bar_index, 0)])
    times, truth_bar_index = times[valid], truth_bar_index[valid]
    within_bar = ((times - actual_starts[truth_bar_index]) /
                  (actual_ends[truth_bar_index] - actual_starts[truth_bar_index]))
    truth_position_beats = (truth_bar_index + within_bar) * beats_per_bar

    chords = sorted((dict(r) for r in annotation["chords"]
                     if float(r["end_time"]) > float(r["start_time"])),
                    key=lambda r: (float(r["start_time"]), float(r["end_time"])))
    chord_starts_beats = np.array([(float(r["start_bar"]) - first_bar + float(r["start_fraction"])) * beats_per_bar
                                  for r in chords])
    chord_ends_beats = np.array([(float(r["end_bar"]) - first_bar + float(r["end_fraction"])) * beats_per_bar
                                for r in chords])
    if np.any(chord_starts_beats < 0) or np.any(chord_ends_beats <= chord_starts_beats):
        raise ValueError("Chord plan must consist of positive musical intervals")
    if np.any(chord_starts_beats[1:] < chord_ends_beats[:-1] - 1e-8):
        raise ValueError("Chord plan overlaps after chronological sorting")
    initial_bpm = float(entry["bpm"])
    if not np.isfinite(initial_bpm) or initial_bpm <= 0:
        raise ValueError("Catalog initial BPM must be positive")
    read_midi = _pilot_reader()
    notes_by_instrument, midi_audit, excluded_drums = {}, [], []
    for item in entry["midi_files"]:
        instrument = item.get("instrument", Path(item["path"]).name.split(".")[0])
        if instrument == "drums" or instrument.startswith("drums_"):
            excluded_drums.append(item["path"])
            continue
        notes, info = read_midi(Path(item["path"]))
        midi_audit.append({"instrument": instrument, **info})
        if instrument in notes_by_instrument:
            notes = np.concatenate([notes_by_instrument[instrument], notes], axis=0)
            notes = notes[np.argsort(notes[:, 0], kind="stable")]
        notes_by_instrument[instrument] = notes
    return {
        "performance_id": entry["performance_id"], "notes_by_instrument": notes_by_instrument,
        "times": times, "chords": chords, "bars": bars,
        "sections": sorted(annotation.get("sections", []), key=lambda r: float(r["start_time"])),
        "start_time": start_time, "end_time": end_time, "initial_bpm": initial_bpm,
        "meter": meter, "beats_per_bar": beats_per_bar, "first_annotation_bar": first_bar,
        "initial_position_beats": 0., "chord_starts_beats": chord_starts_beats,
        "chord_ends_beats": chord_ends_beats, "bar_starts_beats": bar_starts_beats,
        "score_end_beats": len(bars) * beats_per_bar,
        "truth_position_beats": truth_position_beats, "truth_bar_index": truth_bar_index,
        "midi_audit": midi_audit, "excluded_drums": excluded_drums,
    }


def checks():
    """Analytical checks of enharmonic roots and coarse chord reduction."""
    assert pitch_class("Bbb") == 9 and pitch_class("Fb") == 4 and pitch_class("Cb") == 11
    def c(root, triad, seventh="none", **kw):
        return {"root": root, "triad": triad, "seventh": seventh, **kw}
    half_dim = c("D", "dim", "min", alt_b5="True")
    assert quality_id(half_dim) == (2, "dim")
    assert quality_id(half_dim, "seventh") == (2, "hdim7")
    assert set(np.flatnonzero(chord_template(2, "hdim7"))) == {0, 2, 5, 8}
    assert quality_id(c("G", "min", "maj"), "seventh") == (7, "minmaj7")
    assert quality_id(c("C", "maj", "dom", alt_sharp5="True")) == (0, "maj")
    assert quality_id(c("C", "sus", "dom", is_sus4="True"), "seventh") == (0, "sus4_7")
    assert quality_id(c("C", "none")) == (0, "root")
    assert quality_id(c("NA", "none", is_no_chord="True")) is None
    assert label_string(None) == "N.C." and label_string((1, "maj")) == "Db:maj"
    assert all(np.isfinite(chord_template(0, name)).all() for name in QUALITY_INTERVALS)
    return {"status": "passed", "quality_templates": len(QUALITY_INTERVALS)}


if __name__ == "__main__":
    print(json.dumps(checks(), indent=2))
