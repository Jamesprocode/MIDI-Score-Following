"""JazzSAMBA takes for score following with piano input.

``load_take`` returns the same dictionary as ensemble_benchmark/common.py
``load_performance`` (unfolded chord plan in quarter-note beats from the first
musical bar, 100 ms evaluation grid, truth positions interpolated within the
annotated bars), built from the local annotation CSVs instead of the benchmark's
downloaded JSON, so every take is available (the benchmark's manifest lists 124
of the 152 and its paths no longer resolve). ``tests/test_follower.py`` checks it
reproduces load_performance exactly where that JSON exists.

The performance input is the take's recorded piano MIDI (key presses; pedal
ignored), as in Maezawa (2023), where the user plays piano. The benchmark used
bass and horn transcriptions because piano MIDI was not available then. Sync
piano MIDI gets the timing repair of yamaha_timing/yamaha_timing/real_data.py.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
for sub in ("yamaha_timing", "ensemble_benchmark/improvement_study", "ensemble_benchmark", "harmony_pilot"):
    if str(EXPERIMENTS / sub) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS / sub))

from yamaha_timing.real_data import samba_takes  # noqa: E402


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def plan_from_annotations(bars_rows, chord_rows, time_signature, bpm, step_seconds=.1):
    """The body of common.load_performance, from annotation rows."""
    all_bars = sorted(bars_rows, key=lambda r: (float(r["start_time"]), int(r["bar"])))
    positive = [dict(r) for r in all_bars if float(r["end_time"]) > float(r["start_time"])]
    first_musical = next((i for i, r in enumerate(positive) if int(r["measure"]) != 0), None)
    if first_musical is None:
        raise ValueError("No musical bars with nonzero chart measure numbers")
    bars = positive[first_musical:]
    first_bar = int(bars[0]["bar"])
    if any(int(r["bar"]) != first_bar + i for i, r in enumerate(bars)):
        raise ValueError("Musical bar indices must be consecutive")
    meter = tuple(int(x) for x in str(time_signature).split("/"))
    beats_per_bar = meter[0] * 4. / meter[1]
    bar_starts_beats = np.arange(len(bars), dtype=float) * beats_per_bar
    start_time, end_time = float(bars[0]["start_time"]), float(bars[-1]["end_time"])
    first_grid = int(np.ceil(start_time / step_seconds - 1e-9))
    times = np.arange(first_grid * step_seconds, end_time, step_seconds)
    starts = np.array([float(r["start_time"]) for r in bars])
    ends = np.array([float(r["end_time"]) for r in bars])
    truth_bar = np.searchsorted(starts, times, side="right") - 1
    valid = (truth_bar >= 0) & (times < ends[np.maximum(truth_bar, 0)])
    times, truth_bar = times[valid], truth_bar[valid]
    within = (times - starts[truth_bar]) / (ends[truth_bar] - starts[truth_bar])
    chords = sorted((dict(r) for r in chord_rows if float(r["end_time"]) > float(r["start_time"])),
                    key=lambda r: (float(r["start_time"]), float(r["end_time"])))
    chord_starts = np.array([(float(r["start_bar"]) - first_bar + float(r["start_fraction"])) * beats_per_bar
                             for r in chords])
    chord_ends = np.array([(float(r["end_bar"]) - first_bar + float(r["end_fraction"])) * beats_per_bar
                           for r in chords])
    if np.any(chord_starts < 0) or np.any(chord_ends <= chord_starts):
        raise ValueError("Chord plan must consist of positive musical intervals")
    if np.any(chord_starts[1:] < chord_ends[:-1] - 1e-8):
        raise ValueError("Chord plan overlaps after chronological sorting")
    return {"times": times, "chords": chords, "bars": bars, "start_time": start_time, "end_time": end_time,
            "initial_bpm": float(bpm), "meter": meter, "beats_per_bar": beats_per_bar,
            "first_annotation_bar": first_bar, "initial_position_beats": 0., "chord_starts_beats": chord_starts,
            "chord_ends_beats": chord_ends, "bar_starts_beats": bar_starts_beats,
            "score_end_beats": len(bars) * beats_per_bar,
            "truth_position_beats": (truth_bar + within) * beats_per_bar, "truth_bar_index": truth_bar}


def samba_follow_takes(root, protocols=("async", "sync")):
    """(data, take) per SAMBA take: the benchmark-style plan plus piano notes."""
    root = Path(root)
    catalog = {r["song_subdir_basename"]: r for r in _rows(root / "JazzSAMBA-catalog/songs.csv")}
    out = []
    for take in samba_takes(root, protocols):
        if take.info.get("excluded"):
            continue
        song, name = take.info["song"], take.info["take"]
        folder = root / f"{take.info['protocol']}_audio/songs/{song}/{name}/annotations"
        data = plan_from_annotations(_rows(folder / "bars.csv"), _rows(folder / "chords.csv"),
                                     catalog[song]["time_signature"], catalog[song]["bpm"])
        data["performance_id"] = f"{take.info['protocol']}/{song}/{name}"
        data["notes_by_instrument"] = {"piano": np.column_stack([take.onsets, take.offsets, take.pitches])}
        out.append((data, take))
    return out
