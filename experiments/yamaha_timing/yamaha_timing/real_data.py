"""Real piano performances with annotated beats and bars, for evaluation only.

Both loaders return ``Take`` records in seconds: the performed notes (held
keys; pedal is ignored, as in training), the reference beats and downbeats in
our convention (a beat is the time-signature denominator note), and the span
in which the annotations hold.

JazzSAMBA (``samba_takes``): recorded piano MIDI (never transcribed) and
annotated bar times (``annotations/bars.csv``). Beats split each bar evenly by
the song's numerator: exact for async takes, which were played to a click, and
approximate for sync takes (live band), whose bars are annotated by hand.
The BPM condition is the catalog's written tempo.

Sync piano MIDI is mistimed as distributed. Its ticks follow a 175 BPM session
tempo while each file's tempo event holds the song's written tempo, and it
starts a few seconds off the audio and annotations. Measured 2026-10-14: as
stored, onsets are spread evenly over the annotated beat (phase concentration
0.04, versus 0.36 for async). Sync takes are therefore rescaled to 175 BPM and
shifted by a constant per take that samba_sync_timing.py fits from the take's
own piano audio, without annotations (table: samba_sync_timing.csv). Takes
whose fit is not clearly better than its rival (confidence < MIN_CONFIDENCE)
are excluded. Sync results are reported separately as "repaired timing".

ASAP (``asap_takes``): performance MIDI with annotated performance beats and
downbeats. ASAP counts compound meters in dotted beats (6/8 has 2 beats); ours
count the denominator note (6/8 has 6), so each compound beat is split in three
by linear interpolation. Downbeats are unchanged. ASAP has no written tempo:
``bpm`` is each performance's median annotated tempo in our beats, an oracle
the live system would not have.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

SESSION_BPM = 175.0  # tempo of the SAMBA sync recording sessions (see module docstring)
SYNC_TIMING = Path(__file__).resolve().parents[1] / "samba_sync_timing.csv"
MIN_CONFIDENCE = 1.3  # correlation peak over its best rival in samba_sync_timing.py


@dataclass
class Take:
    dataset: str
    name: str
    onsets: np.ndarray
    offsets: np.ndarray
    pitches: np.ndarray
    beats: np.ndarray       # reference beat times, our convention
    downbeats: np.ndarray   # reference bar starts
    span: tuple             # (start, end) seconds where the annotations hold
    bpm: float | None       # BPM condition for the "given" setting
    info: dict = field(default_factory=dict)


def read_midi_notes(path, low=21, high=108):
    """(onsets, offsets, pitches) in seconds from key presses and releases."""
    import mido

    t, active, notes = 0.0, {}, []
    for message in mido.MidiFile(path):
        t += message.time
        if message.type not in ("note_on", "note_off"):
            continue
        key = (message.channel, message.note)
        if key in active:  # a release, or a re-strike that ends the held note
            notes.append((active.pop(key), t, message.note))
        if message.type == "note_on" and message.velocity > 0:
            active[key] = t
    notes += [(start, t, note) for (_, note), start in active.items()]
    notes = np.array([n for n in notes if low <= n[2] <= high and n[1] > n[0]], dtype=float).reshape(-1, 3)
    order = np.argsort(notes[:, 0], kind="stable")
    return notes[order, 0], notes[order, 1], notes[order, 2].astype(np.int64)


def midi_tempo_bpm(path):
    import mido

    midi = mido.MidiFile(path)
    tempo = next((m.tempo for track in midi.tracks for m in track if m.type == "set_tempo"), 500000)
    return 60e6 / tempo


def beat_concentration(onsets, beats):
    """|mean exp(2 pi i * beat position)| of the onsets inside the beat grid: 0 for
    onsets spread evenly over the beat, 1 when all fall at the same point of it."""
    inside = (onsets > beats[0]) & (onsets < beats[-1])
    if inside.sum() < 50:
        return 0.0
    position = np.interp(onsets[inside], beats, np.arange(len(beats)))
    return float(np.abs(np.exp(2j * np.pi * position).mean()))


def sync_timing(path=SYNC_TIMING):
    """(song, take) -> row of samba_sync_timing.csv for the sync takes."""
    return {(r["song"], r["take"]): r for r in csv.DictReader(open(path)) if r["protocol"] == "sync"}


def _split_ids(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")}


def samba_takes(root, protocols=("async", "sync"), timing_table=SYNC_TIMING):
    """All takes; sync takes without a confident timing fix carry info["excluded"]."""
    root = Path(root)
    repairs = sync_timing(timing_table) if "sync" in protocols else {}
    catalog = {row["song_subdir_basename"]: row for row in csv.DictReader(open(root / "JazzSAMBA-catalog/songs.csv"))}
    splits = {}
    for protocol in protocols:
        for split in ("train", "val", "test"):
            for song_id in _split_ids(root / f"{protocol}_audio/splits/{split}.txt"):
                splits[(protocol, song_id)] = split
    takes = []
    for protocol in protocols:
        for bars_csv in sorted(root.glob(f"{protocol}_audio/songs/*/*/annotations/bars.csv")):
            take_dir = bars_csv.parent.parent
            song = catalog[take_dir.parent.name]
            numerator = int(song["time_signature_numerator"])
            rows = list(csv.DictReader(open(bars_csv)))
            starts = np.array([float(r["start_time"]) for r in rows])
            ends = np.array([float(r["end_time"]) for r in rows])
            beats = np.sort(np.concatenate([starts + k * (ends - starts) / numerator for k in range(numerator)]))
            grid = np.append(beats, ends[-1])
            onsets, offsets, pitches = read_midi_notes(take_dir / "piano.mid")
            info = {"protocol": protocol, "split": splits.get((protocol, song["song_id"]), "none"),
                    "song": take_dir.parent.name, "take": take_dir.name, "meter": song["time_signature"],
                    "timing": "as recorded"}
            if protocol == "sync":
                fix = repairs.get((take_dir.parent.name, take_dir.name))
                if fix is None or float(fix["confidence"]) < MIN_CONFIDENCE:
                    info["excluded"] = "no confident timing fix"
                else:
                    scale, shift = float(fix["scale"]), float(fix["shift_s"])
                    onsets, offsets = onsets * scale + shift, offsets * scale + shift
                    info.update({"timing": "repaired", "scale": scale, "shift_s": shift,
                                 "timing_confidence": float(fix["confidence"])})
            info["concentration"] = round(beat_concentration(onsets, grid), 3)
            takes.append(Take("samba", f"{take_dir.parent.name}/{take_dir.name}", onsets, offsets, pitches,
                              beats, starts, (float(starts[0]), float(ends[-1])), float(song["bpm"]), info))
    return takes


def to_denominator_beats(beats, signatures):
    """ASAP beats to denominator-note beats. ``signatures``: sorted (time, "n/d",
    beats per bar); compound bars (n a multiple of beats per bar) split each beat."""
    out = []
    for j, b in enumerate(beats):
        active = [s for s in signatures if s[0] <= b + 1e-6] or signatures[:1]
        numerator, per_bar = int(active[-1][1].split("/")[0]), int(active[-1][2])
        factor = numerator // per_bar if per_bar and numerator % per_bar == 0 else 1
        out.append(b)
        if factor > 1 and j + 1 < len(beats):
            out += list(b + (beats[j + 1] - b) * np.arange(1, factor) / factor)
    return np.array(out)


def asap_takes(root):
    root = Path(root)
    annotations = json.load(open(root / "asap_annotations.json"))
    meta = {row["midi_performance"]: row for row in csv.DictReader(open(root / "metadata.csv"))}
    takes = []
    for performance, a in sorted(annotations.items()):
        beats = np.asarray(a["performance_beats"], dtype=float)
        downbeats = np.asarray(a["performance_downbeats"], dtype=float)
        if len(beats) < 8 or len(downbeats) < 2:
            continue
        signatures = sorted((float(t), s[0], s[1]) for t, s in a["perf_time_signatures"].items())
        ours = to_denominator_beats(beats, signatures)
        compound = sorted({s[1] for s in signatures if int(s[1].split("/")[0]) != int(s[2])})
        onsets, offsets, pitches = read_midi_notes(root / performance)
        row = meta.get(performance, {})
        takes.append(Take("asap", performance, onsets, offsets, pitches, ours, downbeats,
                          (float(beats[0]), float(beats[-1])), float(60.0 / np.median(np.diff(ours))),
                          {"composer": row.get("composer", performance.split("/")[0]), "title": row.get("title", ""),
                           "meter": "+".join(sorted({s[1] for s in signatures})),
                           "meter_class": "compound" if compound else "simple"}))
    return takes
