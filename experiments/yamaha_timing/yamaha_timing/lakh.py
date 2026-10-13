"""Lakh MIDI preparation and Maezawa-style performance synthesis.

Preparation (needs ``mido``) reads score MIDI files, applies the label-quality
checks, and stores pitched notes measured in beats. Synthesis (numpy only)
renders one training example per call from a stored song, following the
paper's description: "a random combination of tracks to simulate comping",
"randomly drifting the tempo and randomly adding or deleting notes", and a
BPM condition "given with 50% probability" (Maezawa 2023, sec. 2.1).

The paper gives no rates or drift sizes; ours live in ``SynthesisConfig``.
Beat labels come from the same beat-to-time map that places the notes, so
tempo drift never misaligns them. Added notes, deleted notes, and the track
choice change the input only. Bar phase assumes 4/4 with bars starting at
beat 0, which the preparation checks enforce.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math

import numpy as np

from .config import TimingConfig

DRUM_CHANNEL = 9                 # MIDI channel 10: note numbers are drum sounds
FX_PROGRAMS = range(120, 128)    # General MIDI sound effects: not musical pitches
SNARES = (38, 40)
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Preparation: MIDI file -> checked score in beats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckConfig:
    """Label-quality rules from the 2,000-file Lakh survey (experiments/lakh_survey)."""

    min_seconds: float = 30.0
    min_pitched_notes: int = 50
    grid_fraction: float = 0.9      # share of note starts that must follow the grid
    grid_tolerance: float = 1 / 16  # beats: within a 64th note of a 16th/triplet position
    fine_tolerance: float = 0.02    # beats: exact positions on a 32nd/16th-triplet grid
    min_snares: int = 16
    snare_shift_share: float = 0.35  # at most this share on beats 2 and 4 -> bars shifted

    def to_dict(self):
        return asdict(self)


def read_score(path):
    """Return tick-level notes, tempo and meter events from a type 0/1 MIDI file.

    Notes are ``(on_tick, off_tick, pitch, track, channel, program)``. A note's
    program is the one in effect at its note-on. Unterminated notes are dropped.
    """
    import mido

    mid = mido.MidiFile(path, clip=True)
    if mid.type == 2:
        raise ValueError("type 2 MIDI files are not supported")
    tpb = mid.ticks_per_beat
    if tpb <= 0:
        raise ValueError("SMPTE time division is not supported")
    notes, tempos, signatures = [], [], []
    for track_index, track in enumerate(mid.tracks):
        tick, program, sounding = 0, {}, defaultdict(list)
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempos.append((tick, msg.tempo))
                elif msg.type == "time_signature":
                    signatures.append((tick, msg.numerator, msg.denominator))
                continue
            if msg.type == "program_change":
                program[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                sounding[(msg.channel, msg.note)].append((tick, program.get(msg.channel, 0)))
            elif msg.type in ("note_off", "note_on"):
                starts = sounding.get((msg.channel, msg.note))
                if starts:
                    on, prog = starts.pop(0)
                    notes.append((on, max(tick, on), msg.note, track_index, msg.channel, prog))
    notes.sort()
    return {"tpb": tpb, "notes": notes, "tempos": sorted(tempos), "signatures": sorted(signatures)}


def _grid_distance(beats, divisions):
    frac = beats - np.floor(beats)
    return np.min([np.abs(frac * d - np.round(frac * d)) / d for d in divisions], axis=0)


def tempo_map(raw):
    """Beat positions and BPM of each tempo segment; 120 BPM before the first event."""
    tpb = raw["tpb"]
    beats, bpms = [0.0], [120.0]
    for tick, micros in raw["tempos"]:
        beat = tick / tpb
        if beat == beats[-1]:
            bpms[-1] = 60e6 / micros
        else:
            beats.append(beat)
            bpms.append(60e6 / micros)
    return np.asarray(beats, dtype=np.float64), np.asarray(bpms, dtype=np.float64)


def score_seconds(beats, tempo_beats, tempo_bpms):
    """Score time of beat positions under a piecewise-constant tempo map."""
    beats = np.asarray(beats, dtype=np.float64)
    seconds_per_beat = 60.0 / tempo_bpms
    starts = np.concatenate(([0.0], np.cumsum(np.diff(tempo_beats) * seconds_per_beat[:-1])))
    segment = np.clip(np.searchsorted(tempo_beats, beats, side="right") - 1, 0, len(tempo_beats) - 1)
    return starts[segment] + (beats - tempo_beats[segment]) * seconds_per_beat[segment]


def check_score(raw, config=CheckConfig()):
    """Apply the label-quality checks. Returns ``(passed, reason, metrics)``."""
    tpb = raw["tpb"]
    notes = raw["notes"]
    pitched = [n for n in notes if n[4] != DRUM_CHANNEL and n[5] not in FX_PROGRAMS]
    metrics = {"pitched_notes": len(pitched)}
    if len(pitched) < config.min_pitched_notes:
        return False, "too_few_notes", metrics
    tempo_beats, tempo_bpms = tempo_map(raw)
    last_beat = max(n[1] for n in pitched) / tpb
    metrics["seconds"] = round(float(score_seconds([last_beat], tempo_beats, tempo_bpms)[0]), 2)
    if metrics["seconds"] < config.min_seconds:
        return False, "too_short", metrics
    signatures = raw["signatures"]
    if not signatures:
        return False, "no_time_signature", metrics
    if any((num, den) != (4, 4) for _, num, den in signatures):
        return False, "not_constant_4_4", metrics
    if any(tick % (4 * tpb) for tick, _, _ in signatures):
        return False, "meter_event_off_bar_line", metrics

    onsets = np.array([n[0] for n in pitched], dtype=np.float64) / tpb
    near = float(np.mean(_grid_distance(onsets, (4, 3)) <= config.grid_tolerance))
    fine = float(np.mean(_grid_distance(onsets, (8, 6, 4, 3)) <= config.fine_tolerance))
    metrics.update(on_grid=round(near, 4), on_fine_grid=round(fine, 4))
    if near < config.grid_fraction and fine < config.grid_fraction:
        return False, "off_grid", metrics

    snares = np.array([n[0] for n in notes if n[4] == DRUM_CHANNEL and n[2] in SNARES],
                      dtype=np.float64) / tpb
    snares = snares[np.abs(snares - np.round(snares)) <= 0.1]
    if len(snares) >= config.min_snares:
        counts = np.bincount(np.round(snares).astype(int) % 4, minlength=4)
        share = float((counts[1] + counts[3]) / counts.sum())
        metrics["snare_on_2_4"] = round(share, 4)
        if share <= config.snare_shift_share:
            return False, "snare_on_1_3", metrics
    return True, "ok", metrics


def score_to_song(raw, low=21, high=108):
    """Pitched, non-effect notes in piano range, measured in beats, with parts.

    A part is one (track, channel) pair. Drum-channel and sound-effect notes are
    removed here; the tracker's input is a pianist's keys.
    """
    tpb = raw["tpb"]
    part_ids = {}
    rows = []
    for on, off, pitch, track, channel, program in raw["notes"]:
        if channel == DRUM_CHANNEL or program in FX_PROGRAMS or not low <= pitch <= high:
            continue
        part = part_ids.setdefault((track, channel), len(part_ids))
        rows.append((on / tpb, off / tpb, pitch, part))
    rows = np.array(rows, dtype=np.float64).reshape(-1, 4)
    tempo_beats, tempo_bpms = tempo_map(raw)
    return {
        "start": rows[:, 0].astype(np.float32), "end": rows[:, 1].astype(np.float32),
        "pitch": rows[:, 2].astype(np.uint8), "part": rows[:, 3].astype(np.uint16),
        "tempo_beat": tempo_beats.astype(np.float32), "tempo_bpm": tempo_bpms.astype(np.float32),
    }


# --------------------------------------------------------------------------
# Packed storage: a few arrays per split instead of thousands of files
# --------------------------------------------------------------------------

NOTE_FIELDS = ("start", "end", "pitch", "part")
TEMPO_FIELDS = ("tempo_beat", "tempo_bpm")


def save_pack(path, songs, ids):
    """Concatenate songs into one compressed ``.npz`` with offset arrays."""
    arrays = {}
    for fields, offsets_name in ((NOTE_FIELDS, "note_offsets"), (TEMPO_FIELDS, "tempo_offsets")):
        lengths = [len(song[fields[0]]) for song in songs]
        arrays[offsets_name] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        for field in fields:
            arrays[field] = np.concatenate([song[field] for song in songs])
    arrays["ids"] = np.asarray(ids, dtype="U64")
    np.savez_compressed(path, **arrays)


class SongPack:
    """Read-only access to a packed split; ``song(i)`` returns array views."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {name: data[name] for name in data.files}
        self.ids = self.arrays["ids"]

    def __len__(self):
        return len(self.ids)

    def song(self, index):
        a = self.arrays
        n0, n1 = a["note_offsets"][index], a["note_offsets"][index + 1]
        t0, t1 = a["tempo_offsets"][index], a["tempo_offsets"][index + 1]
        song = {field: a[field][n0:n1] for field in NOTE_FIELDS}
        song.update({field: a[field][t0:t1] for field in TEMPO_FIELDS})
        return song


# --------------------------------------------------------------------------
# Synthesis: stored song -> one training example
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesisConfig:
    """Augmentation sizes. The paper names the operations, not their sizes.

    Per example: deletion probability ~ U(0, max_delete); inserted notes per
    kept note ~ U(0, max_insert); log-tempo random-walk step std per beat ~
    U(0, drift_step_max), pulled back toward the score tempo by
    ``drift_reversion`` per beat and capped at +-``drift_limit``.
    """

    max_delete: float = 0.1
    max_insert: float = 0.1
    drift_step_max: float = 0.03
    drift_reversion: float = 0.02
    drift_limit: float = 0.4
    hint_probability: float = 0.5
    beats_per_bar: int = 4

    def to_dict(self):
        return asdict(self)


def rasterize(onsets_s, offsets_s, pitches, frames, frame_rate, low=21, keys=88):
    """Binary held-key roll with the same event convention as MidiFrameRasterizer.

    Frame k (time k / frame_rate) includes every event at or before that time.
    """
    k_on = np.ceil((np.asarray(onsets_s) - 1e-9) * frame_rate).astype(np.int64)
    k_off = np.ceil((np.asarray(offsets_s) - 1e-9) * frame_rate).astype(np.int64)
    keep = k_off > k_on
    k_on, k_off = np.clip(k_on[keep], 0, frames), np.clip(k_off[keep], 0, frames)
    column = np.asarray(pitches)[keep].astype(np.int64) - low
    counts = np.zeros((frames + 1, keys), dtype=np.int32)
    np.add.at(counts, (k_on, column), 1)
    np.add.at(counts, (k_off, column), -1)
    return (np.cumsum(counts[:-1], axis=0) > 0).astype(np.uint8)


def _choose_notes(song, rng, synth):
    parts = np.unique(song["part"])
    for _ in range(5):
        chosen = rng.choice(parts, size=rng.integers(1, len(parts) + 1), replace=False)
        selected = np.isin(song["part"], chosen)
        if selected.sum() >= 8:
            break
    else:
        selected = np.ones(len(song["part"]), dtype=bool)
    start = song["start"][selected].astype(np.float64)
    end = song["end"][selected].astype(np.float64)
    pitch = song["pitch"][selected].astype(np.int64)

    keep = rng.random(len(start)) >= rng.uniform(0, synth.max_delete)
    if not keep.any():
        keep[rng.integers(len(keep))] = True
    start, end, pitch = start[keep], end[keep], pitch[keep]

    added = rng.binomial(len(start), rng.uniform(0, synth.max_insert))
    if added:
        donor = rng.integers(len(start), size=added)
        new_start = rng.uniform(start.min(), start.max(), size=added)
        new_end = new_start + (end[donor] - start[donor])
        new_pitch = rng.integers(pitch.min(), pitch.max() + 1, size=added)
        start = np.concatenate((start, new_start))
        end = np.concatenate((end, new_end))
        pitch = np.concatenate((pitch, new_pitch))
    order = np.argsort(start, kind="stable")
    return start[order], end[order], pitch[order]


def render_example(song, rng, timing=TimingConfig(), synth=SynthesisConfig(), lead_in=True):
    """One synthetic performance of ``song`` with frame-level phase labels.

    Returns numpy arrays over L frames: ``roll`` [L, keys] uint8, ``beat`` and
    ``bar`` [L] int64 bins (IGNORE_INDEX where masked), ``mask`` [L] bool,
    ``bpm`` [L] float32 and ``hint_valid`` [L] bool. Time zero is the first
    selected note. With ``lead_in``, context_frames - 1 silent, unlabeled
    frames come first so a whole-song causal pass equals live streaming.
    """
    start, end, pitch = _choose_notes(song, rng, synth)
    first, last = start.min(), max(end.max(), start.max())
    k0 = math.floor(first)
    beats = np.arange(k0, math.ceil(last) + 2, dtype=np.float64)

    # Tempo drift: a mean-reverting random walk of log tempo, one step per beat.
    step = rng.uniform(0, synth.drift_step_max)
    drift = np.zeros(len(beats) - 1)
    for j in range(1, len(drift)):
        drift[j] = np.clip((1 - synth.drift_reversion) * drift[j - 1] + step * rng.standard_normal(),
                           -synth.drift_limit, synth.drift_limit)
    score_t = score_seconds(beats, song["tempo_beat"].astype(np.float64), song["tempo_bpm"].astype(np.float64))
    durations = np.diff(score_t) * np.exp(-drift)          # performed seconds per beat
    knots = np.concatenate(([0.0], np.cumsum(durations)))  # performed time at each integer beat
    knots -= np.interp(first, beats, knots)

    onsets = np.interp(start, beats, knots)
    offsets = np.interp(end, beats, knots)
    fr = timing.frame_rate
    frames = int(math.ceil(offsets.max() * fr)) + 1
    roll = rasterize(onsets, offsets, pitch, frames, fr, timing.midi_low, timing.input_pitches)

    position = np.interp(np.arange(frames) / fr, knots, beats)  # performed time -> beats
    phase = position - np.floor(position)
    beat_bin = np.minimum((phase * timing.beat_bins).astype(np.int64), timing.beat_bins - 1)
    bar_phase = np.mod(position, synth.beats_per_bar) / synth.beats_per_bar
    bar_bin = np.minimum((bar_phase * timing.bar_bins).astype(np.int64), timing.bar_bins - 1)
    local_bpm = 60.0 / durations[np.clip(np.floor(position).astype(np.int64) - k0, 0, len(durations) - 1)]
    given = rng.random() < synth.hint_probability

    lead = timing.context_frames - 1 if lead_in else 0
    pad = lambda values, fill: np.concatenate((np.full(lead, fill, dtype=values.dtype), values))
    return {
        "roll": np.concatenate((np.zeros((lead, roll.shape[1]), dtype=np.uint8), roll)),
        "beat": pad(beat_bin, IGNORE_INDEX), "bar": pad(bar_bin, IGNORE_INDEX),
        "mask": pad(np.ones(frames, dtype=bool), False),
        "bpm": pad(local_bpm.astype(np.float32), np.float32(120.0)),
        "hint_valid": pad(np.full(frames, given, dtype=bool), False),
        "beats": pad(position.astype(np.float32), np.float32(np.nan)),
    }
