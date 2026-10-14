"""Lakh MIDI preparation and Maezawa-style performance synthesis.

Preparation (needs ``mido``) reads score MIDI files, applies the label-quality
checks, and stores pitched notes measured in quarter-note beats together with
the tempo and meter maps. Synthesis (numpy only) renders one training example
per call from a stored song, following the paper's description: "a random
combination of tracks to simulate comping", "randomly drifting the tempo and
randomly adding or deleting notes", and a BPM condition "given with 50%
probability" (Maezawa 2023, sec. 2.1).

The paper gives no rates or drift sizes; ours live in ``SynthesisConfig``.
Labels come from the same beat-to-time map that places the notes, so tempo
drift never misaligns them. Added notes, deleted notes, and the track choice
change the input only.

Meters: bar phase is the position within the current bar, whatever its length;
bars follow the file's time-signature events (4/4 before the first event, the
MIDI default). The beat is the time signature's denominator note: a quarter
note in x/4, an eighth note in x/8 (so 6/8 has six beats per bar).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math

import numpy as np

from .config import TimingConfig

DRUM_CHANNEL = 9                 # MIDI channel 10: note numbers are drum sounds
FX_PROGRAMS = range(120, 128)    # General MIDI sound effects: not musical pitches
SNARES = (38, 40)
IGNORE_INDEX = -100
MAX_BAR_QUARTERS = 8             # longer "bars" (e.g. 252/8) are sequencer artifacts


# --------------------------------------------------------------------------
# Meters
# --------------------------------------------------------------------------

def beat_unit(num, den):
    """Beat length in quarter notes (the denominator note), or None for a meter we
    do not label: bars longer than MAX_BAR_QUARTERS are sequencer artifacts."""
    if num < 1 or den < 1 or num * 4 / den > MAX_BAR_QUARTERS:
        return None
    return 4.0 / den


def meter_segments(raw):
    """``(segments, problem)``. Segments are ``(start_tick, num, den)`` with
    restatements merged; 4/4 is assumed before the first event. ``problem`` is
    None or the reason the meter cannot be labelled: no_time_signature,
    unsupported_meter, or meter_event_off_bar_line (an event that does not
    start a bar under the meter in force before it).
    """
    if not raw["signatures"]:
        return None, "no_time_signature"
    events = {}
    for tick, num, den in raw["signatures"]:
        events[tick] = (num, den)  # same tick: the last event wins
    events = sorted(events.items())
    if any(beat_unit(num, den) is None for _, (num, den) in events):
        return None, "unsupported_meter"
    tpb = raw["tpb"]
    current_tick, current = 0, (4, 4)
    segments = [] if events[0][0] == 0 else [(0, 4, 4)]
    for tick, meter in events:
        bar_ticks = Fraction(current[0] * 4 * tpb, current[1])
        if tick and Fraction(tick - current_tick) % bar_ticks:
            return None, "meter_event_off_bar_line"
        if not segments or meter != segments[-1][1:]:
            segments.append((tick, *meter))
        current_tick, current = tick, meter
    return segments, None


def meter_class(segments):
    return f"{segments[0][1]}/{segments[0][2]}" if len(segments) == 1 else "changes"


def bar_and_beat_phase(position, meter_beat, meter_num, meter_den):
    """Bar phase and beat phase in [0, 1), and beat length in quarter notes,
    for quarter-note positions under a meter map."""
    position = np.asarray(position, dtype=np.float64)
    seg = np.clip(np.searchsorted(meter_beat, position, side="right") - 1, 0, len(meter_beat) - 1)
    num, den = meter_num[seg].astype(np.float64), meter_den[seg].astype(np.float64)
    bar_length = num * 4.0 / den
    unit = 4.0 / den  # the denominator note: quarter in x/4, eighth in x/8
    offset = np.mod(position - meter_beat[seg], bar_length)
    return offset / bar_length, np.mod(offset / unit, 1.0), unit


# --------------------------------------------------------------------------
# Preparation: MIDI file -> checked score in beats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckConfig:
    """Label-quality rules from the Lakh survey (experiments/lakh_survey)."""

    min_seconds: float = 30.0
    min_pitched_notes: int = 50
    grid_fraction: float = 0.9      # share of note starts that must follow the grid
    grid_tolerance: float = 1 / 16  # quarter notes: within a 64th of a 16th/triplet position
    fine_tolerance: float = 0.02    # quarter notes: exact on a 32nd/16th-triplet grid
    min_snares: int = 16
    snare_shift_share: float = 0.35  # at most this share on beats 2 and 4 of 4/4 bars -> shifted

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


def _pitched(notes):
    return [n for n in notes if n[4] != DRUM_CHANNEL and n[5] not in FX_PROGRAMS]


def check_score(raw, config=CheckConfig()):
    """Apply the label-quality checks. Returns ``(passed, reason, metrics)``."""
    tpb = raw["tpb"]
    notes = raw["notes"]
    pitched = _pitched(notes)
    metrics = {"pitched_notes": len(pitched)}
    if len(pitched) < config.min_pitched_notes:
        return False, "too_few_notes", metrics
    tempo_beats, tempo_bpms = tempo_map(raw)
    last_beat = max(n[1] for n in pitched) / tpb
    metrics["seconds"] = round(float(score_seconds([last_beat], tempo_beats, tempo_bpms)[0]), 2)
    if metrics["seconds"] < config.min_seconds:
        return False, "too_short", metrics
    segments, problem = meter_segments(raw)
    if problem:
        return False, problem, metrics
    metrics["meter"] = meter_class(segments)

    onsets = np.array([n[0] for n in pitched], dtype=np.float64) / tpb
    near = float(np.mean(_grid_distance(onsets, (4, 3)) <= config.grid_tolerance))
    fine = float(np.mean(_grid_distance(onsets, (8, 6, 4, 3)) <= config.fine_tolerance))
    metrics.update(on_grid=round(near, 4), on_fine_grid=round(fine, 4))
    if near < config.grid_fraction and fine < config.grid_fraction:
        return False, "off_grid", metrics

    # Shifted bars: in 4/4 bars the snare backbeat belongs on beats 2 and 4.
    meter_beat = np.array([t / tpb for t, _, _ in segments])
    snares = np.array([n[0] for n in notes if n[4] == DRUM_CHANNEL and n[2] in SNARES],
                      dtype=np.float64) / tpb
    seg = np.clip(np.searchsorted(meter_beat, snares, side="right") - 1, 0, len(segments) - 1)
    in_44 = np.array([segments[s][1:] == (4, 4) for s in seg], dtype=bool) if len(seg) else np.zeros(0, bool)
    relative = snares[in_44] - meter_beat[seg[in_44]]  # quarter notes from the section's first bar line
    relative = relative[np.abs(relative - np.round(relative)) <= 0.1]
    if len(relative) >= config.min_snares:
        beat_in_bar = np.round(relative).astype(int) % 4
        counts = np.bincount(beat_in_bar, minlength=4)
        share = float((counts[1] + counts[3]) / counts.sum())
        metrics["snare_on_2_4"] = round(share, 4)
        if share <= config.snare_shift_share:
            return False, "snare_on_1_3", metrics
    return True, "ok", metrics


def content_hash(raw):
    """Fingerprint of the pitched notes (start to 1/48 beat, pitch), for spotting copies."""
    tpb = raw["tpb"]
    notes = sorted((round(n[0] / tpb * 48), n[2]) for n in _pitched(raw["notes"]))
    return hashlib.sha1(repr(notes).encode()).hexdigest()


def score_to_song(raw, low=21, high=108):
    """Pitched, non-effect notes in piano range, measured in quarter notes, with
    parts, tempo map and meter map.

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
    segments, problem = meter_segments(raw)
    if problem:
        raise ValueError(f"meter cannot be labelled: {problem}")
    return {
        "start": rows[:, 0].astype(np.float32), "end": rows[:, 1].astype(np.float32),
        "pitch": rows[:, 2].astype(np.uint8), "part": rows[:, 3].astype(np.uint16),
        "tempo_beat": tempo_beats.astype(np.float32), "tempo_bpm": tempo_bpms.astype(np.float32),
        "meter_beat": np.array([t / tpb for t, _, _ in segments], dtype=np.float64),
        "meter_num": np.array([n for _, n, _ in segments], dtype=np.uint8),
        "meter_den": np.array([d for _, _, d in segments], dtype=np.uint8),
    }


# --------------------------------------------------------------------------
# Packed storage: a few arrays per split instead of thousands of files
# --------------------------------------------------------------------------

NOTE_FIELDS = ("start", "end", "pitch", "part")
TEMPO_FIELDS = ("tempo_beat", "tempo_bpm")
METER_FIELDS = ("meter_beat", "meter_num", "meter_den")
GROUPS = ((NOTE_FIELDS, "note_offsets"), (TEMPO_FIELDS, "tempo_offsets"), (METER_FIELDS, "meter_offsets"))


def save_pack(path, songs, ids, metadata=None):
    """Concatenate songs into one compressed ``.npz`` with offset arrays.

    ``metadata`` holds extra per-song arrays (e.g. group, meter class).
    """
    arrays = {}
    for fields, offsets_name in GROUPS:
        lengths = [len(song[fields[0]]) for song in songs]
        arrays[offsets_name] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        for field in fields:
            arrays[field] = np.concatenate([song[field] for song in songs])
    arrays["ids"] = np.asarray(ids, dtype="U64")
    for name, values in (metadata or {}).items():
        arrays[f"meta_{name}"] = np.asarray(values)
    np.savez_compressed(path, **arrays)


class SongPack:
    """Read-only access to a packed split; ``song(i)`` returns array views and
    ``meta[name]`` the per-song metadata arrays."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {name: data[name] for name in data.files}
        self.ids = self.arrays["ids"]
        self.meta = {name[5:]: values for name, values in self.arrays.items() if name.startswith("meta_")}

    def __len__(self):
        return len(self.ids)

    def song(self, index):
        a = self.arrays
        song = {}
        for fields, offsets_name in GROUPS:
            lo, hi = a[offsets_name][index], a[offsets_name][index + 1]
            song.update({field: a[field][lo:hi] for field in fields})
        return song


# --------------------------------------------------------------------------
# Synthesis: stored song -> one training example
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesisConfig:
    """Augmentation sizes. The paper names the operations, not their sizes.

    Per example: deletion probability ~ U(0, max_delete); inserted notes per
    kept note ~ U(0, max_insert); log-tempo random-walk step std per quarter
    note ~ U(0, drift_step_max), pulled back toward the score tempo by
    ``drift_reversion`` per quarter note and capped at +-``drift_limit``.

    ``max_note_seconds`` is cleaning, not augmentation: Lakh files contain stuck
    notes (missing note-offs; 1.7% of training songs hold a note over 60 s, up
    to 45 min), which no pianist plays. Performed notes are cut to this length.
    """

    max_delete: float = 0.1
    max_insert: float = 0.1
    drift_step_max: float = 0.03
    drift_reversion: float = 0.02
    drift_limit: float = 0.4
    hint_probability: float = 0.5
    max_note_seconds: float = 10.0

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


def crop_example(example, rng, frames, context):
    """Random excerpt of ``frames`` labelled frames from a rendered example.

    The excerpt keeps the ``context`` frames before it (the real past, or the
    silent lead-in at the song start) as unlabelled input, so the causal TCN
    sees exactly what it would see live at that point. Event times are not
    carried over. Examples no longer than ``frames`` are returned whole.
    """
    song = len(example["roll"]) - context
    if song <= frames:
        return example
    start = int(rng.integers(0, song - frames + 1))
    window = slice(start, start + context + frames)
    out = {name: example[name][window].copy() for name in ("roll", "beat", "bar", "mask", "bpm", "hint_valid", "beats")}
    out["mask"][:context] = False
    out["beat"][:context] = IGNORE_INDEX
    out["bar"][:context] = IGNORE_INDEX
    return out


def _grid_positions(meter_beat, meter_num, meter_den, lo, hi, bars):
    """Quarter-note positions of beats (or bar starts) within [lo, hi]."""
    positions = []
    for i, (start, num, den) in enumerate(zip(meter_beat, meter_num, meter_den)):
        end = meter_beat[i + 1] if i + 1 < len(meter_beat) else np.inf
        step = num * 4.0 / den if bars else 4.0 / den
        first = start + step * max(0, math.ceil((lo - start) / step - 1e-9))
        stop = min(end, hi + 1e-9)
        if first < stop:
            positions.append(np.arange(first, stop - 1e-9, step))
    return np.concatenate(positions) if positions else np.zeros(0)


def render_example(song, rng, timing=TimingConfig(), synth=SynthesisConfig(), lead_in=True):
    """One synthetic performance of ``song`` with frame-level phase labels.

    Returns numpy arrays over L frames: ``roll`` [L, keys] uint8; ``beat`` and
    ``bar`` [L] int64 bins (IGNORE_INDEX where masked); ``mask`` [L] bool;
    ``bpm`` [L] float32 in the meter's beats (denominator notes) per minute;
    ``hint_valid`` [L] bool; ``beats`` [L] quarter-note position; and the
    reference event times ``beat_times`` and ``downbeat_times`` (seconds from
    time zero, within the rendered frames) for tracking evaluation.
    Time zero is the first selected note. With ``lead_in``, context_frames - 1
    silent, unlabelled frames come first so a whole-song causal pass equals
    live streaming; event times do not count the lead-in.
    """
    start, end, pitch = _choose_notes(song, rng, synth)
    first, last = start.min(), max(end.max(), start.max())
    k0 = math.floor(first)
    quarters = np.arange(k0, math.ceil(last) + 2, dtype=np.float64)

    # Tempo drift: a mean-reverting random walk of log tempo, one step per quarter note.
    step = rng.uniform(0, synth.drift_step_max)
    drift = np.zeros(len(quarters) - 1)
    for j in range(1, len(drift)):
        drift[j] = np.clip((1 - synth.drift_reversion) * drift[j - 1] + step * rng.standard_normal(),
                           -synth.drift_limit, synth.drift_limit)
    score_t = score_seconds(quarters, song["tempo_beat"].astype(np.float64), song["tempo_bpm"].astype(np.float64))
    durations = np.diff(score_t) * np.exp(-drift)          # performed seconds per quarter note
    knots = np.concatenate(([0.0], np.cumsum(durations)))  # performed time at each quarter note
    knots -= np.interp(first, quarters, knots)

    onsets = np.interp(start, quarters, knots)
    offsets = np.minimum(np.interp(end, quarters, knots), onsets + synth.max_note_seconds)
    fr = timing.frame_rate
    frames = int(math.ceil(offsets.max() * fr)) + 1
    roll = rasterize(onsets, offsets, pitch, frames, fr, timing.midi_low, timing.input_pitches)

    position = np.interp(np.arange(frames) / fr, knots, quarters)  # performed time -> quarter notes
    bar_phase, beat_phase, unit = bar_and_beat_phase(position, song["meter_beat"], song["meter_num"],
                                                     song["meter_den"])
    beat_bin = np.minimum((beat_phase * timing.beat_bins).astype(np.int64), timing.beat_bins - 1)
    bar_bin = np.minimum((bar_phase * timing.bar_bins).astype(np.int64), timing.bar_bins - 1)
    quarter_bpm = 60.0 / durations[np.clip(np.floor(position).astype(np.int64) - k0, 0, len(durations) - 1)]
    local_bpm = quarter_bpm / unit  # tempo in the meter's beats, e.g. eighth notes in 6/8
    given = rng.random() < synth.hint_probability
    meter = (song["meter_beat"], song["meter_num"], song["meter_den"])
    event_times = {
        name: np.interp(_grid_positions(*meter, position[0], position[-1], bars), quarters, knots)
        for name, bars in (("beat_times", False), ("downbeat_times", True))
    }

    lead = timing.context_frames - 1 if lead_in else 0
    pad = lambda values, fill: np.concatenate((np.full(lead, fill, dtype=values.dtype), values))
    return {
        "roll": np.concatenate((np.zeros((lead, roll.shape[1]), dtype=np.uint8), roll)),
        "beat": pad(beat_bin, IGNORE_INDEX), "bar": pad(bar_bin, IGNORE_INDEX),
        "mask": pad(np.ones(frames, dtype=bool), False),
        "bpm": pad(local_bpm.astype(np.float32), np.float32(120.0)),
        "hint_valid": pad(np.full(frames, given, dtype=bool), False),
        "beats": pad(position.astype(np.float32), np.float32(np.nan)),
        **event_times,
    }
