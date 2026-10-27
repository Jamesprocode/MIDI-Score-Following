"""Aligned MIDI of every JazzSAMBA take, with its performance plan, as a separate data folder (the dataset itself
is never modified).

Per take, in <out>/<protocol>/<song>/<take>/:
  ensemble.mid  every part as its own track, plus section markers. The tempo map has one tempo per annotated bar
                (bars.csv), so MIDI bar n+1 is annotated bar n and the file plays in the audio's time. In Logic: open
                ensemble.mid (its tempo map comes with it), then drop the take's mixture.flac at bar 1.
  <part>.mid    each part alone, with the same tempo map (piano, trumpet, saxophone_alto, saxophone_tenor, bass,
                drums; whichever the take has)
  plan.csv      the performance plan, known before the performance: one row per bar with its form measure,
                section and soloist. No times.
  truth.csv     the annotated bar times (bars.csv), for evaluation only
  take.json     song, split, meter, the source of each part, the timing fix applied and what was dropped

Timing, as in yamaha_timing/real_data.py (the follower's own loaders):
  piano, sync takes  recorded MIDI with a wrong clock. Its time is multiplied by the scale and moved by the shift
                     in yamaha_timing/samba_sync_timing.csv (scale = file tempo / 175, the session tempo; shift
                     fitted against the take's own piano audio). Events that land before 0 s are dropped.
  piano, async takes recorded MIDI, as recorded (played to a click)
  horns, bass        the dataset's transcriptions ({protocol}_midi/.../*.transcribed.derived.mid), already in audio
                     time
  drums              sync: the transcription; async: the recorded drums.mid
All MIDI events are kept (velocities, sustain pedal and other controllers), with their times converted; program
changes are replaced by one General MIDI program per part, and each part gets its own channel (drums: 10).

  experiments/yamaha_timing/.venv/bin/python experiments/samba_midi/build_samba_midi.py \
      --samba "Dataset/SAMBA dataset" --out Dataset/JazzSAMBA-MIDI
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mido
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_timing"))
from yamaha_timing.real_data import SYNC_TIMING, read_midi_notes, sync_timing  # noqa: E402

TPB = 960
PARTS = {  # part: (channel, General MIDI program)
    "piano": (0, 0), "trumpet": (1, 56), "saxophone_alto": (2, 65), "saxophone_tenor": (3, 66),
    "bass": (4, 32), "drums": (9, 0),
}
KEEP = {"note_on", "note_off", "control_change", "pitchwheel", "aftertouch", "polytouch"}


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root, protocol, song, take):
    """part -> source MIDI file, for the parts this take has."""
    audio, midi = root / f"{protocol}_audio/songs/{song}/{take}", root / f"{protocol}_midi/songs/{song}/{take}"
    found = {"piano": audio / "piano.mid"}
    for part in ("trumpet", "saxophone_alto", "saxophone_tenor", "bass", "drums"):
        found[part] = midi / f"{part}.transcribed.derived.mid"
    if protocol == "async":
        found["drums"] = audio / "drums.mid"
    return {part: path for part, path in found.items() if path.exists()}


def timed_events(path, scale=1., shift=0.):
    """(seconds, message) of every kept channel message, on the file's own clock, then scaled and shifted."""
    t, events = 0., []
    for message in mido.MidiFile(path):
        t += message.time
        if message.type in KEEP:
            events.append((t * scale + shift, message))
    return events


class BarClock:
    """Seconds <-> ticks through the annotated bars: bar i covers [starts[i], starts[i+1]) and is bar_quarters
    quarter notes long. Before the first bar and after the last, the nearest bar's tempo continues."""

    def __init__(self, starts, ends, bar_quarters):
        self.edges = np.append(starts, ends[-1])
        self.quarters = bar_quarters

    def tick(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        i = np.clip(np.searchsorted(self.edges, seconds, side="right") - 1, 0, len(self.edges) - 2)
        within = (seconds - self.edges[i]) / (self.edges[i + 1] - self.edges[i])
        return np.round((i + within) * self.quarters * TPB).astype(np.int64)

    def tempos(self):
        """(tick, microseconds per quarter) at every bar line."""
        return [(int(round(i * self.quarters * TPB)),
                 int(round((self.edges[i + 1] - self.edges[i]) / self.quarters * 1e6)))
                for i in range(len(self.edges) - 1)]


def conductor(clock, numerator, denominator, markers, name):
    """Tempo-map track: name, meter, one tempo per bar, section markers."""
    events = [(0, 0, mido.MetaMessage("track_name", name=name)),
              (0, 1, mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator))]
    events += [(tick, 2, mido.MetaMessage("set_tempo", tempo=tempo)) for tick, tempo in clock.tempos()]
    events += [(tick, 3, mido.MetaMessage("marker", text=text)) for tick, text in markers]
    return _track(events)


def part_track(part, events, clock):
    channel, program = PARTS[part]
    out = [(0, 0, mido.MetaMessage("track_name", name=part)),
           (0, 1, mido.Message("program_change", channel=channel, program=program))]
    ticks = clock.tick([t for t, _ in events]) if events else []
    struck = {}  # (channel, note) -> tick of its last note-on; a note shorter than a tick keeps one tick
    for k, ((_, message), tick) in enumerate(zip(events, ticks)):
        tick, key = int(tick), (message.channel, getattr(message, "note", None))
        if message.type == "note_on" and message.velocity > 0:
            struck[key] = tick
        elif message.type in ("note_on", "note_off") and struck.get(key) == tick:
            tick += 1
        out.append((tick, 2 + k, message.copy(channel=channel, time=0)))
    return _track(out)


def _track(events):
    track, now = mido.MidiTrack(), 0
    for tick, _, message in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(message.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def save(path, tracks):
    midi = mido.MidiFile(type=1, ticks_per_beat=TPB)
    midi.tracks.extend(tracks)
    midi.save(path)


def plan_rows(bars, sections):
    """One row per bar: form measure, section, soloist (the section that holds the middle of the bar)."""
    spans = [(float(s["start_bar"]) + float(s["start_fraction"]), float(s["end_bar"]) + float(s["end_fraction"]), s)
             for s in sections]
    rows = []
    for b in bars:
        middle = int(b["bar"]) + .5
        here = next((s for lo, hi, s in spans if lo <= middle < hi), None)
        rows.append({"bar": int(b["bar"]), "measure": int(b["measure"]),
                     "section": here["section"] if here else "",
                     "top_level_section": here["top_level_section"] if here else "",
                     "is_head": here["is_head"] if here else "False",
                     "is_solo": here["is_solo"] if here else "False",
                     "soloist": here["soloist"] if here and here["soloist"] != "NA" else ""})
    return rows


def build_take(root, protocol, song, take, catalog, repairs, splits, out):
    annotations = root / f"{protocol}_audio/songs/{song}/{take}/annotations"
    bars, sections = _rows(annotations / "bars.csv"), _rows(annotations / "sections.csv")
    numerator, denominator = int(catalog["time_signature_numerator"]), int(catalog["time_signature_denominator"])
    starts = np.array([float(b["start_time"]) for b in bars])
    ends = np.array([float(b["end_time"]) for b in bars])
    clock = BarClock(starts, ends, numerator * 4 / denominator)
    markers = [(int(clock.tick(float(s["start_time"]))), s["section"]) for s in sections]
    folder = out / protocol / song / take
    folder.mkdir(parents=True, exist_ok=True)
    info = {"protocol": protocol, "song": song, "take": take, "title": catalog["title"],
            "song_id": int(catalog["song_id"]), "split": splits.get(catalog["song_id"], "none"),
            "meter": catalog["time_signature"], "written_bpm": float(catalog["bpm"]), "form": catalog["form"],
            "bars": len(bars), "bar_gap_max_s": round(float(np.abs(starts[1:] - ends[:-1]).max()), 4)
            if len(bars) > 1 else 0., "parts": {}}
    tracks = {}
    for part, path in sources(root, protocol, song, take).items():
        scale, shift, fix = 1., 0., None
        if part == "piano" and protocol == "sync":
            fix = repairs[(song, take)]
            scale, shift = float(fix["scale"]), float(fix["shift_s"])
        events = timed_events(path, scale, shift)
        notes_before = sum(m.type == "note_on" and m.velocity > 0 for _, m in events)
        events = [(t, m) for t, m in events if t >= 0]
        kept = sum(m.type == "note_on" and m.velocity > 0 for _, m in events)
        tracks[part] = part_track(part, events, clock)
        info["parts"][part] = {"source": str(path.relative_to(root)), "sha256": sha256(path), "notes": kept,
                               "dropped_before_0s": notes_before - kept,
                               "timing": "as distributed" if fix is None else
                               {"scale": scale, "shift_s": shift, "confidence": float(fix["confidence"]),
                                "from": "yamaha_timing/samba_sync_timing.csv"}}
        save(folder / f"{part}.mid", [conductor(clock, numerator, denominator, markers, "tempo map"), tracks[part]])
    save(folder / "ensemble.mid", [conductor(clock, numerator, denominator, markers, f"{catalog['title']} ({take})")]
         + list(tracks.values()))
    with open(folder / "plan.csv", "w", newline="") as handle:
        rows = plan_rows(bars, sections)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(folder / "truth.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bar", "measure", "start_time", "end_time"])
        writer.writeheader()
        writer.writerows({k: b[k] for k in writer.fieldnames} for b in bars)
    (folder / "take.json").write_text(json.dumps(info, indent=1))
    return info, clock


def check_take(root, protocol, song, take, folder, clock, repairs):
    """Read the written files back: bar lines against bars.csv, piano onsets against the follower's loader."""
    midi = mido.MidiFile(folder / "ensemble.mid")
    tempo_track = midi.tracks[0]
    tick, seconds, tempo, bar_times = 0, 0., 500000, {}
    lines = {t for t, _ in clock.tempos()}
    for message in tempo_track:  # seconds of every bar line, from the written tempo map
        seconds += mido.tick2second(message.time, TPB, tempo)
        tick += message.time
        if message.type == "set_tempo":
            tempo = message.tempo
            if tick in lines:
                bar_times[tick] = seconds
    bar_error = float(np.max(np.abs(np.array(sorted(bar_times.values())) - clock.edges[:len(bar_times)])))
    result = {"bar_line_error_ms": round(bar_error * 1e3, 3)}
    piano = folder / "piano.mid"
    if piano.exists():
        onsets, _, _ = read_midi_notes(root / f"{protocol}_audio/songs/{song}/{take}/piano.mid")
        if protocol == "sync":
            fix = repairs[(song, take)]
            onsets = onsets * float(fix["scale"]) + float(fix["shift_s"])
        written, _, _ = read_midi_notes(piano)
        expected = np.sort(onsets[onsets >= 0])
        nearest = np.clip(np.searchsorted(expected, written), 1, len(expected) - 1)
        error = np.minimum(np.abs(written - expected[nearest - 1]), np.abs(written - expected[nearest]))
        result["piano_onsets"] = [int(len(expected)), int(len(written))]
        result["piano_onset_error_ms"] = round(float(error.max()) * 1e3, 3)  # each written onset to the nearest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.samba
    catalog = {r["song_subdir_basename"]: r for r in _rows(root / "JazzSAMBA-catalog/songs.csv")}
    repairs = sync_timing(SYNC_TIMING)
    report = []
    for protocol in ("sync", "async"):
        splits = {}
        for split in ("train", "val", "test"):
            for line in (root / f"{protocol}_audio/splits/{split}.txt").read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    splits[line.strip()] = split
        for bars_csv in sorted(root.glob(f"{protocol}_audio/songs/*/*/annotations/bars.csv")):
            take_dir = bars_csv.parent.parent
            song, take = take_dir.parent.name, take_dir.name
            info, clock = build_take(root, protocol, song, take, catalog[song], repairs, splits, args.out)
            check = check_take(root, protocol, song, take, args.out / protocol / song / take, clock, repairs)
            info["check"] = check
            (args.out / protocol / song / take / "take.json").write_text(json.dumps(info, indent=1))
            report.append({"take": f"{protocol}/{song}/{take}", "split": info["split"], "bars": info["bars"],
                           "parts": "+".join(info["parts"]),
                           "piano_dropped": info["parts"].get("piano", {}).get("dropped_before_0s", ""),
                           **{k: v if not isinstance(v, list) else "/".join(map(str, v)) for k, v in check.items()}})
            print(json.dumps(report[-1]), flush=True)
    with open(args.out / "takes.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)


if __name__ == "__main__":
    main()
