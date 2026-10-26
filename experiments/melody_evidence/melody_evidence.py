"""Melody evidence for the lead-sheet follower: does the horn's recent line match the tune at this place in the chart?

The tune (the reference) is a set of note onsets at chart positions (form measure, beat in the bar), each with a
pitch class. JazzSAMBA has no written melodies, so the reference is the *other* take's horn head: the horn notes
(dataset transcriptions) played during that take's head-in and head-out, placed on the chart with that take's
annotated bars. It stands in for a lead-sheet melody. It is a different performance of the same tune by the same
band, so it never contains the notes being followed. ``reference="same"`` uses the followed take's own head
instead, which is only a sanity check of the code (it should be near perfect in the heads).

At each update the follower asks, for every place q in the unfolded chart: of the horn notes started in the last
``window_beats`` beats, what share match a tune note of the same pitch class (any octave) within
``tolerance_beats`` of where that note would sit if we were at q now? Places in bars whose measure the reference
never covers get the average score (no opinion). With fewer than ``min_notes`` recent notes there is no evidence.
The likelihood is ``floor + (1 - floor) * share``, so one bad match can never rule a place out.

The tune is keyed by form measure, so every chorus looks the same to this cue (as to harmony); telling choruses
apart is left to the follower's counting.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS / "yamaha_follower") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS / "yamaha_follower"))
import follower_data  # noqa: E402,F401  (sys.path for yamaha_timing)
from yamaha_timing.real_data import _band_notes  # noqa: E402

OTHER_TAKE = {"preferred": "alternate", "alternate": "preferred"}


@dataclass(frozen=True)
class MelodyConfig:
    window_beats: float = 8.0       # how far back the line is compared (heads have about 1 horn note per beat)
    tolerance_beats: float = 0.5    # slack for swing, anticipation and transcription timing
    min_notes: int = 2              # fewer recent notes: no evidence
    floor: float = 0.1              # likelihood of a place where no recent note matches
    reference: str = "other"        # "other" take's head (the test) or the "same" take's head (code check only)


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def horn_notes(root, protocol, song, take):
    """(onset, offset, pitch) rows of all horn transcriptions of a take, by onset; and the horns found."""
    parts, present = _band_notes(Path(root), protocol, song, take, ("horns",))
    if not parts:
        return np.empty((0, 3)), present
    notes = np.concatenate(parts)
    return notes[np.argsort(notes[:, 0], kind="stable")], present


def reference_tune(root, protocol, song, take, beats_per_bar):
    """Horn onsets in the head sections of ``take``: form measure, beat in the bar, pitch class; plus the
    measures those heads cover."""
    folder = Path(root) / f"{protocol}_audio/songs/{song}/{take}/annotations"
    bars, sections = _rows(folder / "bars.csv"), _rows(folder / "sections.csv")
    starts = np.array([float(b["start_time"]) for b in bars])
    ends = np.array([float(b["end_time"]) for b in bars])
    measures = np.array([int(b["measure"]) for b in bars])
    heads = [(float(s["start_time"]), float(s["end_time"])) for s in sections if s["is_head"] == "True"]
    notes, horns = horn_notes(root, protocol, song, take)
    onsets = notes[:, 0]
    in_head = np.zeros(len(onsets), dtype=bool)
    for a, b in heads:
        in_head |= (onsets >= a) & (onsets < b)
    bar = np.searchsorted(starts, onsets, side="right") - 1
    keep = in_head & (bar >= 0) & (onsets < ends[np.maximum(bar, 0)])
    bar = bar[keep]
    middles = (starts + ends) / 2
    covered = {int(m) for m, t in zip(measures, middles) if any(a <= t < b for a, b in heads)}
    return {"measure": measures[bar], "beat": (onsets[keep] - starts[bar]) / (ends[bar] - starts[bar]) * beats_per_bar,
            "pitch_class": notes[keep, 2].astype(int) % 12, "covered": covered, "horns": horns,
            "source": f"{protocol}/{song}/{take}"}


class MelodyEvidence:
    """The melody cue for one take, on the follower's position grid (``bins_per_beat`` steps per beat)."""

    def __init__(self, data, root, config=MelodyConfig(), bins_per_beat=8):
        protocol, song, take = data["performance_id"].split("/")
        self.config, self.step = config, 1. / bins_per_beat
        per_bar = int(round(data["beats_per_bar"] * bins_per_beat))
        source = take if config.reference == "same" else OTHER_TAKE[take]
        self.reference = reference_tune(root, protocol, song, source, data["beats_per_bar"])
        measures = np.array([int(b["measure"]) for b in data["bars"]])  # unfolded bar -> form measure
        tune = np.zeros((len(measures) * per_bar, 12), dtype=bool)
        ref = self.reference
        for i, m in enumerate(measures):
            here = ref["measure"] == m
            index = np.round(i * per_bar + ref["beat"][here] * bins_per_beat).astype(int)
            inside = (index >= 0) & (index < len(tune))
            tune[index[inside], ref["pitch_class"][here][inside]] = True
        spread = tune.copy()  # a tune note counts within +-tolerance of where it is written
        for shift in range(1, int(round(config.tolerance_beats * bins_per_beat)) + 1):
            spread[shift:] |= tune[:-shift]
            spread[:-shift] |= tune[shift:]
        self.tune = spread
        self.known = np.repeat([m in ref["covered"] for m in measures], per_bar)
        notes, self.horns = horn_notes(root, protocol, song, take)
        self.onsets, self.pitch_classes = notes[:, 0], notes[:, 2].astype(int) % 12

    def scores(self, now, bpm, n_positions):
        """Share of the recent horn notes that match the tune, for positions 0 .. n_positions-1 of the grid
        (None when there is no evidence). ``bpm`` turns how long ago a note started into beats."""
        window = self.config.window_beats * 60. / bpm
        lo, hi = np.searchsorted(self.onsets, [now - window, now], side="right")
        if hi - lo < self.config.min_notes:
            return None
        n = min(n_positions, len(self.tune))
        known = np.zeros(n_positions, dtype=bool)
        known[:n] = self.known[:n]
        if not known.any():
            return None
        lags = np.round((now - self.onsets[lo:hi]) * bpm / 60. / self.step).astype(int)
        index = np.arange(n_positions)
        hits = np.zeros(n_positions)
        for lag, pitch_class in zip(lags, self.pitch_classes[lo:hi]):
            source = index - lag
            inside = (source >= 0) & (source < len(self.tune))
            hits[inside] += self.tune[source[inside], pitch_class]
        share = hits / (hi - lo)
        share[~known] = share[known].mean()
        return share

    def log_likelihood(self, now, bpm, n_positions):
        share = self.scores(now, bpm, n_positions)
        if share is None:
            return None
        return np.log(self.config.floor + (1 - self.config.floor) * share)


def section_kinds(root, performance_id, times):
    """Per instant: head / horn_solo / other_solo / other (intro, outro, interludes), from sections.csv."""
    protocol, song, take = performance_id.split("/")
    sections = _rows(Path(root) / f"{protocol}_audio/songs/{song}/{take}/annotations/sections.csv")
    kinds = np.full(len(times), "other", dtype=object)
    for row in sections:
        if row["is_head"] == "True":
            kind = "head"
        elif row["is_solo"] == "True":
            kind = "horn_solo" if row["soloist"].startswith(("trumpet", "saxophone")) else "other_solo"
        else:
            continue
        kinds[(times >= float(row["start_time"])) & (times < float(row["end_time"]))] = kind
    return kinds
