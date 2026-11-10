"""What the band is doing at each instant, from SAMBA's sections.csv and section_musicians.csv: one place for the
three labellings the older scripts each defined (CODE_REVIEW finding 6).

  kind    8 kinds (harmony_cue/chord_rank.py): head, piano_solo, horn_solo, bass_solo, drums_piano_in,
          drums_piano_out (drum solo with or without the piano), intro_outro, other
  coarse  4 kinds (melody_evidence.section_kinds): head, horn_solo, other_solo (piano, bass or drum solo), other.
          Derived from ``kind``; SAMBA has no other soloists, so the two agree everywhere (tests/test_sections.py)
  piano   what the piano does (sweep_fixed_weights.piano_roles): head, piano_solo, comping (someone else solos),
          laid_out (the piano is not listed for the section, which wins over the rest), other

Instants outside every section get "other". Sections are half-open [start, end).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

KINDS = ("head", "piano_solo", "horn_solo", "bass_solo", "drums_piano_in", "drums_piano_out", "intro_outro", "other")
COARSE = {"head": "head", "horn_solo": "horn_solo", "piano_solo": "other_solo", "bass_solo": "other_solo",
          "drums_piano_in": "other_solo", "drums_piano_out": "other_solo", "intro_outro": "other", "other": "other"}
COARSE_KINDS = ("head", "horn_solo", "other_solo", "other")
PIANO_ROLES = ("head", "piano_solo", "comping", "laid_out", "other")
HORNS = {"trumpet", "saxophone_alto", "saxophone_tenor"}


def _read(samba, performance_id):
    protocol, song, take = performance_id.split("/")
    folder = Path(samba) / f"{protocol}_audio/songs/{song}/{take}/annotations"
    with open(folder / "sections.csv") as handle:
        sections = list(csv.DictReader(handle))
    players = {}
    with open(folder / "section_musicians.csv") as handle:
        for row in csv.DictReader(handle):
            players.setdefault(row["section_index"], set()).add(row["instrument"])
    return sections, players


def labels(samba, performance_id, times):
    """{"kind", "coarse", "piano"}: one string array each, per instant of ``times``."""
    sections, players = _read(samba, performance_id)
    times = np.asarray(times, dtype=float)
    kind = np.full(len(times), "other", dtype=object)
    piano = np.full(len(times), "other", dtype=object)
    for row in sections:
        plays = "piano" in players.get(row["section_index"], set())
        soloist = row["soloist"]
        if row["is_head"] == "True":
            k = "head"
        elif row["is_solo"] == "True":
            k = ("piano_solo" if soloist == "piano" else "horn_solo" if soloist in HORNS else "bass_solo"
                 if soloist == "bass" else ("drums_piano_in" if plays else "drums_piano_out") if soloist == "drums"
                 else "other")
        elif row["is_intro"] == "True" or row["is_outro"] == "True":
            k = "intro_outro"
        else:
            k = "other"
        if not plays:
            p = "laid_out"
        elif row["is_solo"] == "True":
            p = "piano_solo" if soloist == "piano" else "comping"
        elif row["is_head"] == "True":
            p = "head"
        else:
            p = "other"
        inside = (times >= float(row["start_time"])) & (times < float(row["end_time"]))
        kind[inside], piano[inside] = k, p
    return {"kind": kind.astype(str), "coarse": np.array([COARSE[k] for k in kind], dtype=str).reshape(len(times)),
            "piano": piano.astype(str)}
