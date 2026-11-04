"""How the CREPE Notes transcriptions of the live horn and bass stems compare with the dataset's own (MuScriptor).

Per take and part, in audio seconds (both sets of notes are in the audio's time):
  head-in agreement  note onsets matched one-to-one within 50 ms, same MIDI pitch: F-measure, precision and recall
                     with the dataset's transcription as the reference (it is clean in the head-ins), and the same with
                     pitch classes (octave errors forgiven). This is agreement with another transcriber, not accuracy.
  onset offset       median time from the dataset's onset to the matched CREPE onset
  resting density    notes per bar in solos by the piano, bass or drums, where a horn rests (horns only), for both
  max at once        most notes sounding together (CREPE Notes is one line by construction, so 1)
  out of range       notes outside the instrument's range (concert MIDI: trumpet 52-89, alto 49-84, bass 28-72)
The 8 parts whose dataset transcription broke down (leakage) are marked.

CREPE Notes keeps any note louder than 6/127 of the stem's peak and never uses CREPE's confidence (its check is left
commented out). --min-confidence keeps only notes whose median CREPE confidence over the note (from
<part>.crepe.f0.csv) reaches the value, and --min-velocity only notes at least that loud. --grid instead tries every
pair of the listed values and prints, for preferred and alternate takes separately, the median head-in F and the
resting density of the horns, so a setting chosen on one set of takes can be checked on the other.

  experiments/yamaha_timing/.venv/bin/python experiments/transcription/evaluate_transcriptions.py \
      --midi Dataset/JazzSAMBA-MIDI --crepe Dataset/crepe_notes --out experiments/transcription/results \
      [--min-confidence 0.7] [--min-velocity 0] [--grid]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mido
import numpy as np

RANGE = {"trumpet": (52, 89), "saxophone_alto": (49, 84), "bass": (28, 72)}
BROKEN = {("52-bernies_tune", "preferred", "trumpet"), ("57-chega_de_saudade", "preferred", "trumpet"),
          ("64-love", "alternate", "trumpet"), ("66-nardis", "preferred", "trumpet"),
          ("66-nardis", "alternate", "bass"), ("68-on_the_sunny_side_of_the_street", "alternate", "trumpet"),
          ("74-waltz_for_debby", "preferred", "trumpet"), ("74-waltz_for_debby", "alternate", "trumpet")}
TOLERANCE = 0.05


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def notes(path, with_velocity=False):
    """(onset s, offset s, pitch[, velocity]), sorted by onset."""
    out, sounding, now = [], {}, 0.0
    for message in mido.MidiFile(path):
        now += message.time
        if message.type == "note_on" and message.velocity > 0:
            sounding.setdefault(message.note, []).append((now, message.velocity))
        elif message.type in ("note_on", "note_off") and sounding.get(message.note):
            start, velocity = sounding[message.note].pop(0)
            out.append((start, now, message.note) + ((velocity,) if with_velocity else ()))
    return np.array(sorted(out)) if out else np.zeros((0, 4 if with_velocity else 3))


def crepe_notes_filtered(midi_path, min_confidence=0.0, min_velocity=0):
    """CREPE Notes' notes with median CREPE confidence >= min_confidence and velocity >= min_velocity."""
    n = notes(midi_path, with_velocity=True)
    if not len(n):
        return n[:, :3]
    confidence = np.genfromtxt(str(midi_path).replace(".transcription.mid", ".f0.csv"), delimiter=",",
                               names=True)["confidence"]
    median = np.array([np.median(confidence[int(round(a * 100)):max(int(round(b * 100)), int(round(a * 100)) + 1)])
                       for a, b, *_ in n])
    return n[(median >= min_confidence) & (n[:, 3] >= min_velocity)][:, :3]


def match(reference, estimate, same):
    """One-to-one onset matches within TOLERANCE where same(ref pitch, est pitch): greedy by time distance."""
    pairs = []
    for i, (r_on, _, r_pitch) in enumerate(reference):
        lo, hi = np.searchsorted(estimate[:, 0], [r_on - TOLERANCE, r_on + TOLERANCE + 1e-9])
        for j in range(lo, hi):
            if same(r_pitch, estimate[j, 2]):
                pairs.append((abs(estimate[j, 0] - r_on), i, j))
    pairs.sort()
    used_r, used_e, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
            matched.append((i, j))
    return matched


def scores(reference, estimate, same):
    m = match(reference, estimate, same)
    p = len(m) / len(estimate) if len(estimate) else np.nan
    r = len(m) / len(reference) if len(reference) else np.nan
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    offset = np.median([estimate[j, 0] - reference[i, 0] for i, j in m]) if m else np.nan
    return f, p, r, offset


def max_at_once(n):
    if not len(n):
        return 0
    events = sorted([(a, 1) for a in n[:, 0]] + [(b, -1) for b in n[:, 1]])
    now = best = 0
    for _, step in events:
        now += step
        best = max(best, now)
    return best


RESTING = {"solos-piano", "solos-bass", "solos-drums"}


def evaluate(midi_root, crepe_root, min_confidence=0.0, min_velocity=0):
    rows = []
    for song in sorted(p for p in (midi_root / "sync").iterdir() if p.is_dir()):
        for take in ("preferred", "alternate"):
            folder = song / take
            plan = {int(r["bar"]): r["top_level_section"] for r in _rows(folder / "plan.csv")}
            truth = [(int(r["bar"]), float(r["start_time"]), float(r["end_time"])) for r in _rows(folder / "truth.csv")]

            def within(n, kinds):
                keep = np.zeros(len(n), bool)
                for bar, start, end in truth:
                    if plan.get(bar, "") in kinds:
                        keep |= (n[:, 0] >= start) & (n[:, 0] < end)
                return n[keep]

            rest_bars = sum(1 for b, *_ in truth if plan.get(b, "") in RESTING)
            for part, (low, high) in RANGE.items():
                crepe_path = crepe_root / "sync" / song.name / take / f"{part}.crepe.transcription.mid"
                if not (folder / f"{part}.mid").exists() or not crepe_path.exists():
                    continue
                ours = crepe_notes_filtered(crepe_path, min_confidence, min_velocity)
                theirs = notes(folder / f"{part}.mid")
                head_ours, head_theirs = within(ours, {"head-in"}), within(theirs, {"head-in"})
                f, p, r, offset = scores(head_theirs, head_ours, lambda a, b: a == b)
                f_pc, *_ = scores(head_theirs, head_ours, lambda a, b: a % 12 == b % 12)
                horn = part != "bass" and rest_bars
                rows.append({
                    "song": song.name, "take": take, "part": part,
                    "dataset_broken": (song.name, take, part) in BROKEN,
                    "notes_crepe": len(ours), "notes_dataset": len(theirs),
                    "head_in_f": round(f, 3), "head_in_precision": round(p, 3), "head_in_recall": round(r, 3),
                    "head_in_f_pitch_class": round(f_pc, 3),
                    "onset_offset_ms": round(1000 * offset, 1) if offset == offset else "",
                    "resting_notes_per_bar_crepe": round(len(within(ours, RESTING)) / rest_bars, 2) if horn else "",
                    "resting_notes_per_bar_dataset": round(len(within(theirs, RESTING)) / rest_bars, 2) if horn else "",
                    "max_at_once_crepe": max_at_once(ours), "max_at_once_dataset": max_at_once(theirs),
                    "out_of_range_crepe": int(((ours[:, 2] < low) | (ours[:, 2] > high)).sum()) if len(ours) else 0,
                    "out_of_range_dataset": int(((theirs[:, 2] < low) | (theirs[:, 2] > high)).sum()) if len(theirs) else 0,
                })
    return rows


def med(values):
    values = [float(v) for v in values if v != "" and v == v]
    return np.median(values) if values else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--midi", type=Path, required=True, help="JazzSAMBA-MIDI (plan, truth, dataset parts)")
    parser.add_argument("--crepe", type=Path, required=True, help="the fetched CREPE Notes outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, nargs="+", default=[0.0])
    parser.add_argument("--min-velocity", type=int, nargs="+", default=[0])
    parser.add_argument("--grid", action="store_true", help="try every pair of the listed values")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.grid:
        lines = []
        for c in args.min_confidence:
            for v in args.min_velocity:
                rows = [r for r in evaluate(args.midi, args.crepe, c, v) if not r["dataset_broken"]]
                for part in RANGE:
                    for take in ("preferred", "alternate"):
                        sub = [r for r in rows if r["part"] == part and r["take"] == take]
                        if sub:
                            lines.append({"min_confidence": c, "min_velocity": v, "part": part, "take": take,
                                          "parts": len(sub), "head_in_f": round(med(r["head_in_f"] for r in sub), 3),
                                          "head_in_precision": round(med(r["head_in_precision"] for r in sub), 3),
                                          "head_in_recall": round(med(r["head_in_recall"] for r in sub), 3),
                                          "resting_notes_per_bar": round(med(r["resting_notes_per_bar_crepe"] for r in sub), 2)})
        with open(args.out / "crepe_threshold_grid.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, lines[0].keys())
            writer.writeheader()
            writer.writerows(lines)
        for line in lines:
            print("  ".join(f"{k} {v}" for k, v in line.items()))
        return

    rows = evaluate(args.midi, args.crepe, args.min_confidence[0], args.min_velocity[0])
    with open(args.out / "crepe_vs_dataset.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} parts, min confidence {args.min_confidence[0]}, min velocity {args.min_velocity[0]}")
    for part in RANGE:
        sub = [r for r in rows if r["part"] == part and not r["dataset_broken"]]
        if not sub:
            continue
        print(f"{part:15s} ({len(sub)} clean parts) head-in F {med(r['head_in_f'] for r in sub):.2f} "
              f"(P {med(r['head_in_precision'] for r in sub):.2f}, R {med(r['head_in_recall'] for r in sub):.2f}; "
              f"pitch class {med(r['head_in_f_pitch_class'] for r in sub):.2f}), onset offset "
              f"{med(r['onset_offset_ms'] for r in sub):+.0f} ms; resting notes/bar CREPE "
              f"{med(r['resting_notes_per_bar_crepe'] for r in sub):.2f} vs dataset "
              f"{med(r['resting_notes_per_bar_dataset'] for r in sub):.2f}")
    print("dataset-broken parts:")
    for r in rows:
        if r["dataset_broken"]:
            print(f"  {r['song']:36s} {r['take']:9s} {r['part']:8s} notes {r['notes_crepe']:5d} vs {r['notes_dataset']:5d}  "
                  f"max at once {r['max_at_once_crepe']} vs {r['max_at_once_dataset']}  out of range "
                  f"{r['out_of_range_crepe']} vs {r['out_of_range_dataset']}  head-in F {r['head_in_f']}  resting "
                  f"{r['resting_notes_per_bar_crepe']} vs {r['resting_notes_per_bar_dataset']}")


if __name__ == "__main__":
    main()
