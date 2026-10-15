"""Does the timing network do better on JazzSAMBA with more of the band as input?

The network was trained on random mixes of non-drum Lakh tracks; the SAMBA
evaluation so far gave it the piano only (the paper's use case). Here each take
is scored with four inputs, merged into one 88-key roll:

  piano                  recorded piano MIDI (sync takes with the timing repair)
  piano+bass             plus the transcribed bass
  piano+bass+horns       plus the transcribed saxophones and trumpet (no drums:
                         the network never saw drums)
  bass                   the transcribed bass alone

each with the BPM condition at the written tempo (as in the results table) and
at the played tempo. Scores are the same as eval_real.py, plus a breakdown by
what the piano is doing in each annotated section (solo, comping, laid out),
time-weighted. Bass and horns are automatic transcriptions, so their errors
are part of the result. Writes per_take.csv and summary.md to --out.

  python eval_samba_inputs.py --data DIR --checkpoint best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from analyze_samba import f_measure, sections_of, within
from eval_real import run_take
from yamaha_timing.lakh import SynthesisConfig
from yamaha_timing.real_data import samba_takes
from yamaha_timing.training import load_checkpoint

INPUTS = {"piano": ("piano",), "piano+bass": ("piano", "bass"), "piano+bass+horns": ("piano", "bass", "horns"),
          "bass": ("bass",)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/samba_inputs")))
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, "cpu")[0].eval()
    try:
        model = model.to(device)
    except RuntimeError as exc:  # a node's GPU can be busy or unavailable; this pass is light on the CPU
        print(f"CUDA unavailable ({exc.__class__.__name__}); running on the CPU", flush=True)
        device = torch.device("cpu")
    max_note = SynthesisConfig().max_note_seconds
    rows, sections_rows = [], []
    for input_name, instruments in INPUTS.items():
        takes = [t for t in samba_takes(args.data, instruments=instruments) if not t.info.get("excluded")]
        if args.limit:
            takes = takes[:args.limit // 2] + takes[-(args.limit - args.limit // 2):]
        for take in takes:
            if not len(take.onsets):
                continue
            played = float(60 / np.median(np.diff(take.beats)))
            sections = sections_of(args.data, take)
            for tempo, bpm in (("written", take.bpm), ("played", played)):
                scores, example = run_take(model, take, bpm, device, max_note)
                rows.append({"input": input_name, "tempo": tempo, "name": take.name, "protocol": take.info["protocol"],
                             "instruments": take.info["instruments"], **{k: round(v, 5) for k, v in scores.items()}})
                for s in sections:
                    start, end = max(s["start"], take.span[0]), min(s["end"], take.span[1])
                    if end - start < 4:
                        continue
                    sections_rows.append({"input": input_name, "tempo": tempo, "role": s["role"], "seconds": end - start,
                                          "beat_f": f_measure(within(example["true_beats"], start, end),
                                                              within(example["pred_beats"], start, end)),
                                          "downbeat_f": f_measure(within(example["true_downbeats"], start, end),
                                                                  within(example["pred_downbeats"], start, end))})
        print(f"{input_name}: {len(takes)} takes", flush=True)
    with open(args.out / "per_take.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = lambda sel, k: float(np.mean([r[k] for r in sel])) if sel else float("nan")
    lines = ["# JazzSAMBA: piano only vs more of the band as input", "",
             "mir_eval F-measure (±70 ms); bass and horns are dataset transcriptions; no drums.", ""]
    for tempo in ("written", "played"):
        lines += [f"## BPM condition: {tempo} tempo", "",
                  "| input | takes | beat F all | downbeat F all | beat F async | downbeat F async | beat F sync | "
                  "downbeat F sync | beat CMLt all | downbeat CMLt all |", "|---|---|---|---|---|---|---|---|---|---|"]
        for input_name in INPUTS:
            sel = [r for r in rows if r["input"] == input_name and r["tempo"] == tempo]
            a = [r for r in sel if r["protocol"] == "async"]
            y = [r for r in sel if r["protocol"] == "sync"]
            lines.append(f"| {input_name} | {len(sel)} | {mean(sel, 'beat_f_measure'):.3f} | {mean(sel, 'downbeat_f_measure'):.3f} | "
                         f"{mean(a, 'beat_f_measure'):.3f} | {mean(a, 'downbeat_f_measure'):.3f} | "
                         f"{mean(y, 'beat_f_measure'):.3f} | {mean(y, 'downbeat_f_measure'):.3f} | "
                         f"{mean(sel, 'beat_cmlt'):.3f} | {mean(sel, 'downbeat_cmlt'):.3f} |")
        lines += ["", "By what the piano is doing (sections, time-weighted, beat F / downbeat F):", "",
                  "| input | piano solo | comping | piano laid out |", "|---|---|---|---|"]
        for input_name in INPUTS:
            cells = []
            for role in ("piano solo", "comping", "laid out"):
                sel = [s for s in sections_rows if s["input"] == input_name and s["tempo"] == tempo and s["role"] == role]
                w = np.array([s["seconds"] for s in sel])
                avg = lambda k: np.average([s[k] if np.isfinite(s[k]) else 0 for s in sel], weights=w) if sel else np.nan
                cells.append(f"{avg('beat_f'):.2f} / {avg('downbeat_f'):.2f}")
            lines.append(f"| {input_name} | " + " | ".join(cells) + " |")
        lines.append("")
    # paired: how many takes improve with the full band (played tempo)
    base = {r["name"]: r for r in rows if r["input"] == "piano" and r["tempo"] == "played"}
    full = {r["name"]: r for r in rows if r["input"] == "piano+bass+horns" and r["tempo"] == "played"}
    common = sorted(set(base) & set(full))
    if common:
        diff = np.array([full[n]["downbeat_f_measure"] - base[n]["downbeat_f_measure"] for n in common])
        diff_b = np.array([full[n]["beat_f_measure"] - base[n]["beat_f_measure"] for n in common])
        lines += ["## Paired, played tempo: piano+bass+horns minus piano", "",
                  f"- takes: {len(common)}; downbeat F better in {int((diff > 0).sum())}, worse in {int((diff < 0).sum())} "
                  f"(median change {np.median(diff):+.3f}); beat F better in {int((diff_b > 0).sum())}, worse in "
                  f"{int((diff_b < 0).sum())} (median change {np.median(diff_b):+.3f})"]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    (args.out / "DONE").touch()


if __name__ == "__main__":
    main()
