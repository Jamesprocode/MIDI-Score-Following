"""Transcribe the live JazzSAMBA horn and bass stems with CREPE Notes (Riley & Dixon, SMC 2023).

For every sync take's trumpet, saxophone_alto and bass stem (sync_audio/songs/<song>/<take>/<part>.flac), as the
crepe_notes command line does it (crepe_notes 0.2.0): CREPE, full model, 10 ms steps, Viterbi smoothing, on the stem
as loaded by librosa (native rate, mono); then crepe_notes.process with its defaults (sensitivity 0.001, minimum
velocity 6, tuning offset estimated, repeated notes split at madmom CNN onsets), except a minimum duration of 0.05 s
for bass instead of 0.03 s, as the crepe_notes README advises for double bass.

Per stem, in <out>/sync/<song>/<take>/ (next to a symlink <part>.flac to the stem, so nothing is written into the
data):
  <part>.crepe.transcription.mid   the notes
  <part>.crepe.f0.csv              time, frequency, confidence (10 ms)
  <part>.onsets.npz                madmom onset activations
  <part>.amp_envelope.npz          the amplitude envelope crepe_notes uses for velocities
  <part>.crepe.activation.npz      CREPE's pitch activation (360 bins, float16) and times, kept for the audio
                                   melody cue
The f0, onset and envelope files let the note segmentation be re-run with other settings without CREPE.
Stems already transcribed are skipped, and a stem whose CREPE files exist reuses them, so a rerun resumes without
repeating CREPE. madmom reads the FLAC through ffmpeg, so ffmpeg must be on PATH (it is in the amt-tools
environment). progress.ckpt is touched after each stem (the job chain
takes a new *ckpt* file as proof of progress), and one JSON line per stem goes to stdout and <out>/stems.jsonl.

  python ../transcription/run_crepe_notes.py --samba ~/scratch/MIDI-Score-Following/data/samba --out $RUN_DIR \
      --shard 0 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PARTS = ("trumpet", "saxophone_alto", "bass")
MIN_DURATION = {"bass": 0.05}  # seconds; crepe_notes default 0.03 otherwise


def stems(samba):
    return sorted(p for part in PARTS for p in (samba / "sync_audio" / "songs").glob(f"*/*/{part}.flac"))


def shard_of(paths, k, n):
    """Largest first, dealt round-robin, so each shard gets a similar amount of audio."""
    ordered = sorted(paths, key=lambda p: -p.stat().st_size)
    return sorted(ordered[k::n])


def transcribe(stem, out_root):
    import numpy as np
    from crepe_notes.crepe_notes import parse_f0, process, save_f0

    song, take, part = stem.parent.parent.name, stem.parent.name, stem.stem
    folder = out_root / "sync" / song / take
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / stem.name
    if not link.exists():
        link.symlink_to(stem.resolve())
    midi = folder / f"{part}.crepe.transcription.mid"
    if midi.exists():
        return None
    started = time.time()
    f0_path, activation_path = folder / f"{part}.crepe.f0.csv", folder / f"{part}.crepe.activation.npz"
    if f0_path.exists() and activation_path.exists():
        frequency, confidence = parse_f0(f0_path)
    else:
        import crepe
        from librosa import load
        audio, sr = load(str(link), sr=None, mono=True)
        times, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, verbose=0)
        np.savez_compressed(activation_path, time=times.astype("float32"), activation=activation.astype("float16"))
        save_f0(f0_path, frequency, confidence)
    process(frequency, confidence, link, output_label="crepe.transcription", use_cwd=False,
            min_duration=MIN_DURATION.get(part, 0.03), save_analysis_files=True, pitch_tracker="crepe")
    return {"stem": f"sync/{song}/{take}/{part}", "audio_s": round(len(frequency) * 0.01, 1),
            "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    args = parser.parse_args()
    todo = shard_of(stems(args.samba), *args.shard)
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stems": len(todo), "shard": args.shard}), flush=True)
    for stem in todo:
        result = transcribe(stem, args.out)
        if result is None:
            continue
        line = json.dumps(result)
        print(line, flush=True)
        with open(args.out / "stems.jsonl", "a") as handle:
            handle.write(line + "\n")
        (args.out / "progress.ckpt").touch()


if __name__ == "__main__":
    main()
