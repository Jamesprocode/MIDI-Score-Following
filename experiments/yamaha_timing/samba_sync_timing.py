"""Fit the time shift of each JazzSAMBA sync piano MIDI from its own piano audio.

Sync piano MIDI ticks follow a 175 BPM session tempo while each file's tempo
event holds the song's written tempo (see yamaha_timing/real_data.py), and the
MIDI starts a few seconds off the audio and annotations. The MIDI piano was
recorded together with its microphones (piano_left/right.flac), so the shift is
found without annotations: cross-correlate an onset envelope of the piano
audio with the rescaled MIDI onsets (100 Hz, lags within +-max_shift). Async
takes, whose MIDI is aligned as recorded, go through the same procedure with
their file timing: their common fitted shift is the envelope's own lag (the
spectral-flux frames are stamped at the window start), and it is subtracted
from every sync shift (column bias_s).

Columns: confidence = correlation peak over the best peak at least 0.2 s away;
concentration_* = onset concentration on the annotated beat grid (0 = spread
evenly over the beat, 1 = all at one point) with file timing and after the fix.
Needs ffmpeg on PATH. Writes samba_sync_timing.csv next to this script.

  python samba_sync_timing.py "../../Dataset/SAMBA dataset"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import correlate

from yamaha_timing.real_data import SESSION_BPM, beat_concentration, midi_tempo_bpm, read_midi_notes

RATE = 100      # envelope frames per second
SR = 12000      # decoding sample rate; hop = SR / RATE


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def onset_envelope(signal, n_fft=1024):
    """Half-wave rectified log-magnitude spectral flux, locally mean-removed."""
    hop = SR // RATE
    frames = 1 + (len(signal) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    flux = np.zeros(frames, dtype=np.float32)
    previous = None
    for start in range(0, frames, 4096):  # chunks keep memory small
        idx = np.arange(start, min(frames, start + 4096))[:, None] * hop + np.arange(n_fft)
        spectrum = np.log1p(100 * np.abs(np.fft.rfft(signal[idx] * window, axis=1)))
        if previous is not None:
            spectrum = np.vstack([previous, spectrum])
        diff = np.maximum(np.diff(spectrum, axis=0), 0).sum(axis=1)
        flux[start + (0 if previous is not None else 1): start + len(idx)] = diff
        previous = spectrum[-1:]
    local = np.convolve(flux, np.ones(RATE // 2) / (RATE // 2), mode="same")
    return np.maximum(flux - local, 0)


def onset_train(onsets, length):
    train = np.zeros(length, dtype=np.float32)
    index = np.round(onsets * RATE).astype(int)
    np.add.at(train, index[(index >= 0) & (index < length)], 1.0)
    kernel = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)  # 20 ms Gaussian
    return np.convolve(train, kernel, mode="same")


def fit_shift(envelope, onsets, max_shift):
    train = onset_train(onsets, len(envelope))
    full = correlate(envelope, train, mode="full", method="fft")
    lags = np.arange(len(full)) - (len(train) - 1)
    keep = np.abs(lags) <= max_shift * RATE
    lags, values = lags[keep], full[keep]
    best = values.argmax()
    rival = values[np.abs(lags - lags[best]) >= 0.2 * RATE].max()
    return lags[best] / RATE, float(values[best] / max(rival, 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="JazzSAMBA root (JazzSAMBA-catalog, *_audio)")
    parser.add_argument("--max-shift", type=float, default=30.0)
    parser.add_argument("--async-checks", type=int, default=10, help="async takes run as a sanity check")
    args = parser.parse_args()
    catalog = {r["song_subdir_basename"]: r for r in csv.DictReader(open(args.root / "JazzSAMBA-catalog/songs.csv"))}
    jobs = [("sync", p) for p in sorted(args.root.glob("sync_audio/songs/*/*/annotations/bars.csv"))]
    jobs += [("async", p) for p in sorted(args.root.glob("async_audio/songs/*/preferred/annotations/bars.csv"))
             ][:args.async_checks]
    rows, grids = [], {}
    for protocol, bars_csv in jobs:
        take = bars_csv.parent.parent
        numerator = int(catalog[take.parent.name]["time_signature_numerator"])
        bars = list(csv.DictReader(open(bars_csv)))
        starts = np.array([float(r["start_time"]) for r in bars])
        ends = np.array([float(r["end_time"]) for r in bars])
        grid = np.sort(np.append(np.concatenate([starts + k * (ends - starts) / numerator
                                                 for k in range(numerator)]), ends[-1]))
        grids[take] = grid
        onsets, _, _ = read_midi_notes(take / "piano.mid")
        file_bpm = midi_tempo_bpm(take / "piano.mid")
        scale = file_bpm / SESSION_BPM if protocol == "sync" else 1.0
        audio = sum(decode(take / f"piano_{side}.flac") for side in ("left", "right"))
        shift, confidence = fit_shift(onset_envelope(audio), onsets * scale, args.max_shift)
        row = {"protocol": protocol, "song": take.parent.name, "take": take.name, "file_bpm": round(file_bpm, 3),
               "scale": round(scale, 6), "shift_s": round(shift, 3), "confidence": round(confidence, 3),
               "concentration_file": round(beat_concentration(onsets, grid), 3),
               "concentration_fixed": round(beat_concentration(onsets * scale + shift, grid), 3)}
        rows.append(row)
        print(row, flush=True)
    bias = float(np.median([r["shift_s"] for r in rows if r["protocol"] == "async"]))
    for row in rows:  # remove the envelope lag measured on the aligned async takes
        take = args.root / f"{row['protocol']}_audio/songs/{row['song']}/{row['take']}"
        row["bias_s"], row["shift_s"] = bias, round(row["shift_s"] - bias, 3)
        onsets, _, _ = read_midi_notes(take / "piano.mid")
        row["concentration_fixed"] = round(beat_concentration(onsets * row["scale"] + row["shift_s"], grids[take]), 3)
    out = Path(__file__).with_name("samba_sync_timing.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
