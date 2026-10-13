"""Read-only survey of a random Lakh (LMD-full) sample for beat-label quality.

Usage (from this folder): python lakh_survey.py ../../Dataset/lmd_full results 2000 0
Works in MIDI ticks so beat positions are exact. Writes per-file rows (JSONL),
a summary (JSON), and a few example piano-roll plots against the beat grid.
"""
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import mido
import numpy as np

SNARES = {38, 40}
FX_PROGRAMS = range(120, 128)
FAMILIES = ["piano", "chrom_perc", "organ", "guitar", "bass", "strings", "ensemble", "brass",
            "reed", "pipe", "synth_lead", "synth_pad", "synth_fx", "ethnic", "percussive", "sound_fx"]
GRID_TOL = 0.02  # beats


def read(path):
    mid = mido.MidiFile(path, clip=True)
    tpb = mid.ticks_per_beat
    if mid.type == 2 or tpb <= 0:
        raise ValueError(f"unsupported type {mid.type} / tpb {tpb}")
    notes = []          # (on_tick, off_tick, pitch, part, is_drum, program)
    tempos, sigs, lyrics = [], [], 0
    for ti, track in enumerate(mid.tracks):
        tick = 0
        program = {}
        open_notes = {}
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempos.append((tick, msg.tempo))
                elif msg.type == "time_signature":
                    sigs.append((tick, msg.numerator, msg.denominator))
                elif msg.type == "lyrics":
                    lyrics += 1
                continue
            if msg.type == "program_change":
                program.setdefault(msg.channel, msg.program)
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault((msg.channel, msg.note), []).append(tick)
            elif msg.type in ("note_off", "note_on"):
                starts = open_notes.get((msg.channel, msg.note))
                if starts:
                    on = starts.pop(0)
                    ch = msg.channel
                    notes.append((on, max(tick, on + 1), msg.note, (ti, ch), ch == 9, program.get(ch, 0)))
    return tpb, notes, sorted(tempos), sorted(sigs), lyrics


def seconds(ticks, tpb, tempos):
    points = [(0, 500000)] + [t for t in tempos if t[0] > 0] if not tempos or tempos[0][0] > 0 else tempos
    ticks = np.asarray(ticks, dtype=np.float64)
    out = np.zeros_like(ticks)
    acc = 0.0
    for i, (t0, us) in enumerate(points):
        t1 = points[i + 1][0] if i + 1 < len(points) else np.inf
        mask = (ticks >= t0) & (ticks < t1)
        out[mask] = acc + (ticks[mask] - t0) * us / 1e6 / tpb
        if np.isfinite(t1):
            acc += (t1 - t0) * us / 1e6 / tpb
    return out


def survey(path):
    row = {"file": str(path)}
    try:
        tpb, notes, tempos, sigs, lyrics = read(path)
    except Exception as exc:  # corrupt or unsupported
        row["error"] = type(exc).__name__
        return row
    pitched = [n for n in notes if not n[4] and n[5] not in FX_PROGRAMS]
    drums = [n for n in notes if n[4]]
    parts = {n[3]: (n[4], n[5]) for n in notes}
    last_tick = max((n[1] for n in notes), default=0)
    row.update(
        tpb=tpb, lyrics=lyrics, n_notes=len(notes), n_pitched_notes=len(pitched),
        n_parts=len(parts), n_drum_parts=sum(d for d, _ in parts.values()),
        n_fx_parts=sum((not d) and p in FX_PROGRAMS for d, p in parts.values()),
        families=sorted({FAMILIES[p // 8] for d, p in parts.values() if not d}),
        duration_s=float(seconds([last_tick], tpb, tempos)[0]) if notes else 0.0,
        n_tempo_events=len(tempos), n_distinct_tempi=len({t for _, t in tempos}),
        first_bpm=round(60e6 / tempos[0][1], 2) if tempos else None,
        n_ts_events=len(sigs), ts=sorted({f"{n}/{d}" for _, n, d in sigs}),
        first_ts_tick=sigs[0][0] if sigs else None,
    )
    row["constant_44"] = bool(sigs) and row["ts"] == ["4/4"]
    if pitched:
        beats = np.array([n[0] for n in pitched], dtype=np.float64) / tpb
        frac = beats - np.floor(beats)
        d16 = np.abs(frac * 4 - np.round(frac * 4)) / 4
        d3 = np.abs(frac * 3 - np.round(frac * 3)) / 3
        row["on_grid"] = float(np.mean(np.minimum(d16, d3) <= GRID_TOL))
    else:
        row["on_grid"] = None
    snare = np.array([n[0] for n in drums if n[2] in SNARES], dtype=np.float64) / tpb
    near = snare[np.abs(snare - np.round(snare)) <= 0.1]
    if row["constant_44"] and len(near) >= 16:
        counts = np.bincount(np.round(near).astype(int) % 4, minlength=4)
        share24 = float((counts[1] + counts[3]) / counts.sum())
        row["snare_counts_by_beat"] = counts.tolist()
        row["snare_on_2_4"] = share24
        row["drum_check"] = "backbeat_2_4" if share24 >= 0.65 else "on_1_3_shifted" if share24 <= 0.35 else "unclear"
    else:
        row["drum_check"] = "no_check" if not row["constant_44"] else "too_few_snares"
    return row


def funnel(rows):
    steps = []
    alive = rows
    steps.append(("sampled", len(alive)))
    alive = [r for r in alive if "error" not in r]
    steps.append(("opens without error", len(alive)))
    alive = [r for r in alive if r["duration_s"] >= 30 and r["n_pitched_notes"] >= 50]
    steps.append(("≥30 s and ≥50 pitched notes", len(alive)))
    alive = [r for r in alive if r["constant_44"]]
    steps.append(("explicit, constant 4/4", len(alive)))
    alive = [r for r in alive if r["on_grid"] is not None and r["on_grid"] >= 0.9]
    steps.append(("≥90% of note starts on 16th/triplet grid", len(alive)))
    alive = [r for r in alive if r["drum_check"] != "on_1_3_shifted"]
    steps.append(("snare not on beats 1 & 3 (not shifted)", len(alive)))
    return steps, alive


def plot_example(path, out_png, title, bars=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tpb, notes, tempos, sigs, _ = read(path)
    pitched = [n for n in notes if not n[4] and n[5] not in FX_PROGRAMS]
    drums = [n for n in notes if n[4]]
    first = min(n[0] for n in pitched) / tpb
    start = 4 * np.floor(first / 4)
    end = start + 4 * bars
    parts = sorted({n[3] for n in pitched})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(parts), 1)))
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
    for n in pitched:
        b0, b1 = n[0] / tpb, n[1] / tpb
        if b1 < start or b0 > end:
            continue
        ax.add_patch(plt.Rectangle((b0, n[2] - 0.4), b1 - b0, 0.8,
                                   color=colors[parts.index(n[3])], alpha=0.8, lw=0))
    for n in drums:
        b0 = n[0] / tpb
        if start <= b0 <= end:
            y = 1 if n[2] in SNARES else 0 if n[2] in (35, 36) else 0.5
            axd.plot(b0, y, "k|" if y == 0.5 else ("r|" if y == 1 else "b|"), ms=10)
    for b in np.arange(start, end + 1):
        for a in (ax, axd):
            a.axvline(b, color="0.3" if b % 4 == 0 else "0.85", lw=1.4 if b % 4 == 0 else 0.6, zorder=0)
    inside = [n[2] for n in pitched if start <= n[0] / tpb <= end]
    ax.set_ylim(min(inside) - 2, max(inside) + 2)
    ax.set_xlim(start, end)
    ax.set_ylabel("MIDI pitch")
    ax.set_title(title, fontsize=10, loc="left")
    axd.set_yticks([0, 0.5, 1], ["kick", "other", "snare"])
    axd.set_ylim(-0.5, 1.5)
    axd.set_xlabel("beats from the file's tempo map  (dark lines = downbeats, light = beats)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*/*.mid"))
    sample = random.Random(seed).sample(files, n)
    with Pool(8) as pool:
        rows = pool.map(survey, sample, chunksize=16)
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    steps, kept = funnel(rows)
    on_grid = np.array([r["on_grid"] for r in ok if r["on_grid"] is not None])
    fam = Counter(f for r in ok for f in r["families"])
    summary = {
        "total_files_in_folder": len(files), "sample": n, "seed": seed,
        "errors": dict(Counter(r["error"] for r in rows if "error" in r)),
        "funnel": steps,
        "duration_min_median": round(float(np.median([r["duration_s"] for r in ok])) / 60, 2),
        "parts_median": float(np.median([r["n_parts"] for r in ok])),
        "has_drums": round(np.mean([r["n_drum_parts"] > 0 for r in ok]), 3),
        "has_piano_family": round(fam["piano"] / len(ok), 3),
        "family_share": {k: round(v / len(ok), 3) for k, v in fam.most_common()},
        "has_lyrics": round(np.mean([r["lyrics"] > 0 for r in ok]), 3),
        "time_signature": {
            "none": round(np.mean([r["n_ts_events"] == 0 for r in ok]), 3),
            "constant_4_4": round(np.mean([r["constant_44"] for r in ok]), 3),
            "other_constant": round(np.mean([len(r["ts"]) == 1 and r["ts"] != ["4/4"] for r in ok]), 3),
            "changes": round(np.mean([len(r["ts"]) > 1 for r in ok]), 3),
            "top": Counter(t for r in ok for t in r["ts"]).most_common(6),
        },
        "tempo": {
            "single_tempo": round(np.mean([r["n_distinct_tempi"] <= 1 for r in ok]), 3),
            "exactly_120_only": round(np.mean([r["n_distinct_tempi"] <= 1 and r["first_bpm"] in (None, 120.0) for r in ok]), 3),
            "first_bpm_quartiles": np.percentile([r["first_bpm"] for r in ok if r["first_bpm"]], [10, 25, 50, 75, 90]).round(1).tolist(),
        },
        "on_grid_deciles": np.percentile(on_grid, [10, 20, 30, 40, 50, 60, 70, 80, 90]).round(3).tolist(),
        "on_grid_share_ge_0.9": round(float(np.mean(on_grid >= 0.9)), 3),
        "on_grid_share_lt_0.5": round(float(np.mean(on_grid < 0.5)), 3),
        "drum_check_among_constant_44": dict(Counter(r["drum_check"] for r in ok if r["constant_44"])),
        "ticks_per_beat_top": Counter(r["tpb"] for r in ok).most_common(5),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Examples: a clean file, a played-in file, and a shifted-bar file.
    rng = random.Random(seed)
    picks = {
        "clean": [r for r in kept if r["drum_check"] == "backbeat_2_4"],
        "played_in": [r for r in ok if r["constant_44"] and r["on_grid"] is not None and r["on_grid"] < 0.4
                      and r["n_pitched_notes"] > 300],
        "shifted": [r for r in ok if r["drum_check"] == "on_1_3_shifted" and (r["on_grid"] or 0) >= 0.9],
    }
    captions = {
        "clean": "Passes every check: notes sit on the grid, snare (red) on beats 2 and 4",
        "played_in": "Played in without a click: notes drift off the tempo-map grid, so beat labels would be wrong",
        "shifted": "On the grid, but the snare lands on beats 1 and 3: bars likely shifted by one beat",
    }
    for kind, pool_rows in picks.items():
        if not pool_rows:
            print("no example for", kind)
            continue
        r = rng.choice(pool_rows)
        png = out / f"example_{kind}.png"
        plot_example(r["file"], png, f"{captions[kind]}\n{Path(r['file']).name}  "
                     f"(on-grid {r['on_grid']:.2f}, tempo {r['first_bpm']} BPM)")
        print("wrote", png, "from", r["file"], "candidates:", len(pool_rows))


if __name__ == "__main__":
    main()
