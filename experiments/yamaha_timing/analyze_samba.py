"""Where does the timing network work on JazzSAMBA, and where does it fail?

For every SAMBA take (sync takes with the timing repair) the network runs once
with the BPM condition at the written tempo and once at the played tempo (the
median annotated tempo: what a player would announce, the paper's assumption).
Using the played-tempo runs, it then breaks the scores down:

  per take     beat/downbeat F, CMLt, plus song facts: genre, swing, played
               tempo, tempo variability (4-bar tempo range), piano note
               density, share of the take with the piano laid out
  per section  each annotated section scored on its own (mir_eval F-measure,
               no 5 s trim) and labelled by the piano's role from
               section_musicians.csv / the section name: piano solo, comping,
               or laid out
  over time    beat and downbeat F in 10 s windows (5 s hop), with the local
               annotated tempo and the piano's onset rate

and draws figures for the best, median and worst take (by downbeat F).
Writes per_take.csv, per_section.csv, summary.md and figures/ to --out.

  python analyze_samba.py --data DIR --checkpoint best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from eval_real import run_take
from yamaha_timing.lakh import SynthesisConfig
from yamaha_timing.plots import plot_tracking
from yamaha_timing.real_data import samba_takes
from yamaha_timing.training import load_checkpoint

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order
ROLE_TINT = {"piano solo": "#d7e6f7", "comping": None, "laid out": "#fbe1d6"}  # comping: unshaded


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def f_measure(reference, estimated):
    import mir_eval
    if len(reference) < 2:
        return float("nan")
    return float(mir_eval.beat.f_measure(np.asarray(reference), np.asarray(estimated)))


def within(events, start, end):
    events = np.asarray(events)
    return events[(events >= start) & (events < end)]


def sections_of(root, take):
    folder = Path(root) / f"{take.info['protocol']}_audio/songs/{take.info['song']}/{take.info['take']}/annotations"
    players = {}
    for r in rows(folder / "section_musicians.csv"):
        players.setdefault(r["section_index"], set()).add(r["instrument"])
    out = []
    for r in rows(folder / "sections.csv"):
        start, end = float(r["start_time"]), float(r["end_time"])
        if end <= start:
            continue
        role = ("piano solo" if r["section"].startswith("solos-piano")
                else "comping" if "piano" in players.get(r["section_index"], set()) else "laid out")
        out.append({"label": r["section"], "start": start, "end": end, "role": role})
    return out


def tempo_facts(take):
    bars = np.append(take.downbeats, take.span[1])
    durations = np.diff(bars)
    beats_per_bar = np.median(np.diff(bars) / np.median(np.diff(take.beats)))
    bar_bpm = 60 * beats_per_bar / durations
    smooth = np.convolve(bar_bpm, np.ones(4) / 4, "valid") if len(bar_bpm) >= 4 else bar_bpm
    return {"played_bpm": float(60 / np.median(np.diff(take.beats))),
            "tempo_range_4bar": float(smooth.max() / smooth.min() - 1),
            "bar_to_bar_change": float(np.mean(np.abs(np.diff(durations) / durations[:-1])))}, bars, bar_bpm


def downbeat_offset(example, numerator):
    """Which beat of the bar the network calls "1": for each predicted downbeat,
    the bar position (1..numerator) of the nearest true beat within half a beat.
    Returns the most common position, its share, and downbeat F if that beat
    were the annotated downbeat (the error is then only a fixed offset)."""
    beats, downbeats = np.asarray(example["true_beats"]), np.asarray(example["true_downbeats"])
    predicted = np.asarray(example["pred_downbeats"])
    if len(beats) < 8 or len(downbeats) < 2 or not len(predicted):
        return {"called_one_beat": 0, "called_one_share": float("nan"), "downbeat_f_offset_fixed": float("nan")}
    bar = np.clip(np.searchsorted(downbeats, beats + 1e-6) - 1, 0, None)
    beat_length = np.median(np.diff(beats))
    position = np.round((beats - downbeats[bar]) / beat_length).astype(int) % numerator
    nearest = np.clip(np.searchsorted(beats, predicted), 1, len(beats) - 1)
    nearest = np.where(np.abs(beats[nearest - 1] - predicted) < np.abs(beats[nearest] - predicted), nearest - 1, nearest)
    close = np.abs(beats[nearest] - predicted) < beat_length / 2
    if not close.any():
        return {"called_one_beat": 0, "called_one_share": float("nan"), "downbeat_f_offset_fixed": float("nan")}
    counts = np.bincount(position[nearest[close]], minlength=numerator)
    called = int(counts.argmax())
    return {"called_one_beat": called + 1, "called_one_share": float(counts.max() / counts.sum()),
            "downbeat_f_offset_fixed": f_measure(beats[position == called], predicted)}


def timeline(take, example, sections, bars, bar_bpm, window=10.0, hop=5.0):
    start, end = take.span
    centers, beat_f, down_f, tempo, onset_rate, role = [], [], [], [], [], []
    for t0 in np.arange(start, end - window + 1e-9, hop):
        t1 = t0 + window
        centers.append(t0 + window / 2)
        beat_f.append(f_measure(within(example["true_beats"], t0, t1), within(example["pred_beats"], t0, t1)))
        down_f.append(f_measure(within(example["true_downbeats"], t0, t1), within(example["pred_downbeats"], t0, t1)))
        i = np.clip(np.searchsorted(bars, t0 + window / 2) - 1, 0, len(bar_bpm) - 1)
        tempo.append(bar_bpm[i])
        onset_rate.append(len(within(take.onsets, t0, t1)) / window)
        mid = t0 + window / 2
        role.append(next((s["role"] for s in sections if s["start"] <= mid < s["end"]), "laid out"))
    return {k: np.asarray(v) if k != "role" else v for k, v in
            dict(t=centers, beat_f=beat_f, downbeat_f=down_f, tempo=tempo, onset_rate=onset_rate, role=role).items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def plot_timeline(take, row, line, sections, hint_bpm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]},
                             facecolor=SURFACE)
    for ax in axes:
        style(ax)
        for s in sections:
            if ROLE_TINT[s["role"]]:
                ax.axvspan(s["start"], s["end"], color=ROLE_TINT[s["role"]], lw=0, zorder=0)
    ax = axes[0]
    ax.plot(line["t"], line["beat_f"], color=BLUE, lw=1.6, label="beat F")
    ax.plot(line["t"], line["downbeat_f"], color=ORANGE, lw=1.6, label="downbeat F")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("F-measure\n(10 s windows)")
    handles = ax.get_legend_handles_labels()[0] + [
        Patch(facecolor=ROLE_TINT[r] or SURFACE, edgecolor=GRID if r == "comping" else "none",
              label=r if r == "piano solo" else f"piano {r}") for r in ("piano solo", "comping", "laid out")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=5, frameon=False,
              labelcolor=INK2)
    ax = axes[1]
    ax.plot(line["t"], line["tempo"], color=INK, lw=1.4, label="played tempo (bar average)")
    ax.axhline(hint_bpm, color=INK2, lw=1, ls="--", label=f"BPM condition {hint_bpm:.0f}")
    low, high = min(np.nanmin(line["tempo"]), hint_bpm), max(np.nanmax(line["tempo"]), hint_bpm)
    pad = max(0.05 * hint_bpm, 0.1 * (high - low))  # a steady click stays flat, not magnified jitter
    ax.set_ylim(low - pad, high + pad)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_ylabel("tempo\n(quarter BPM)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=2, frameon=False, labelcolor=INK2)
    ax = axes[2]
    ax.plot(line["t"], line["onset_rate"], color=INK2, lw=1.2)
    ax.set_ylabel("piano notes\nper second")
    ax.set_xlabel("seconds")
    fig.suptitle(f"{take.name} ({take.info['protocol']}, {row['genre']}, {row['played_bpm']:.0f} BPM played)   "
                 f"beat F {row['beat_f_measure']:.2f}   downbeat F {row['downbeat_f_measure']:.2f}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(per_take, per_section, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1.2, 0.8]})
    for ax in axes:
        style(ax)
    genres = sorted({r["genre"] for r in per_take}, key=lambda g: np.mean([r["downbeat_f_measure"] for r in per_take
                                                                            if r["genre"] == g]))
    ax = axes[0]
    for k, (key, color, label) in enumerate((("beat_f_measure", BLUE, "beat F"), ("downbeat_f_measure", ORANGE, "downbeat F"))):
        means = [np.mean([r[key] for r in per_take if r["genre"] == g]) for g in genres]
        ax.scatter(means, np.arange(len(genres)) + (k - .5) * .18, s=40, color=color, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(range(len(genres)), [f"{g} ({sum(r['genre'] == g for r in per_take)})" for g in genres])
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("mean F-measure per take (takes in brackets)")
    ax.set_title("By style", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK2)
    ax = axes[1]
    for protocol, color in (("async", BLUE), ("sync", ORANGE)):
        sel = [r for r in per_take if r["protocol"] == protocol]
        ax.scatter([r["played_bpm"] for r in sel], [r["downbeat_f_measure"] for r in sel], s=26, color=color,
                   edgecolors=SURFACE, linewidths=1, label=f"{protocol} ({len(sel)})", zorder=3)
    ax.set_xlabel("played tempo (quarter BPM)")
    ax.set_ylabel("downbeat F per take")
    ax.set_ylim(0, 1)
    ax.set_title("By tempo", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax = axes[2]
    roles = ["piano solo", "comping", "laid out"]
    for k, (key, color, label) in enumerate((("beat_f", BLUE, "beat F"), ("downbeat_f", ORANGE, "downbeat F"))):
        values = []
        for role in roles:
            sel = [s for s in per_section if s["role"] == role and np.isfinite(s[key])]
            weights = np.array([s["seconds"] for s in sel])
            values.append(float(np.average([s[key] for s in sel], weights=weights)) if sel else np.nan)
        ax.bar(np.arange(3) + (k - .5) * .36, values, width=.34, color=color, label=label, zorder=3)
    ax.set_xticks(range(3), roles)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F-measure (time-weighted)")
    ax.set_title("By what the piano is doing", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    result = spearmanr(np.asarray(x)[ok], np.asarray(y)[ok])
    return float(result.correlation), float(result.pvalue)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/analyze_samba")))
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, "cpu")[0].to(device).eval()
    max_note = SynthesisConfig().max_note_seconds
    catalog = {r["song_subdir_basename"]: r for r in rows(args.data / "JazzSAMBA-catalog/songs.csv")}
    takes = [t for t in samba_takes(args.data) if not t.info.get("excluded")]
    if args.limit:
        takes = takes[:args.limit // 2] + takes[-(args.limit - args.limit // 2):]  # some async, some sync
    per_take, per_section, lines, examples = [], [], {}, {}
    for take in takes:
        song = catalog[take.info["song"]]
        facts, bars, bar_bpm = tempo_facts(take)
        sections = sections_of(args.data, take)
        span = take.span[1] - take.span[0]
        row = {"name": take.name, "protocol": take.info["protocol"], "genre": song["genre"],
               "swung": song["is_swung"] == "True", "meter": song["time_signature"], "written_bpm": float(song["bpm"]),
               **facts, "played_over_written": facts["played_bpm"] / float(song["bpm"]),
               "piano_notes_per_s": float(len(within(take.onsets, *take.span)) / span),
               "laid_out_share": float(sum(min(s["end"], take.span[1]) - max(s["start"], take.span[0])
                                            for s in sections if s["role"] == "laid out") / span)}
        written, _ = run_take(model, take, take.bpm, device, max_note)
        played, example = run_take(model, take, facts["played_bpm"], device, max_note)
        row.update({f"written_{k}": written[k] for k in ("beat_f_measure", "downbeat_f_measure", "beat_cmlt",
                                                         "downbeat_cmlt")})
        row.update({k: played[k] for k in ("beat_f_measure", "downbeat_f_measure", "beat_cmlt", "downbeat_cmlt",
                                           "beat_amlt", "bar_acc")})
        row.update(downbeat_offset(example, int(song["time_signature_numerator"])))
        per_take.append(row)
        for s in sections:
            start, end = max(s["start"], take.span[0]), min(s["end"], take.span[1])
            if end - start < 4:
                continue
            per_section.append({"name": take.name, "protocol": row["protocol"], "genre": row["genre"],
                                "section": s["label"], "role": s["role"], "seconds": end - start,
                                "beat_f": f_measure(within(example["true_beats"], start, end),
                                                    within(example["pred_beats"], start, end)),
                                "downbeat_f": f_measure(within(example["true_downbeats"], start, end),
                                                        within(example["pred_downbeats"], start, end))})
        lines[take.name] = (timeline(take, example, sections, bars, bar_bpm), sections)
        examples[take.name] = (take, example)
        print(f"{take.name}: beat F {row['beat_f_measure']:.2f} downbeat F {row['downbeat_f_measure']:.2f}", flush=True)

    for name, table in (("per_take.csv", per_take), ("per_section.csv", per_section)):
        with open(args.out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    ranked = sorted(per_take, key=lambda r: r["downbeat_f_measure"])
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
    for label, row in picks.items():
        take, example = examples[row["name"]]
        line, sections = lines[row["name"]]
        stem = f"{label}__{row['name'].replace('/', '__')}"
        plot_timeline(take, row, line, sections, row["played_bpm"], args.out / "figures" / f"{stem}__timeline.png")
        example["title"] = f"{label}: {row['name']}  ({row['genre']}, played {row['played_bpm']:.0f} BPM, condition = played tempo)"
        plot_tracking(example, args.out / "figures" / f"{stem}__detail.png", model.config.frame_rate,
                      skip_seconds=take.span[0] + 60.0)
    plot_overview(per_take, per_section, args.out / "figures" / "overview.png")

    mean = lambda sel, k: float(np.mean([r[k] for r in sel]))
    lines_md = ["# Timing network on JazzSAMBA: breakdown", "",
                "BPM condition at the played tempo unless marked; mir_eval F-measure (±70 ms).", "",
                "| subset | takes | beat F | downbeat F | beat CMLt | downbeat CMLt | beat F, written tempo | downbeat F, written tempo |",
                "|---|---|---|---|---|---|---|---|"]
    for subset, sel in (("all", per_take), ("async (click)", [r for r in per_take if r["protocol"] == "async"]),
                        ("sync (live)", [r for r in per_take if r["protocol"] == "sync"])):
        lines_md.append(f"| {subset} | {len(sel)} | {mean(sel, 'beat_f_measure'):.3f} | {mean(sel, 'downbeat_f_measure'):.3f} | "
                        f"{mean(sel, 'beat_cmlt'):.3f} | {mean(sel, 'downbeat_cmlt'):.3f} | "
                        f"{mean(sel, 'written_beat_f_measure'):.3f} | {mean(sel, 'written_downbeat_f_measure'):.3f} |")
    lines_md += ["", "## By style", "", "| genre | takes | beat F | downbeat F | beat CMLt |", "|---|---|---|---|---|"]
    for genre in sorted({r["genre"] for r in per_take}, key=lambda g: -mean([r for r in per_take if r["genre"] == g],
                                                                           "downbeat_f_measure")):
        sel = [r for r in per_take if r["genre"] == genre]
        lines_md.append(f"| {genre} | {len(sel)} | {mean(sel, 'beat_f_measure'):.3f} | {mean(sel, 'downbeat_f_measure'):.3f} | "
                        f"{mean(sel, 'beat_cmlt'):.3f} |")
    lines_md += ["", "## By what the piano is doing (sections, time-weighted)", "",
                 "| piano | sections | minutes | beat F | downbeat F |", "|---|---|---|---|---|"]
    for role in ("piano solo", "comping", "laid out"):
        sel = [s for s in per_section if s["role"] == role]
        if not sel:
            continue
        w = np.array([s["seconds"] for s in sel])
        avg = lambda k: np.average([s[k] if np.isfinite(s[k]) else 0 for s in sel], weights=w)
        lines_md.append(f"| {role} | {len(sel)} | {w.sum() / 60:.0f} | {avg('beat_f'):.3f} | {avg('downbeat_f'):.3f} |")
    lines_md += ["", "## Is the downbeat wrong, or just on the wrong beat?", "",
                 "For each take: the beat of the bar the network most often calls \"1\", and downbeat F if that",
                 "offset were the only error (predicted downbeats scored against that beat of every bar).", "",
                 "| subset | takes | calls beat 1 | calls another beat | downbeat F | downbeat F, offset fixed |",
                 "|---|---|---|---|---|---|"]
    for subset, sel in (("all", per_take), ("async (click)", [r for r in per_take if r["protocol"] == "async"]),
                        ("sync (live)", [r for r in per_take if r["protocol"] == "sync"])):
        ok = [r for r in sel if r["called_one_beat"]]
        lines_md.append(f"| {subset} | {len(ok)} | {sum(r['called_one_beat'] == 1 for r in ok)} | "
                        f"{sum(r['called_one_beat'] > 1 for r in ok)} | {mean(ok, 'downbeat_f_measure'):.3f} | "
                        f"{float(np.mean([max(r['downbeat_f_measure'], r['downbeat_f_offset_fixed']) for r in ok])):.3f} |")
    lines_md += ["", "## What goes with a good downbeat F (Spearman, per take)", "",
                 "Pooled over both protocols; live takes score higher and vary more in tempo, so check within protocol.", "",
                 "| take property | rho | p | rho async | p | rho sync | p |", "|---|---|---|---|---|---|---|"]
    y = np.array([r["downbeat_f_measure"] for r in per_take])
    for key in ("played_bpm", "tempo_range_4bar", "bar_to_bar_change", "piano_notes_per_s", "laid_out_share", "swung"):
        rho, p = spearman(np.array([float(r[key]) for r in per_take]), y)
        cells = [f"{rho:+.2f}", f"{p:.3f}"]
        for protocol in ("async", "sync"):
            sel = [r for r in per_take if r["protocol"] == protocol]
            x = np.array([float(r[key]) for r in sel])
            if np.std(x) == 0:
                cells += ["-", "-"]
                continue
            rho_p, p_p = spearman(x, np.array([r["downbeat_f_measure"] for r in sel]))
            cells += [f"{rho_p:+.2f}", f"{p_p:.3f}"]
        lines_md.append(f"| {key} | " + " | ".join(cells) + " |")
    lines_md += ["", "## Best and worst takes (downbeat F)", "",
                 "| take | genre | played BPM | 4-bar tempo range | piano notes/s | laid out | beat F | downbeat F | calls as \"1\" |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for row in ranked[-5:][::-1] + [None] + ranked[:5]:
        if row is None:
            lines_md.append("| ... | | | | | | | | |")
            continue
        lines_md.append(f"| {row['name']} | {row['genre']} | {row['played_bpm']:.0f} | {100 * row['tempo_range_4bar']:.0f}% | "
                        f"{row['piano_notes_per_s']:.1f} | {100 * row['laid_out_share']:.0f}% | "
                        f"{row['beat_f_measure']:.2f} | {row['downbeat_f_measure']:.2f} | "
                        f"beat {row['called_one_beat']} ({100 * row['called_one_share']:.0f}%) |")
    lines_md += ["", f"Figures: best `{picks['best']['name']}`, median `{picks['median']['name']}`, "
                     f"worst `{picks['worst']['name']}` (timeline and 30 s detail each), and overview.png."]
    (args.out / "summary.md").write_text("\n".join(lines_md) + "\n")
    print("\n".join(lines_md), flush=True)
    (args.out / "DONE").touch()


if __name__ == "__main__":
    main()
