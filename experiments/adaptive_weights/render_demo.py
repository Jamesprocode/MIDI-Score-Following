"""Render a demo video of the follower on one JazzSAMBA take, with the band's recording as audio.

The picture is a chord chart of one chorus. A blue dot shows where the band
really is (the annotated position); an orange dot shows where the follower
thinks it is. A strip below shows the piano notes the follower hears, and a
status line says whether the follower is in the right bar.

Positions come from the fixed-weight sweep (sweep_fixed_weights.py), which
stored the follower's estimate every 200 ms; the video interpolates between
those samples for smooth motion (display only; a jump of a beat or more is
drawn as a jump). The audio is the take's mixture.flac; the follower itself
only ever sees the piano MIDI.

  ../yamaha_timing/.venv/bin/python render_demo.py --take sync/67-on_green_dolphin_street/preferred \
      --start 30 --end 80 --weights 1 1 --out demo.mp4
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
import follower_data  # noqa: E402

BLUE, ORANGE, NAVY, GREY, LIGHT, GOLD = "#3A5DAE", "#E04F39", "#003057", "#54585A", "#D6DBD4", "#B3A369"
PRETTY = {"M7": "maj7", "-7": "m7", "Maj": "", "-": "m", "h7": "ø7", "o7": "°7", "-6": "m6", "-M7": "m(maj7)"}


def pretty_chord(label):
    for root_len in (2, 1):
        root, rest = label[:root_len], label[root_len:]
        if root_len == 2 and root[1] not in "b#":
            continue
        root = root.replace("b", "♭").replace("#", "♯")
        rest = PRETTY.get(rest, ("m" + rest[1:]) if rest.startswith("-") else rest)
        rest = rest.replace("#", "♯").replace("b", "♭")
        return root + rest
    return label


def load_take(samba_root, performance_id):
    takes = follower_data.samba_follow_takes(samba_root, protocols=(performance_id.split("/")[0],))
    data, take = next(t for t in takes if t[0]["performance_id"] == performance_id)
    protocol, song, name = performance_id.split("/")
    folder = Path(samba_root) / f"{protocol}_audio/songs/{song}/{name}"
    with open(folder / "annotations/sections.csv") as handle:
        sections = list(csv.DictReader(handle))
    return data, take, folder, sections


def chorus_layout(data, sections):
    """Bars per chorus (from the head-in sections) and the chord labels of chorus 1, by bar."""
    head = [s for s in sections if s["top_level_section"] == "head-in"] or sections
    first_bar, last_bar = int(head[0]["start_bar"]), int(head[-1]["end_bar"])
    length = last_bar - first_bar
    labels = [[] for _ in range(length)]
    for c in data["chords"]:
        b0, f0 = int(c["start_bar"]), float(c["start_fraction"])
        b1, f1 = int(c["end_bar"]), float(c["end_fraction"])
        if c["is_no_chord"] == "True" or b0 < first_bar or b0 >= last_bar:
            continue
        for b in range(b0, min(b1 + (1 if f1 > 0 else 0), last_bar)):
            text = pretty_chord(c["chord"])
            if (b == b0 and f0 == 0) or (b > b0 and not labels[b - first_bar]):
                labels[b - first_bar].append(text if b == b0 else "%")
            elif b == b0:
                labels[b - first_bar].append(text)
    labels = [[x for i, x in enumerate(l) if x != "%" or i == 0] for l in labels]
    return first_bar, length, labels


def bar_and_phase(position_beats, data):
    """Unfolded position in beats -> (annotation bar number, fraction of the bar)."""
    starts = data["bar_starts_beats"]
    i = np.clip(np.searchsorted(starts, position_beats, side="right") - 1, 0, len(starts) - 1)
    phase = (position_beats - starts[i]) / data["beats_per_bar"]
    return i + data["first_annotation_bar"], np.clip(phase, 0, .999)


def interpolate(times, values, at):
    """Linear between samples, except across jumps of a beat or more (drawn as a step)."""
    k = np.clip(np.searchsorted(times, at, side="right") - 1, 0, len(times) - 2)
    a, b = values[k], values[k + 1]
    w = np.clip((at - times[k]) / (times[k + 1] - times[k]), 0, 1)
    return np.where(np.abs(b - a) < 1.0, a + w * (b - a), a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, default=HERE.parent.parent / "Dataset" / "SAMBA dataset")
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "sweep_b7")
    parser.add_argument("--take", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0), metavar=("W_H", "W_D"))
    parser.add_argument("--label", default="computer (fixed weights)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data, take, folder, sections = load_take(args.samba, args.take)
    z = np.load(args.run / "frames" / f"{args.take.replace('/', '__')}.npz")
    g = [tuple(map(float, x)) for x in z["grid"]].index(tuple(args.weights))
    times, truth, predicted = z["times"], z["truth_position"], z["predicted_position"][g]
    first_bar, length, labels = chorus_layout(data, sections)
    per_row = 8
    rows = int(np.ceil(length / per_row))
    title = take.info.get("title") or args.take.split("/")[1].split("-", 1)[1].replace("_", " ").title()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 13})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(.03, .955, title, fontsize=22, color=NAVY, weight="bold", va="top")
    fig.text(.03, .905, "live band recording  ·  the computer hears only the piano (MIDI)", fontsize=12, color=GREY,
             va="top")
    chart = fig.add_axes([.03, .30, .94, .56])
    chart.set_xlim(0, per_row)
    chart.set_ylim(rows, 0)
    chart.axis("off")
    for b in range(length):
        r, c = divmod(b, per_row)
        chart.add_patch(Rectangle((c, r + .08), 1, .84, fill=False, ec=GREY, lw=1))
        chart.text(c + .5, r + .42, "  ".join(labels[b]) if labels[b] else "", ha="center", va="center",
                   fontsize=15 if len(labels[b]) < 2 else 12, color="black")
    band_cell = chart.add_patch(Rectangle((0, 0), 1, .84, color=BLUE, alpha=.18, lw=0))
    band_dot, = chart.plot([], [], "o", ms=13, color=BLUE, zorder=5)
    follow_dot, = chart.plot([], [], "o", ms=13, color=ORANGE, zorder=6, mec="white", mew=1.5)
    follow_note = chart.text(0, 0, "", color=ORANGE, fontsize=11, ha="center", va="bottom", weight="bold")
    fig.text(.03, .255, "●", color=BLUE, fontsize=16, va="center")
    fig.text(.05, .255, "band (where they really are)", fontsize=12, va="center")
    fig.text(.33, .255, "●", color=ORANGE, fontsize=16, va="center")
    fig.text(.35, .255, args.label, fontsize=12, va="center")
    status = fig.text(.97, .255, "", fontsize=14, ha="right", va="center", weight="bold")
    section_text = fig.text(.97, .955, "", fontsize=14, ha="right", va="top", color=NAVY)

    roll = fig.add_axes([.03, .04, .94, .17])
    notes = np.column_stack([take.onsets, take.offsets, take.pitches])
    window = 6.0
    roll.set_ylim(np.percentile(notes[:, 2], 1) - 3, np.percentile(notes[:, 2], 99) + 3)
    roll.set_yticks([])
    roll.set_xticks([])
    for s in roll.spines.values():
        s.set_color(LIGHT)
    roll.set_title("piano notes the computer hears (last 6 s)", fontsize=10, color=GREY, loc="left", pad=3)
    bars = []

    def section_at(t):
        for s in sections:
            if float(s["start_time"]) <= t < float(s["end_time"]):
                name = s["section"].replace("solos-", "").replace("head-in", "head in").replace("head-out", "head out")
                return name.replace("/", " · ").replace("saxophone_", "").replace("_", " ")
        return ""

    frame_times = np.arange(args.start, args.end, 1 / args.fps)
    truth_at = interpolate(times, truth, frame_times)
    follow_at = interpolate(times, predicted, frame_times)
    true_bar, true_phase = bar_and_phase(truth_at, data)
    est_bar, est_phase = bar_and_phase(follow_at, data)

    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "video.mp4"
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
        with writer.saving(fig, str(silent), dpi=100):
            for k, t in enumerate(frame_times):
                tb, eb = true_bar[k] - first_bar, est_bar[k] - first_bar
                t_chorus, t_cell = divmod(tb, length)
                e_chorus, e_cell = divmod(eb, length)
                r, c = divmod(t_cell, per_row)
                band_cell.set_xy((c, r + .08))
                band_dot.set_data([c + true_phase[k]], [r + .78])
                r2, c2 = divmod(e_cell, per_row)
                follow_dot.set_data([c2 + est_phase[k]], [r2 + .78])
                follow_note.set_position((c2 + est_phase[k], r2 + .68))
                follow_note.set_text("" if e_chorus == t_chorus else f"chorus {e_chorus + 1}")
                if true_bar[k] == est_bar[k]:
                    status.set_text("✓ right bar")
                    status.set_color("#2E7D32")
                else:
                    off = int(est_bar[k] - true_bar[k])
                    status.set_text(f"✗ lost: {abs(off)} bar{'s' if abs(off) > 1 else ''} {'ahead' if off > 0 else 'behind'}")
                    status.set_color(ORANGE)
                section_text.set_text(f"chorus {t_chorus + 1}  ·  {section_at(t)}")
                for artist in bars:
                    artist.remove()
                bars.clear()
                shown = notes[(notes[:, 1] > t - window) & (notes[:, 0] < t)]
                for on, off, pitch in shown:
                    bars.append(roll.add_patch(Rectangle((on, pitch - .6), min(off, t) - on, 1.2, color=ORANGE,
                                                         alpha=.8, lw=0)))
                roll.set_xlim(t - window, t + .3)
                writer.grab_frame()
        audio = folder / "mixture.flac"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-ss", f"{args.start}",
                        "-t", f"{args.end - args.start}", "-i", str(audio), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out)], check=True)
    print(args.out)


if __name__ == "__main__":
    main()
