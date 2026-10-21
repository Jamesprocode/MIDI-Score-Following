"""Render a demo video of the follower on one JazzSAMBA take, with the band's recording as audio.

The picture matches the demo page (demo_player/): one chorus of the lead sheet
(section letters, chords; a chord carried over from the previous bar is grey),
a roadmap of the performance (head, solos, head out), a blue dot where the band
really is (the annotated position) and an orange ring where the follower thinks
it is, tagged with its chorus when that differs. A strip below shows the piano
notes the follower hears, and a status line says whether it is in the right bar.

Follower positions come either from the fixed-weight sweep (--run, any weight
setting; sweep_fixed_weights.py) or from a demo page data file (--demo-json with
--follower). Both hold the estimate every 200 ms; the video interpolates between
those samples for smooth motion (display only; a jump of a beat or more is drawn
as a jump). The audio is the take's mixture.flac; the follower itself only ever
sees the piano MIDI.

  ../yamaha_timing/.venv/bin/python render_demo.py --take sync/67-on_green_dolphin_street/preferred \\
      --start 30 --end 80 --weights 1 1 --out demo.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
import follower_data  # noqa: E402

BLUE, ORANGE, NAVY, GREY, LIGHT, BAND_FILL = "#3A5DAE", "#E04F39", "#003057", "#54585A", "#D6DBD4", "#D9EAFB"
PRETTY = {"M7": "maj7", "-7": "m7", "Maj": "", "-": "m", "h7": "ø7", "o7": "°7", "o": "°", "-6": "m6",
          "-M7": "m(maj7)", "M9": "maj9", "-9": "m9", "M6": "6"}


def pretty_chord(label):
    if label in ("N", "NC", ""):
        return "N.C."
    root, rest = (label[:2], label[2:]) if len(label) > 1 and label[1] in "b#" else (label[:1], label[1:])
    slash = ""
    if "/" in rest:
        rest, slash = rest.split("/", 1)
        slash = "/" + slash.replace("#", "♯").replace("b", "♭")
    root = root.replace("b", "♭").replace("#", "♯")
    rest = PRETTY.get(rest, ("m" + rest[1:]) if rest.startswith("-") else rest)
    return root + rest.replace("#", "♯").replace("b", "♭") + slash


def group_name(name):
    if name == "head-in":
        return "Head in"
    if name == "head-out":
        return "Head out"
    if name.startswith("solos-"):
        who = name[6:].replace("saxophone_alto", "alto sax").replace("saxophone_tenor", "tenor sax").replace("_", " ")
        return who[:1].upper() + who[1:] + " solo"
    return name[:1].upper() + name[1:].replace("_", " ")


def load_take(samba_root, performance_id):
    takes = follower_data.samba_follow_takes(samba_root, protocols=(performance_id.split("/")[0],))
    data, take = next(t for t in takes if t[0]["performance_id"] == performance_id)
    protocol, song, name = performance_id.split("/")
    folder = Path(samba_root) / f"{protocol}_audio/songs/{song}/{name}"
    with open(folder / "annotations/sections.csv") as handle:
        sections = list(csv.DictReader(handle))
    return data, take, folder, sections


def form_layout(data, sections):
    """The written form from bars.csv measure numbers: chords and section letters per measure,
    and per annotation bar its measure and chorus."""
    bars = [b for b in data["bars"] if int(b["measure"]) > 0]
    length = max(int(b["measure"]) for b in bars)
    intro = set()  # bars of intro/outro sections: shown on the chart, but not counted as a chorus
    for sec in sections:
        if sec["is_intro"] == "True" or sec["is_outro"] == "True":
            intro.update(range(int(sec["start_bar"]), int(sec["end_bar"])))
    measure_of, chorus_of, first_bar = {}, {}, {}
    chorus, previous = 0, float("inf")
    for b in bars:
        bar, measure = int(b["bar"]), int(b["measure"])
        measure_of[bar] = measure
        if bar in intro:
            chorus_of[bar] = 0
            continue
        first_bar.setdefault(measure, bar)
        if measure <= previous:
            chorus += 1
        previous = measure
        chorus_of[bar] = chorus
    for b in bars:  # a measure that only the intro plays
        first_bar.setdefault(int(b["measure"]), int(b["bar"]))
    chords = []
    for c in data["chords"]:
        if c["is_no_chord"] == "True":
            continue
        chords.append((int(c["start_bar"]) + float(c["start_fraction"]), int(c["end_bar"]) + float(c["end_fraction"]),
                       pretty_chord(c["chord"])))
    labels = {}
    for m, bar in first_bar.items():
        labels[m] = [(max(0.0, s - bar), s < bar - 1e-6, text) for s, e, text in chords if s < bar + 1 - 1e-6 and e > bar + 1e-6]
    letters = {}
    for s in sections:
        if s["is_head"] == "True" or s["is_solo"] == "True":
            measure = measure_of.get(int(s["start_bar"]))
            if measure is not None:
                letters.setdefault(measure, s["bottom_level_section"])
    groups = []
    for s in sections:
        name = s["section"].split("/")[0]
        if groups and groups[-1]["name"] == name:
            groups[-1]["end"] = float(s["end_time"])
        else:
            groups.append({"name": name, "start": float(s["start_time"]), "end": float(s["end_time"])})
    return length, labels, letters, measure_of, chorus_of, chorus, groups


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


def follower_track(args):
    if args.demo_json:
        payload = json.loads(Path(args.demo_json).read_text())
        samples = payload["followers"][args.follower]["samples"]
        return (np.array([s["time"] for s in samples]), np.array([s["truthPositionBeats"] for s in samples]),
                np.array([s["estimatedPositionBeats"] for s in samples]))
    z = np.load(args.run / "frames" / f"{args.take.replace('/', '__')}.npz")
    g = [tuple(map(float, x)) for x in z["grid"]].index(tuple(args.weights))
    return z["times"], z["truth_position"], z["predicted_position"][g]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, default=HERE.parent.parent / "Dataset" / "SAMBA dataset")
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "sweep_b7")
    parser.add_argument("--demo-json", type=Path, default=None, help="demo_player/<take>.json instead of --run")
    parser.add_argument("--follower", default="fixed", help="follower id in --demo-json")
    parser.add_argument("--take", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0), metavar=("W_H", "W_D"))
    parser.add_argument("--label", default="computer (fixed weights)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data, take, folder, sections = load_take(args.samba, args.take)
    times, truth, predicted = follower_track(args)
    length, labels, letters, measure_of, chorus_of, total, groups = form_layout(data, sections)
    per_row = 8 if length >= 24 or length % 8 == 0 else 4
    rows = int(np.ceil(length / per_row))
    title = args.title or args.take.split("/")[1].split("-", 1)[1].replace("_", " ").title()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 13})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(.03, .965, title, fontsize=22, color=NAVY, weight="bold", va="top")
    fig.text(.03, .918, "live band recording  ·  the computer hears only the piano (MIDI)", fontsize=12, color=GREY,
             va="top")
    counter = fig.text(.97, .965, "", fontsize=13, ha="right", va="top", color=NAVY, weight="bold")
    counter_f = fig.text(.97, .925, "", fontsize=12, ha="right", va="top", color=ORANGE)

    road = fig.add_axes([.03, .835, .94, .05])
    road.set_xlim(0, 1)
    road.set_ylim(0, 1)
    road.axis("off")
    chips, width = [], 1 / max(len(groups), 1)
    for k, g in enumerate(groups):
        box = Rectangle((k * width + .004, .1), width - .008, .8, fc="#F3F6F9", ec=LIGHT, lw=1)
        road.add_patch(box)
        road.text(k * width + width / 2, .5, group_name(g["name"]), ha="center", va="center", fontsize=10, color=NAVY)
        chips.append(box)

    chart = fig.add_axes([.03, .30, .94, .51])
    chart.set_xlim(0, per_row)
    chart.set_ylim(rows, 0)
    chart.axis("off")
    for m in range(1, length + 1):
        r, c = divmod(m - 1, per_row)
        chart.add_patch(Rectangle((c, r + .06), 1, .88, fill=False, ec="#9FB1C0", lw=1))
        chart.text(c + .96, r + .16, str(m), ha="right", va="center", fontsize=8, color=GREY)
        if m in letters:
            chart.text(c + .05, r + .17, letters[m], ha="left", va="center", fontsize=11, weight="bold", color=NAVY,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec=NAVY, lw=1.2))
            if m > 1:
                chart.plot([c, c], [r + .06, r + .94], color=NAVY, lw=3)
        items = labels.get(m, [])
        for offset, carried, text in items:
            chart.text(c + .06 + offset * .82, r + .46, text, ha="left", va="center",
                       fontsize={1: 16, 2: 13, 3: 10.5}.get(len(items), 9), color="#A7B5C1" if carried else "black",
                       weight="normal" if carried else "bold")
        for k in range(int(data["beats_per_bar"])):
            x = c + .06 + .88 * k / data["beats_per_bar"]
            chart.plot([x, x], [r + .80, r + .84], color="#C9D5DE", lw=1)
    band_cell = chart.add_patch(Rectangle((0, 0), 1, .88, color=BAND_FILL, lw=0, zorder=0))
    band_line = chart.add_patch(Rectangle((0, 0), 1, .04, color=BLUE, lw=0, zorder=1))
    follow_box = chart.add_patch(Rectangle((0, 0), 1, .88, fill=False, ec=ORANGE, lw=3, zorder=4))
    band_dot, = chart.plot([], [], "o", ms=11, color=BLUE, mec="white", mew=1.5, zorder=5)
    follow_dot, = chart.plot([], [], "o", ms=17, mfc="none", mec=ORANGE, mew=3, zorder=6)
    follow_tag = chart.text(0, 0, "", color=ORANGE, fontsize=10, ha="right", va="center", weight="bold", zorder=7)

    fig.text(.03, .265, "●", color=BLUE, fontsize=16, va="center")
    fig.text(.05, .265, "band (where they really are)", fontsize=12, va="center")
    fig.text(.33, .265, "○", color=ORANGE, fontsize=18, va="center", weight="bold")
    fig.text(.35, .265, args.label, fontsize=12, va="center")
    status = fig.text(.97, .265, "", fontsize=14, ha="right", va="center", weight="bold")

    roll = fig.add_axes([.03, .04, .94, .18])
    notes = np.column_stack([take.onsets, take.offsets, take.pitches])
    window = 6.0
    roll.set_ylim(np.percentile(notes[:, 2], 1) - 3, np.percentile(notes[:, 2], 99) + 3)
    roll.set_yticks([])
    roll.set_xticks([])
    for spine in roll.spines.values():
        spine.set_color(LIGHT)
    roll.set_title("piano notes the computer hears (last 6 s)", fontsize=10, color=GREY, loc="left", pad=3)
    note_patches = []

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
                tm, em = measure_of.get(int(true_bar[k])), measure_of.get(int(est_bar[k]))
                tc, ec = chorus_of.get(int(true_bar[k])), chorus_of.get(int(est_bar[k]))
                if tm:
                    r, c = divmod(tm - 1, per_row)
                    band_cell.set_xy((c, r + .06))
                    band_line.set_xy((c, r + .06))
                    band_dot.set_data([c + .06 + .88 * true_phase[k]], [r + .82])
                if em:
                    r2, c2 = divmod(em - 1, per_row)
                    follow_box.set_xy((c2, r2 + .06))
                    follow_dot.set_data([c2 + .06 + .88 * est_phase[k]], [r2 + .82])
                    follow_tag.set_position((c2 + .96, r2 + .66))
                    follow_tag.set_text("" if ec == tc else ("intro" if ec == 0 else f"chorus {ec}"))
                if true_bar[k] == est_bar[k]:
                    status.set_text("✓ right bar")
                    status.set_color("#2E7D32")
                else:
                    off = int(est_bar[k] - true_bar[k])
                    status.set_text(f"✗ lost: {abs(off)} bar{'s' if abs(off) > 1 else ''} {'ahead' if off > 0 else 'behind'}")
                    status.set_color(ORANGE)
                where = lambda c: "intro" if c == 0 else f"chorus {c}"
                counter.set_text(f"band: {where(tc)}{'' if tc == 0 else f' of {total}'} · bar {tm}" if tm else "")
                counter_f.set_text(f"computer: {where(ec)} · bar {em}" if em else "")
                for chip, g in zip(chips, groups):
                    band_here = g["start"] <= t < g["end"]
                    chip.set_facecolor(BAND_FILL if band_here else "#F3F6F9")
                    chip.set_edgecolor(BLUE if band_here else LIGHT)
                for artist in note_patches:
                    artist.remove()
                note_patches.clear()
                shown = notes[(notes[:, 1] > t - window) & (notes[:, 0] < t)]
                for on, off, pitch in shown:
                    note_patches.append(roll.add_patch(Rectangle((on, pitch - .6), min(off, t) - on, 1.2,
                                                                 color=ORANGE, alpha=.8, lw=0)))
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
