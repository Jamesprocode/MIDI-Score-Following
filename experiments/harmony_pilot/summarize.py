"""Produce compact, inspectable reports from the diagnostic CSVs (stdlib only)."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def percent(number):
    return f"{100 * float(number):.1f}%"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows
    ) + "</tbody></table>"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    rows = read("metrics.csv")
    primary = [r for r in rows if r["take"] == "preferred" and r["feature"] == "onsets"
               and float(r["window_seconds"]) == 2]
    groups = defaultdict(list)
    for row in primary:
        groups[(row["input"], row["method"])].append(row)
    macro = []
    for (condition, method), group in sorted(groups.items()):
        macro.append({"input": condition, "method": method,
                      **{key: mean(float(g[key]) for g in group) for key in [
                          "chord_accuracy_all_frames", "coverage", "tied_top_score_rate_active",
                          "tie_randomized_expected_accuracy_all_frames", "unique_correct_all_frames"]}})
    with (OUT / "primary_macro.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)

    per_song = []
    for song in summary["songs"]:
        match = {r["method"]: r for r in primary if r["song"] == song["song"] and r["input"] == "pooled"}
        per_song.append([song["song"], song["number_of_chords"],
                         percent(match["binary_cosine"]["chord_accuracy_all_frames"]),
                         percent(match["soft_extension_cosine"]["chord_accuracy_all_frames"]),
                         percent(song["uniform_guess_accuracy"]),
                         percent(song["largest_label_fraction_descriptive_not_trained_baseline"])])
    macro_table = [[row["input"], row["method"], percent(row["chord_accuracy_all_frames"]),
                    percent(row["coverage"]), percent(row["tied_top_score_rate_active"]),
                    percent(row["tie_randomized_expected_accuracy_all_frames"])] for row in macro]
    context_table = []
    for mode in ("onsets", "elapsed_duration"):
        for window in (0.5, 1, 2, 4):
            selected = [r for r in rows if r["take"] == "preferred" and r["input"] == "pooled"
                        and r["method"] == "binary_cosine" and r["feature"] == mode and float(r["window_seconds"]) == window]
            context_table.append([mode, window, percent(mean(float(r["chord_accuracy_all_frames"]) for r in selected)),
                                  percent(mean(float(r["coverage"]) for r in selected))])

    # Oracle switching is a diagnostic ceiling for this finite set of decisions.
    # It uses reference labels and is never a runnable inference method or learned result.
    predictions = read("predictions_primary.csv")
    frames = defaultdict(dict)
    for row in predictions:
        if row["take"] == "preferred" and row["method"] == "soft_extension_cosine":
            frames[(row["song"], row["time_seconds"])][row["input"]] = row["true_chord"] == row["predicted_chord"]
    oracle_by_song = defaultdict(list)
    for (song, _), outputs in frames.items():
        oracle_by_song[song].append(any(outputs[k] for k in ("bass", "horns", "pooled")))
    oracle = mean(mean(values) for values in oracle_by_song.values())
    oracle_data = {"description": "label-informed selection of an already correct bass/horns/pooled decision; not a trained gate",
                   "preferred_macro_accuracy": oracle,
                   "by_song": {song: mean(v) for song, v in oracle_by_song.items()}}
    (OUT / "oracle_diagnostic.json").write_text(json.dumps(oracle_data, indent=2) + "\n")

    doc = """<!doctype html><meta charset="utf-8"><title>JazzSAMBA harmonic pilot</title>
<style>body{max-width:1080px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#20242a}h1{font-size:28px}h2{font-size:21px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}th{background:#f2f4f6}code{background:#f2f4f6;padding:2px 4px}.notice{padding:14px 18px;background:#edf4fc;border-left:4px solid #376e9d}a{color:#245c91}</style>
<h1>JazzSAMBA: first harmonic-matching pilot</h1>
<p class="notice"><b>Completed diagnostic, not a position tracker.</b> Three songs, preferred and alternate takes, 18 separate bass/saxophone/trumpet MIDI files. No piano or drums, no training, no learned weighting. Primary tables below use preferred takes only. All 384 conditions and alternate results are in the CSV files.</p>
<h2>What was computed</h2>
<p>Every 100 ms, collect pitch classes from a trailing window and compare them with every distinct chord label in that song's chart. Templates are twelve-element vectors; cosine similarity measures how well their pitch classes match. Timing labels are used for evaluation, never to reset the input window or tell the matcher which chord comes next.</p>
<ul><li><b>Binary template:</b> written chord tones and explicit alterations have weight 1; other notes 0.</li>
<li><b>Soft extension template:</b> explicit tones remain 1; selected unaltered extensions receive 0.25 and other pitches 0.05. This is a deliberately simple, fully documented heuristic, not a reproduction of Maezawa, CHORD, or a uniquely correct chord-scale mapping.</li>
<li><b>Inputs:</b> bass; both horns; all three pooled; or all three with each active instrument normalized to equal total mass. Equal-instrument normalization is fixed, not adaptive learning.</li>
<li><b>Context:</b> 0.5, 1, 2, and 4 seconds. Onset counts versus elapsed note duration, clipped at the current time. Primary setting was declared as two-second onset counts before examining results.</li></ul>
<h2>Primary result by song</h2>
<p>All three available instruments pooled. Accuracy is the percentage of evaluated timestamps assigned the annotated chart chord. A window with no notes abstains and counts as incorrect. This evaluates chart compatibility, not necessarily the exact harmony performed.</p>
"""
    doc += table(["Song", "Chart labels", "Binary", "Soft extension", "Uniform guess", "Largest label share*"], per_song)
    doc += "<p>*Largest label share is calculated from evaluation annotations as an imbalance diagnostic, not a trained deployable baseline. Each pooled matcher is below it on these songs; do not report the raw percentages as strong recognition performance.</p>"
    doc += "<h2>Which instruments help?</h2><p>Equal mean over the three preferred songs. Coverage means the window contains some usable notes. Top-score ties are broken alphabetically for ordinary accuracy. The final column also shows expected accuracy under random tie breaking, so arbitrary ordering is visible.</p>"
    doc += table(["Input", "Matcher", "Accuracy", "Coverage", "Ties when active", "Random-tie expected accuracy"], macro_table)
    doc += "<h2>Window length and note representation</h2><p>Binary templates, pooled input, preferred takes. These are exploratory comparisons, not tuned test-set claims. Longer context can mix adjacent chords.</p>"
    doc += table(["Note representation", "Window (s)", "Accuracy", "Coverage"], context_table)
    doc += f"<h2>Does adaptive selection have room to help?</h2><p>An oracle that uses the correct labels to select an already correct bass-only, horns-only, or pooled prediction reaches <b>{percent(oracle)}</b> under the primary soft-template setting. This shows some complementary decisions. It is not a learned result, does not prove the selector is learnable, and is not an upper bound for all possible score-level fusion methods.</p>"
    doc += """<h2>What we can conclude</h2>
<ul><li>Adding the available horn notes modestly improves aggregate compatibility over bass alone; it does not produce reliable chord recognition.</li>
<li>Soft extensions give only a small average improvement. The input representation, context length, source availability, and annotation correspondence warrant attention before a larger weighting model.</li>
<li>Several different labels receive the same best score. G-6 and E-7b5 even have identical binary pitch-class templates. An uncertain compatibility distribution is more appropriate than treating every top label as certain.</li>
<li>The next controlled comparison should add the missing piano MIDI while keeping the same templates and evaluation. A learned root-relative pitch-class profile is a subsequent alternative, trained only on other songs.</li></ul>
<h2>Boundaries and verification</h2>
<ul><li>Three deliberately selected exploratory songs, not independent model-selection and test sets. No claim of generalization.</li>
<li>Official annotation JSONs describe preferred performances. Alternate evaluation uses the release's documented copied async grid and is kept as exploratory.</li>
<li>Original timing origins are preserved; no time shift was fitted to improve matching. Audio and annotation alignment were not independently verified. Played substitutions, anticipation, transcription errors, or reference errors can reduce compatibility.</li>
<li>Initial padding and the terminal zero-length Tunisia chord are excluded, as are note tails outside labeled intervals. No labeled transition frames are otherwise discarded.</li>
<li>All note velocities are 100. Dynamics and pedal cannot provide evidence in these files.</li>
<li>All files have one musical note track plus metadata. Instrument identity comes from the filename, since all note channels are zero.</li>
<li>The checks cover future-note/duration invariance, timing conversion, root spelling, chord alterations, and annotation continuity. Full MIDI and annotation hashes are saved.</li>
<li>Offline vectorized runtime is not streaming latency. No beat tracker, positional filter, recovery benchmark, or adaptive network was evaluated.</li></ul>
<h2>Reproduce and inspect</h2>
<p>Run <code>python run_pilot.py</code> (NumPy required), then <code>python summarize.py</code>. Original dataset files remain unchanged.</p>
<ul><li><a href="metrics.csv">All condition metrics</a></li><li><a href="by_section.csv">Primary conditions by annotated section</a></li><li><a href="predictions_primary.csv">Primary predictions at every timestamp</a></li><li><a href="summary.json">Definitions, chord templates, and configuration</a></li><li><a href="midi_audit.json">Input-file audit and hashes</a></li><li><a href="../data/sources.json">Official annotation provenance</a></li></ul>
<p>Sources: <a href="https://zenodo.org/records/22943963">JazzSAMBA release</a>; <a href="https://github.com/pnlong/jazzsamba-demo/tree/main/data/annotations/async">official annotation exports</a>. Template-matching precedent: <a href="https://interactiveaudiolab.github.io/assets/papers/pardo-birmingham-cmj02.pdf">Pardo and Birmingham (2002)</a>; the cosine baselines here are not a reproduction of its score.</p>
"""
    (OUT / "report.html").write_text(doc)
    print(json.dumps({"report": str(OUT / "report.html"), "oracle_diagnostic": oracle_data}, indent=2))


if __name__ == "__main__":
    main()
