"""Build a static scientific report from saved ensemble benchmark results.

Reads CSV/JSON outputs only; never reruns models or modifies source datasets.
All aggregate metrics are equal means of song-level metrics within one protocol
and take. Best-window tables are explicitly exploratory selection on these songs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
METHOD_NAMES = {
    "cosine_onsets": "Cosine · onset counts",
    "cosine_duration": "Cosine · elapsed duration",
    "pardo": "Pardo-style segment score",
    "yamaha_active": "Maezawa-style active notes",
    "empirical_profile": "Empirical pitch-class profile",
    "clock": "Constant-tempo clock",
    "uniform_filter": "Shared filter · no harmonic evidence",
}
CHORD_METRICS = (
    "chord_accuracy", "fractional_accuracy", "balanced_accuracy", "root_accuracy",
    "coverage", "tie_rate", "chart_majority_accuracy",
)
POSITION_METRICS = ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats")
CASE_FIELDS = ("protocol", "take", "scheme", "input", "preprocess", "method", "window_seconds", "initial_tempo_factor")


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def pct(value):
    value = number(value)
    return "—" if value is None else f"{100 * value:.1f}%"


def decimal(value):
    value = number(value)
    return "—" if value is None else f"{value:.2f}"


def label(method):
    return METHOD_NAMES.get(method, method)


def seconds(value):
    value = number(value)
    if value is None:
        return "—"
    return "Current" if value == 0 else f"{value:g} s"


def table(headers, rows):
    if not rows:
        return "<p>No results for this condition.</p>"
    return '<div class="table-wrap"><table><thead><tr>' + "".join(
        f"<th>{esc(value)}</th>" for value in headers
    ) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows
    ) + "</tbody></table></div>"


def read_csv(path, required, warnings):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    identities = set()
    for row in rows:
        key = (row["performance_id"],) + tuple(row.get(k, "") for k in CASE_FIELDS)
        if key in identities:
            raise ValueError(f"Duplicate condition for a performance in {path.name}: {key}")
        identities.add(key)
        if number(row["frames"]) is None or float(row["frames"]) <= 0:
            raise ValueError(f"Invalid evaluation frame count in {path.name}: {key}")
        for metric in CHORD_METRICS if path.name == "chord_metrics.csv" else POSITION_METRICS:
            value = number(row.get(metric))
            if value is None:
                warnings.append(f"{path.name}: missing/nonfinite {metric} for {row['performance_id']} ({row['method']}).")
            elif "accuracy" in metric or metric in ("coverage", "tie_rate"):
                if not -1e-9 <= value <= 1 + 1e-9:
                    raise ValueError(f"Metric outside [0,1]: {path.name}, {metric}, {value}")
    return rows


def aggregate(rows, metrics):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in CASE_FIELDS)
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        # One performance per song within each take. Refuse silent repetition.
        if len({row["song"] for row in group}) != len(group):
            raise ValueError(f"Repeated song within one aggregate: {key}")
        result = dict(zip(CASE_FIELDS, key))
        result.update(n_songs=len(group), n_frames=sum(int(row["frames"]) for row in group),
                      performance_ids=frozenset(row["performance_id"] for row in group))
        for metric in metrics:
            values = [number(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            result[metric] = mean(values) if values else None
            result[metric + "_n"] = len(values)
        output.append(result)
    return output


def select(rows, **criteria):
    return [row for row in rows if all(str(row.get(k, "")) == str(value) for k, value in criteria.items())]


def fixed_context(row):
    window = number(row["window_seconds"])
    return window == 0 if row["method"] == "yamaha_active" else window == 1


def chord_table(rows):
    rows = sorted(rows, key=lambda row: (-(number(row["chord_accuracy"]) or 0), row["method"], number(row["window_seconds"]) or 0))
    return table(
        ["Method", "Context", "Songs", "Chord accuracy", "Fractional ties", "Balanced accuracy", "Root accuracy", "Coverage", "Tie rate"],
        [[label(row["method"]), seconds(row["window_seconds"]), row["n_songs"],
          pct(row["chord_accuracy"]), pct(row["fractional_accuracy"]), pct(row["balanced_accuracy"]),
          pct(row["root_accuracy"]), pct(row["coverage"]), pct(row["tie_rate"])] for row in rows],
    )


def position_table(rows):
    rows = sorted(rows, key=lambda row: (row["method"] != "clock", -(number(row["bar_accuracy"]) or 0), row["method"]))
    return table(
        ["Method", "Songs", "Exact unfolded bar", "Within 2 beats", "Mean absolute error (beats)", "Mean song median error (beats)"],
        [[label(row["method"]), row["n_songs"], pct(row["bar_accuracy"]), pct(row["within2beat_accuracy"]),
          decimal(row["mean_abs_beats"]), decimal(row["median_abs_beats"])] for row in rows],
    )


def per_song_table(rows, task):
    """Show comparable saved decisions without averaging across songs."""
    methods = sorted({row["method"] for row in rows},
                     key=lambda method: (method not in ("clock", "uniform_filter"),
                                         method != "clock", label(method)))
    by_song = defaultdict(dict)
    for row in rows:
        if row["method"] in by_song[row["song"]]:
            raise ValueError(f"Repeated per-song {task} comparison: {row['song']}/{row['method']}")
        by_song[row["song"]][row["method"]] = row
    headers = ["Song", "Frames"]
    if task == "chord":
        headers += ["Chart labels", "Chart majority"]
    headers += [label(method) for method in methods]
    output = []
    for song, values in sorted(by_song.items()):
        example = next(iter(values.values()))
        counts = sorted({int(row["frames"]) for row in values.values()})
        cells = [song, "/".join(map(str, counts))]
        if task == "chord":
            cells += [example["n_chord_labels"], pct(example["chart_majority_accuracy"])]
        for method in methods:
            row = values.get(method)
            if row is None:
                cells.append("—")
            elif task == "chord":
                cells.append(pct(row["chord_accuracy"]))
            else:
                cells.append(pct(row["bar_accuracy"]) + " / " + pct(row["within2beat_accuracy"]))
        output.append(cells)
    return table(headers, output)


def check_cohort(rows, title, warnings):
    if len({row["performance_ids"] for row in rows}) > 1:
        warnings.append(f"{title}: methods/configurations have different song coverage; use the displayed sample counts and inspect per-song CSVs before direct comparison.")


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def build_report(results):
    warnings = []
    summary = load_json(results / "summary.json")
    position_pending = summary is not None and summary.get("position_complete") is False
    identity = ("performance_id", "song", "protocol", "take", "scheme", "method", "window_seconds", "frames")
    chord = read_csv(results / "chord_metrics.csv", identity + ("input", "preprocess") + CHORD_METRICS, warnings)
    position_path = results / "position_metrics.csv"
    position = read_csv(position_path, identity + ("initial_tempo_factor",) + POSITION_METRICS, warnings) if position_path.exists() and not position_pending else []
    if position_pending:
        warnings.append("Position evaluation is pending. Any existing position CSV was deliberately ignored, so stale or superseded position results are not displayed.")
    elif not position:
        warnings.append("Position results are absent; no position-tracking conclusion can be drawn from this report.")
    chord_macro = aggregate(chord, CHORD_METRICS)
    position_macro = aggregate(position, POSITION_METRICS)
    broad_path = results / "position_metrics_broad_tempo.csv"
    broad_config_path = results / "position_broad_tempo_config.json"
    broad_summary_path = results / "position_broad_tempo_summary.json"
    broad_config = load_json(broad_config_path)
    broad_summary = load_json(broad_summary_path)
    broad_pending = broad_summary is not None and (
        broad_summary.get("complete") is False or broad_summary.get("position_complete") is False)
    broad = []
    if broad_path.exists() and not broad_pending:
        broad = read_csv(broad_path, identity + ("initial_tempo_factor",) + POSITION_METRICS, warnings)
        if broad_config is None or broad_summary is None:
            warnings.append("The broad-tempo sensitivity CSV exists but its separate config/summary metadata is incomplete; inspect its provenance before interpreting it.")
        invalid = [row for row in broad if row["take"] != "preferred" or row["scheme"] != "triad"
                   or number(row["initial_tempo_factor"]) != 1]
        if invalid:
            raise ValueError("The broad-tempo sensitivity must contain only preferred, triad, nominal-initial-tempo rows.")
    elif broad_pending:
        warnings.append("The separate broad-tempo sensitivity run is incomplete and is not displayed.")
    broad_macro = aggregate(broad, POSITION_METRICS)
    config_path = next((p for p in (results / "config.json", results / "configuration.json") if p.exists()), None)
    config = load_json(config_path) if config_path else None
    audit = load_json(HERE / "audit" / "summary.json")
    if config is None:
        warnings.append("No config.json or configuration.json is present next to the CSVs.")
    if summary is None:
        warnings.append("No summary.json is present next to the CSVs.")
    elif summary.get("complete") is False and not position_pending:
        warnings.append("This is a partial/smoke run, not the completed all-song benchmark.")

    doc = ['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JazzSAMBA ensemble benchmark</title>
<style>body{max-width:1220px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui,sans-serif;color:#20242a}h1{font-size:29px}h2{font-size:23px;margin-top:34px}h3{font-size:18px;margin-top:23px}.notice{background:#fff5df;border-left:4px solid #986728;padding:14px 18px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 22px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #ddd;font-variant-numeric:tabular-nums}th{background:#f3f4f5}small,.muted{color:#59616a}a{color:#245c91}code{font-size:13px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}</style></head><body>
<h1>JazzSAMBA ensemble benchmark</h1>
<p class="notice"><b>Exploratory comparison on the available songs.</b> These are harmonic scorers and a shared position-filter baseline, not reproductions of the complete Maezawa or Pardo systems. Scores selected as best on these same songs are not held-out test results. Async and sync are evaluated separately.</p>''']
    if position_pending and summary.get("chord_complete") is True:
        doc.append('<p class="notice"><b>Chord evaluation complete; position evaluation pending.</b> Chord tables use the completed run. Position tables will appear after the position rerun is validated.</p>')
    elif summary is not None and summary.get("complete") is False:
        doc.append('<p class="notice"><b>Partial/smoke run:</b> these tables do not yet represent the completed all-song benchmark.</p>')
    identities = {row["performance_id"]: row for row in chord}
    counts = Counter((row["protocol"], row["take"]) for row in identities.values())
    doc.append("<h2>Evaluation scope</h2>")
    doc.append(table(["Protocol", "Take", "Performances"], [[protocol, take, count] for (protocol, take), count in sorted(counts.items())]))
    doc.append(f"<p>{len({row['song'] for row in chord})} distinct song folders; {len(identities)} evaluated performances. Each table averages song-level metrics equally. Long songs do not receive more weight. Preferred takes are primary; alternate async takes are secondary. The available symbolic ensemble is bass plus available horns; piano MIDI is absent, and percussion is excluded from harmonic evidence.</p>")
    doc.append('<p>These are provisional results on automatic MIDI transcriptions. Annotation coordinates are internally consistent, but independent audio/MIDI/annotation origin alignment has not been verified. Local audio duration differences concern debleeded overlays and do not by themselves demonstrate a timing offset. See the <a href="../audit/timing_provenance.md">timing provenance audit</a>.</p>')
    doc.append("<p>Every scorer chooses among the unique reduced chord labels in the known chart; this is chart-constrained recognition. All methods receive the same target vocabulary and evaluation frames within a condition. Chord accuracy uses a single predicted label, including the root and the quality retained by the target scheme. Triad targets are primary; seventh targets are secondary. Pardo applies its root-weight tie rule; remaining ties use the fixed vocabulary order. Fractional-tie accuracy shares one point among remaining tied winners and is shown separately from hard classification accuracy. Input silence counts as an incorrect/abstaining chord prediction. Balanced accuracy is the per-song mean recall over represented target classes; root accuracy ignores quality. Coverage and tie rates help interpret sparse or ambiguous evidence.</p>")

    preferred = select(chord_macro, take="preferred", input="pooled", preprocess="raw")
    protocols = [protocol for protocol in ("async", "sync") if any(row["protocol"] == protocol for row in preferred)]
    schemes = [scheme for scheme in ("triad", "seventh") if any(row["scheme"] == scheme for row in preferred)]
    doc.append("<h2>Chord recognition: fixed context</h2><p>Raw pooled pitched input, one-second past context. Maezawa-style active-note scoring uses only currently sounding notes (shown as “Current”). The representation and context differences are explicit; these are not isolated changes to the scoring formula.</p>")
    for scheme in schemes:
        for protocol in protocols:
            rows = [row for row in select(preferred, scheme=scheme, protocol=protocol) if fixed_context(row)]
            title = f"{protocol} · {scheme} targets"
            check_cohort(rows, title, warnings)
            doc.append(f"<h3>{esc(title)}</h3>" + chord_table(rows))
            if rows:
                baseline = rows[0]["chart_majority_accuracy"]
                doc.append(f"<p class=\"muted\">Chart-only majority baseline: {pct(baseline)} mean accuracy. It always predicts the chord occupying the most musical beats in the supplied untimed chart. Its choice uses no performance timestamps; the displayed accuracy is measured against the performed reference.</p>")
            song_rows = [row for row in select(chord, take="preferred", input="pooled", preprocess="raw", scheme=scheme, protocol=protocol)
                         if fixed_context(row)]
            doc.append(f"<details><summary>Per-song preferred chord accuracy · {esc(title)}</summary>"
                       "<p>Same fixed context and raw pooled input as the table above. Percentages are ordinary chord accuracy; missing method results appear as a dash.</p>"
                       + per_song_table(song_rows, "chord") + "</details>")

    doc.append("<h2>Position tracking: shared follower and clock</h2><p>Every harmonic method uses the same causal position/tempo filter and the same known unfolded chord plan, synchronized start position/time, and initial BPM. The constant-tempo clock uses that initialization without harmonic observations. Exact bar accuracy requires the correct unfolded bar occurrence, including its repeat/chorus. The within-two-beat rate is reported separately. Error columns are equal means over songs; the last column averages each song's median and is not a pooled median.</p>")
    doc.append("<p class=\"notice\">Async was recorded against a fixed timing plan. With the correct synchronized start and nominal tempo, a perfect or near-perfect clock is expected and does not demonstrate score-following ability. Sync is the more relevant condition for testing correction of tempo drift. Initial-tempo perturbations, when available below, test a distinct initialization stress condition.</p>")
    for scheme in schemes:
        for protocol in protocols:
            rows = [row for row in select(position_macro, take="preferred", scheme=scheme, protocol=protocol)
                    if number(row["initial_tempo_factor"]) == 1]
            check_cohort(rows, f"Position {protocol}/{scheme}", warnings)
            doc.append(f"<h3>{esc(protocol)} · {esc(scheme)} targets · nominal initial tempo</h3>" + position_table(rows))
            song_rows = [row for row in select(position, take="preferred", scheme=scheme, protocol=protocol)
                         if number(row["initial_tempo_factor"]) == 1]
            doc.append(f"<details><summary>Per-song preferred position comparison · {esc(protocol)} · {esc(scheme)}</summary>"
                       "<p>Each method cell shows exact unfolded-bar accuracy / within-two-beat accuracy. These are the primary nominal-tempo conditions only.</p>"
                       + per_song_table(song_rows, "position") + "</details>")

    if broad or broad_pending or broad_config is not None or broad_summary is not None:
        doc.append("<h2>Separate sensitivity: broader tempo range</h2>"
                   "<p>This exploratory rerun allows tempo hypotheses from 0.5 to 1.8 times the supplied initial BPM, using the same shared follower otherwise. It uses preferred takes, triad targets, and nominal initial tempo. It remains separate from the primary position tables and does not replace their configuration or results.</p>")
        if broad_pending:
            doc.append("<p>The broader-tempo run is still pending; no partial or superseded results are shown.</p>")
        for protocol in protocols:
            rows = select(broad_macro, protocol=protocol)
            if rows:
                check_cohort(rows, f"Broad-tempo sensitivity {protocol}", warnings)
                doc.append(f"<h3>{esc(protocol)} · broader tempo range</h3>" + position_table(rows))
        if broad_config is not None or broad_summary is not None:
            doc.append("<details><summary>Broader-tempo configuration and run status</summary>")
            for name, data in ((broad_config_path.name, broad_config), (broad_summary_path.name, broad_summary)):
                if data is not None:
                    doc.append(f'<p><a href="{esc(name)}">{esc(name)}</a></p><pre style="overflow:auto;white-space:pre-wrap">{esc(json.dumps(data, indent=2))}</pre>')
            doc.append("</details>")

    doc.append("<h2>Exploratory window selection</h2><p>For each method and protocol, this table selects the window with the highest song-macro chord accuracy on the very songs being reported. This is a descriptive sweep, not model selection followed by an independent test. The fixed-context tables above remain separate. Ties select the shorter context. Raw pooled input is retained.</p>")
    for scheme in schemes:
        for protocol in protocols:
            candidates = select(preferred, scheme=scheme, protocol=protocol)
            by_method = defaultdict(list)
            for row in candidates:
                by_method[row["method"]].append(row)
            best = []
            for method, rows in by_method.items():
                check_cohort(rows, f"Window sweep {protocol}/{scheme}/{method}", warnings)
                best.append(max(rows, key=lambda row: (number(row["chord_accuracy"]) or 0, -(number(row["window_seconds"]) or 0))))
            doc.append(f"<h3>{esc(protocol)} · {esc(scheme)} targets</h3>" + chord_table(best))

    secondary = [row for row in chord_macro if row["protocol"] == "async" and row["take"] == "alternate"
                 and row["input"] == "pooled" and row["preprocess"] == "raw" and row["scheme"] == "triad" and fixed_context(row)]
    doc.append("<h2>Secondary checks</h2><h3>Alternate async takes · triad targets</h3><p>The release documents the shared async annotation grid. These are additional takes of the same songs, not an independent composition-level test set.</p>" + chord_table(secondary))
    sensitivity = [row for row in chord_macro if row["take"] == "preferred" and row["scheme"] == "triad"
                   and fixed_context(row) and (row["preprocess"] != "raw" or row["input"] != "pooled")]
    doc.append("<details><summary>Input and preprocessing sensitivity at fixed context</summary><p>These variants change the evidence. A duration cap is a declared preprocessing choice, not a correction known to recover the performed notes.</p>")
    doc.append(table(["Protocol", "Input", "Preprocessing", "Method", "Songs", "Chord accuracy", "Coverage"],
        [[row["protocol"], row["input"], row["preprocess"], label(row["method"]), row["n_songs"], pct(row["chord_accuracy"]), pct(row["coverage"])]
         for row in sorted(sensitivity, key=lambda row: (row["protocol"], row["input"], row["preprocess"], row["method"]))]))
    doc.append("</details>")
    stress = [row for row in position_macro if row["take"] == "preferred" and row["scheme"] == "triad"
              and number(row["initial_tempo_factor"]) not in (None, 1)]
    doc.append("<details><summary>Initial-tempo stress · triad targets</summary>")
    for protocol in protocols:
        factors = sorted({float(row["initial_tempo_factor"]) for row in stress if row["protocol"] == protocol})
        for factor in factors:
            rows = [row for row in stress if row["protocol"] == protocol and float(row["initial_tempo_factor"]) == factor]
            doc.append(f"<h3>{esc(protocol)} · initial BPM × {factor:g}</h3>" + position_table(rows))
    if not stress:
        doc.append("<p>No initial-tempo stress results were saved.</p>")
    doc.append("</details>")

    doc.append("<h2>Interpretation and verification</h2><ul>")
    for text in [
        "Maezawa-style scoring reconstructs the published harmonic dot-product form with declared template values; the published downbeat network, HHSMM and accompaniment system are not reproduced.",
        "Pardo-style scoring adapts event-partition note weights and its matching/missing-tone score to causal windows and the benchmark vocabulary. Complete segmentation and lead-sheet sequence alignment are not reproduced.",
        "The empirical profile is a small data-derived harmonic model, not a neural model. Its training split must exclude the target composition and all its takes; consult the saved configuration for the implemented split.",
        "Repeated chord labels alone cannot identify a unique bar. Position results also depend on initialization, the shared filter, and the known unfolded form.",
        "Transcription errors, held-note artifacts, anticipations, substitutions, and annotation disagreements can reduce chart compatibility. Valid MIDI syntax does not establish a correct transcription.",
        "No adaptive cue-weighting model is tested merely by comparing these scorers. No result here establishes unseen-composition generalization for window or preprocessing selection.",
        "Offline execution time is not streaming latency. This report does not claim real-time deployment or recovery from arbitrary repeats/skips.",
    ]:
        doc.append(f"<li>{esc(text)}</li>")
    if audit:
        doc.append(f"<li>The saved local audit found {esc(audit.get('total_midi_files', 'unknown'))} MIDI files and {esc(audit.get('quality_flagged_core_files', 'unknown'))} core files with predefined content-quality flags. MIDI meter tags do not reliably encode the chart meter; tempo metadata is used for tick-to-second conversion, not as proof of constant performed tempo.</li>")
    doc.append("</ul>")
    if warnings:
        doc.append("<div class=\"notice\"><b>Report QA notices</b><ul>" + "".join(f"<li>{esc(warning)}</li>" for warning in sorted(set(warnings))) + "</ul></div>")
    else:
        doc.append("<p>Report QA passed: no duplicate performance/condition rows, invalid frame counts, nonfinite required metrics, or mismatched cohorts in the displayed primary comparisons were detected.</p>")
    doc.append("<h2>Inspect and reproduce</h2><p>Tables are regenerated from saved per-song metrics by <code>python report_benchmark.py</code>. This does not rerun inference or alter the original datasets.</p><ul>")
    for name, title in [("chord_metrics.csv", "Per-song chord metrics"), ("position_metrics.csv", "Per-song position metrics"),
                        (broad_path.name, "Separate broader-tempo position metrics"),
                        (broad_config_path.name, "Broader-tempo configuration"),
                        (broad_summary_path.name, "Broader-tempo run summary"),
                        (config_path.name if config_path else "config.json", "Configuration and method definitions"),
                        ("summary.json", "Run summary")]:
        if (results / name).exists():
            doc.append(f'<li><a href="{esc(name)}">{esc(title)}</a></li>')
    doc.append('<li><a href="../audit/README.md">Local source-data audit</a></li></ul>')
    doc.append(f"<p class=\"muted\">Report generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}. Source rows: {len(chord):,} chord conditions; {len(position):,} primary position conditions; {len(broad):,} separate broader-tempo conditions.</p></body></html>")
    (results / "report.html").write_text("\n".join(doc))
    return {"report": str(results / "report.html"), "chord_rows": len(chord), "position_rows": len(position),
            "broad_tempo_rows": len(broad), "performances": len(identities), "warnings": sorted(set(warnings))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    print(json.dumps(build_report(args.results.resolve()), indent=2))


if __name__ == "__main__":
    main()
