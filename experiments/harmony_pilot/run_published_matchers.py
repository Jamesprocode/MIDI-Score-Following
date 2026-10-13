"""Compare published harmonic scoring rules, not complete published followers.

Uses the same untimed chart vocabulary, MIDI, and 100-ms evaluation grid as the
initial pilot. Source-specific representations and every adaptation are recorded.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np

import run_pilot as base
import pardo_matcher as pardo
import yamaha_matcher as yamaha

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_published_matchers"


def evaluate(scores, availability, target, winner_mask=None):
    if winner_mask is None:
        winner_mask = np.isclose(scores, scores.max(axis=1, keepdims=True), atol=1e-10, rtol=0)
    else:
        winner_mask = winner_mask.copy()
        empty = ~winner_mask.any(axis=1)
        assert not availability[empty].any(), "An active row must retain a candidate"
        winner_mask[empty] = True  # The availability mask still forces abstention/zero credit.
    tie_count = winner_mask.sum(axis=1)
    assert np.all(tie_count > 0)
    indices = np.arange(len(scores))
    correct_in_tie = winner_mask[indices, target]
    credit = correct_in_tie / tie_count * availability
    # Column order is alphabetical chart label order, exactly as in original pilot.
    hard_winner = winner_mask.argmax(axis=1)
    hard_correct = (hard_winner == target) & availability
    unique_correct = correct_in_tie & (tie_count == 1) & availability
    return {
        "frames": len(target), "active_frames": int(availability.sum()),
        "coverage": float(availability.mean()),
        "fractional_tie_accuracy": float(credit.mean()),
        "alphabetical_tie_accuracy": float(hard_correct.mean()),
        "unique_correct_accuracy": float(unique_correct.mean()),
        "tie_rate_active": float((tie_count[availability] > 1).mean()) if availability.any() else None,
        "fractional_accuracy_active": float(credit[availability].mean()) if availability.any() else None,
    }, credit, hard_winner, tie_count


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return f"{float(value)*100:.1f}%"


def html_table(headers, rows):
    return "<table><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</table>"


def main():
    OUT.mkdir(exist_ok=True)
    base.checks()
    yamaha.checks()
    pardo.checks()
    started = time.perf_counter()
    rows, predictions, sections, audit, inputs = [], [], [], [], []
    for song in base.SONGS:
        annotation = json.loads((HERE / "data" / "annotations" / f"{song}.json").read_text())
        chords = [c for c in annotation["chords"] if float(c["end_time"]) > float(c["start_time"])
                  and not base.truth(c["is_no_chord"])]
        starts = np.array([float(c["start_time"]) for c in chords])
        ends = np.array([float(c["end_time"]) for c in chords])
        times = np.arange(0, ends.max(), base.STEP)
        idx = np.searchsorted(starts, times, side="right") - 1
        mask = (idx >= 0) & (times < ends[np.maximum(idx, 0)])
        times, idx = times[mask], idx[mask]
        lookup = {c["chord"]: c for c in chords}
        labels = sorted(lookup)
        definitions = [lookup[label] for label in labels]
        templates = np.stack([base.chord_template(c) for c in definitions])
        soft_templates = np.stack([base.chord_template(c, True) for c in definitions])
        templates88 = yamaha.expand_chroma_templates(templates)
        soft88 = yamaha.expand_chroma_templates(soft_templates)
        target = np.array([labels.index(chords[i]["chord"]) for i in idx])
        roots = np.array([base.pitch_class(c["root"]) for c in definitions])
        sec_starts = np.array([float(s["start_time"]) for s in annotation["sections"]])
        sec_idx = np.searchsorted(sec_starts, times, side="right") - 1
        section_names = np.array([annotation["sections"][i]["top_level_section"] for i in sec_idx])
        native_sets = [{0,4,7}, {0,4,7,10}, {0,3,7}, {0,3,6,9}, {0,3,6,10}, {0,3,6}]
        supported = np.array([set(np.flatnonzero(np.roll(template, -root))) in native_sets
                              for template, root in zip(templates, roots)])
        audit.append({"song": song, "labels": labels,
                      "native_pardo72_supported_labels": [l for l,s in zip(labels, supported) if s],
                      "native_pardo72_unavailable_labels": [l for l,s in zip(labels, supported) if not s],
                      "native_pardo72_label_coverage": float(supported[target].mean()),
                      "evaluated_frames_per_take": len(times)})
        for take in ("preferred", "alternate"):
            streams = {}
            for path in sorted((HERE.parents[1] / "async" / "songs" / song / take).glob("*.mid")):
                notes, info = base.read_midi(path)
                streams[path.name.split(".")[0]] = notes
                inputs.append({"song": song, "take": take, **info,
                               "notes_outside_yamaha_piano_range": int(((notes[:,2] < 21) | (notes[:,2] > 108)).sum())})
            for condition in ("bass", "horns", "pooled"):
                parts = [n for k,n in streams.items() if condition == "pooled" or
                         (condition == "bass" and k == "bass") or (condition == "horns" and k != "bass")]
                notes = np.concatenate(parts)
                notes = notes[np.argsort(notes[:,0])]
                active88 = yamaha.piano_roll88(notes, times)
                active_available = active88.sum(axis=1) > 0
                case_list = [
                    ("yamaha_active_binary", "active_binary_pianoroll88", 0.0,
                     yamaha.harmonic_log_scores(active88, templates88), active_available, None),
                    ("yamaha_active_soft_sensitivity", "active_binary_pianoroll88", 0.0,
                     yamaha.harmonic_log_scores(active88, soft88), active_available, None),
                    ("cosine_active_control", "active_binary_pianoroll88", 0.0,
                     base.cosine_scores(active88, templates88), active_available, None),
                ]
                for window in base.WINDOWS:
                    profiles = pardo.segment_profiles(notes, times, window)
                    pscores = pardo.score(profiles, templates)
                    available = profiles.sum(axis=1) > 0
                    case_list.append(("pardo_segment_score", "minimal_segment_counts", window, pscores, available, None))
                    case_list.append(("pardo_root_tiebreak", "minimal_segment_counts", window, pscores, available,
                                      pardo.root_tie_mask(profiles, pscores, roots)))
                    # Same representation, only scoring rule changes: separates scoring from context effects.
                    case_list.append(("cosine_segment_control", "minimal_segment_counts", window,
                                      base.cosine_scores(profiles, templates), available, None))
                    case_list.append(("dot_segment_control", "minimal_segment_counts", window,
                                      profiles @ templates.T, available, None))
                    if window == base.PRIMARY_WINDOW:
                        onset = base.features(notes, times, window, "onsets")
                        case_list.append(("previous_cosine_onsets", "onset_counts", window,
                                          base.cosine_scores(onset, templates), onset.sum(axis=1) > 0, None))
                for method, feature, window, scores, available, winner_mask in case_list:
                    metrics, credit, pred_idx, ties = evaluate(scores, available, target, winner_mask)
                    metadata = {"song": song, "take": take, "input": condition, "method": method,
                                "feature": feature, "window_seconds": window}
                    rows.append({**metadata, **metrics})
                    if window in (0.0, base.PRIMARY_WINDOW):
                        for section in sorted(set(section_names)):
                            selected = section_names == section
                            section_metrics, _, _, _ = evaluate(scores[selected], available[selected], target[selected],
                                                                winner_mask[selected] if winner_mask is not None else None)
                            sections.append({**metadata, "section": section, **section_metrics})
                        for i,t in enumerate(times):
                            predictions.append({**metadata, "time_seconds": round(float(t),6), "section": section_names[i],
                                                "true_chord": labels[target[i]],
                                                "predicted_chord_alphabetical": labels[pred_idx[i]] if available[i] else "ABSTAIN",
                                                "available": int(available[i]), "top_tie_count": int(ties[i]),
                                                "correct_fractional_credit": float(credit[i])})
    csv_write(OUT / "metrics.csv", rows)
    csv_write(OUT / "by_section.csv", sections)
    csv_write(OUT / "predictions.csv", predictions)
    macro = []
    for take in ("preferred", "alternate"):
        for key in sorted(set((r["input"],r["method"],r["feature"],r["window_seconds"]) for r in rows)):
            condition, method, feature, window = key
            group = [r for r in rows if r["take"] == take and (r["input"],r["method"],r["feature"],r["window_seconds"]) == key]
            assert len(group) == 3
            macro.append({"take": take, "input": condition, "method": method, "feature": feature,
                          "window_seconds": window,
                          **{m:mean(r[m] for r in group) for m in ["fractional_tie_accuracy", "alphabetical_tie_accuracy",
                                                                  "unique_correct_accuracy", "coverage", "tie_rate_active"]}})
    csv_write(OUT / "macro.csv", macro)
    # Regression check: re-running the original representation/scorer must reproduce original per-song results.
    with (HERE / "results" / "metrics.csv").open() as f:
        old = list(csv.DictReader(f))
    regressions = 0
    for row in rows:
        if row["method"] != "previous_cosine_onsets":
            continue
        match = [r for r in old if r["song"] == row["song"] and r["take"] == row["take"] and r["input"] == row["input"]
                 and r["feature"] == "onsets" and float(r["window_seconds"]) == 2 and r["method"] == "binary_cosine"]
        assert len(match) == 1
        assert np.isclose(row["alphabetical_tie_accuracy"], float(match[0]["chord_accuracy_all_frames"]), atol=1e-12)
        assert np.isclose(row["fractional_tie_accuracy"], float(match[0]["tie_randomized_expected_accuracy_all_frames"]), atol=1e-12)
        regressions += 1
    configuration = {
        "task": "harmonic scoring rule comparison; NOT the full Maezawa or Pardo score followers",
        "training": "none", "grid_seconds": base.STEP, "primary_pardo_window_seconds": 2,
        "song_protocol": "async fixed-tempo overdubs", "take_scope": "preferred primary; alternate exploratory",
        "candidate_vocabulary": "each song's full set of written chord labels, shared across methods; no timed input labels",
        "yamaha": {
            "source": "Maezawa 2023 ISMIR LBD, Section 2.2, page 2",
            "source_url": "https://ismir2023program.ismir.net/lbd_365.html",
            "equation": "log p_h(x|s=i) up to an additive constant = x^T T_ci,ri + x^T m",
            "implemented": "binary currently held MIDI21..108; chord template tiled over pitch octaves; absent melody m=0",
            "underspecified": "published numerical templates and piano-roll value encoding unavailable",
            "chosen_defaults": "binary active roll, binary explicit chord tones; separately labeled soft-template sensitivity",
            "not_implemented": "trained TCN/downbeat network, HHSMM/tempo dynamics, melody, or accompaniment",
        },
        "pardo": {
            "source": "Pardo and Birmingham 2002 Algorithms for Chordal Analysis, pp.28-31",
            "source_url": "https://interactiveaudiolab.github.io/assets/papers/pardo-birmingham-cmj02.pdf",
            "equation": "S=P-(M+N); note weight = number of note-event minimal segments spanned",
            "implemented": "published score, minimal-segment weights, optional root-weight tie rule",
            "adaptations": "causal sliding windows rather than complete segmentation; adds chord qualities absent from the native vocabulary while restricting candidates to this song's chart labels",
            "not_implemented": "CHORD global segmentation/optimization or Pardo lead-sheet sequence follower",
        },
        "accuracy": {
            "primary": "correct annotation among k tied top labels yields 1/k, otherwise 0; absent input yields 0",
            "provenance": "Pardo2002 ambiguity-credit rule adapted from minimal segments to common 100-ms frames",
            "secondary": "alphabetical-tie top1 reproduces previous-pilot convention; unique-correct and coverage also saved",
            "aggregation": "equal mean across three songs, preferred and alternate separate",
        },
        "controls": "cosine vs dot on same active88 or minimal-segment representation to isolate scoring effects",
        "alignment": "original MIDI zero origins, no fitted shifts; annotation/audio not independently reverified",
        "regression_checks_passed": regressions,
        "elapsed_offline_seconds_not_streaming_latency": time.perf_counter()-started,
        "songs": audit,
    }
    (OUT / "configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    (OUT / "inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE/"pardo_matcher.py",HERE/"yamaha_matcher.py",HERE/"run_pilot.py"]}
    (OUT / "code_hashes.json").write_text(json.dumps(source_hashes,indent=2)+"\n")
    create_report(rows, macro, audit)
    print(json.dumps({"report":str(OUT / "report.html"),"conditions":len(rows),"regression_checks":regressions,
                      "primary_preferred_pooled":[r for r in macro if r["take"]=="preferred" and r["input"]=="pooled"
                        and r["window_seconds"] in [0,2]]},indent=2))


def create_report(rows, macro, audit):
    main_methods = ["previous_cosine_onsets","yamaha_active_binary","yamaha_active_soft_sensitivity",
                    "pardo_segment_score","pardo_root_tiebreak"]
    primary = [r for r in macro if r["take"]=="preferred" and r["input"]=="pooled"
               and r["method"] in main_methods and r["window_seconds"] in (0,2)]
    main_table = [[r["method"],r["window_seconds"],pct(r["fractional_tie_accuracy"]),pct(r["alphabetical_tie_accuracy"]),
                   pct(r["coverage"]),pct(r["tie_rate_active"])] for r in primary]
    per_song = []
    for song in base.SONGS:
        selected = {r["method"]:r for r in rows if r["song"]==song and r["take"]=="preferred" and r["input"]=="pooled"
                    and r["window_seconds"] in (0,2)}
        per_song.append([song]+[pct(selected[m]["fractional_tie_accuracy"]) for m in main_methods])
    controls = [r for r in macro if r["take"]=="preferred" and r["input"]=="pooled" and r["window_seconds"] in (0,2)
                and r["method"] in ["yamaha_active_binary","cosine_active_control","pardo_segment_score","cosine_segment_control","dot_segment_control"]]
    windows = [r for r in macro if r["take"]=="preferred" and r["input"]=="pooled" and r["method"]=="pardo_segment_score"]
    doc = """<!doctype html><meta charset="utf-8"><title>Yamaha and Pardo harmonic-scoring pilot</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui;color:#20242a}h1{font-size:28px}h2{font-size:21px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}th{background:#f2f4f6}code{background:#f2f4f6}.notice{padding:14px 18px;background:#fff3dd;border-left:4px solid #986728}a{color:#245c91}</style>
<h1>Yamaha and Pardo: harmonic-scoring comparison</h1>
<p class="notice"><b>These are documented harmonic-component reconstructions/adaptations, not published full-system accuracy.</b> No Yamaha trained downbeat model, HHSMM, accompaniment, or Pardo sequence follower has been run. Three async songs, bass + saxophone + trumpet MIDI; piano and drums are absent. No model was trained.</p>
<h2>What was implemented</h2>
<p><b>Yamaha / Maezawa (2023), Section 2.2:</b> the harmonic log likelihood is the dot product of the current 88-note piano roll and a chord template, plus a written-melody term. We use the dot product directly, which preserves the exponential likelihood's ranking. Melody is unavailable and omitted. The paper does not specify numerical templates: binary chord membership and binary held notes are explicit reconstruction choices. The soft-template variant checks sensitivity to that missing detail, using the same heuristic as the previous pilot. We evaluate every 100 ms; the paper's timing-network input is 20 FPS.</p>
<p><b>Pardo and Birmingham (2002), CHORD segment labeling:</b> reward matching pitch classes, subtract weights of extra notes, and subtract one for each missing template pitch class. Note weights count how many minimal note-on/off segments they span, not seconds. The root-weight tie rule is reported separately. We adapt segment selection to past-only sliding windows. The shared chart vocabulary adds chord qualities absent from the original six families while restricting candidates to this song's chart labels. Global/offline segmentation and the distinct lead-sheet follower are not reproduced.</p>
<h2>Primary comparison: preferred takes, pooled instruments</h2>
<p>Equal mean across three songs. Accuracy uses Pardo's rule for ambiguous answers: when the correct chord is among k tied maxima, credit is 1/k. We adapt the evaluation unit to 100-ms time frames instead of minimal note-event segments. Empty observations receive zero credit. The previous alphabetical tie convention is also shown, making the earlier 26.0% result comparable. Different source-specific contexts are part of this first comparison; controlled comparisons follow.</p>
"""
    doc += html_table(["Method","Window seconds (0=current)","Tie-aware accuracy","Alphabetical top1","Coverage","Ties when active"],main_table)
    doc += "<p><b>Finding:</b> at the declared two-second setting, Pardo with root tie resolution essentially ties the earlier cosine baseline. The Yamaha-style scorer with chosen binary templates is lower; this is not evidence about the full published piano follower. In the exploratory window sweep, Pardo with root tie resolution reaches 31.1% at 0.5 seconds, versus 24.7% at two seconds. Window selection needs validation on other songs.</p>"
    doc += "<h2>Per-song results</h2>" + html_table(["Song"]+main_methods,per_song)
    doc += "<h2>Same-input controls</h2><p>These comparisons isolate scoring rules. The segment dot-product condition aggregates past events and must not be called the paper's instantaneous Yamaha model.</p>"
    doc += html_table(["Method","Representation","Tie-aware accuracy","Coverage"],[[r["method"],r["feature"],pct(r["fractional_tie_accuracy"]),pct(r["coverage"])] for r in controls])
    doc += "<h2>Pardo window sensitivity</h2><p>Exploratory windows fixed before this run; no held-out tuning or best-setting generalization claim.</p>"
    doc += html_table(["Window seconds","Tie-aware accuracy","Coverage"],[[r["window_seconds"],pct(r["fractional_tie_accuracy"]),pct(r["coverage"])] for r in windows])
    doc += "<h2>Why the vocabulary adaptation matters</h2><p>CHORD's original six chord families do not cover many written extensions or major/minor sevenths in these jazz charts. The table shows exact support under the original vocabulary. We retain the complete chart vocabulary for every main comparison rather than silently discard unsupported frames or simplify only one method's target.</p>"
    doc += html_table(["Song","Native 72-template coverage","Unsupported chart labels"],[[a["song"],pct(a["native_pardo72_label_coverage"]),", ".join(a["native_pardo72_unavailable_labels"]) or "none"] for a in audit])
    doc += """<h2>Interpretation limits</h2>
<ul><li>All three songs are asynchronous fixed-tempo overdubs; this does not test natural ensemble tempo adaptation.</li>
<li>Scores concern the chart chord, not a unique bar occurrence or full position following. Repeated chords remain ambiguous.</li>
<li>These are transcription-derived bass/horn inputs; the Yamaha demonstration used piano. Results should not be attributed to the published system's performance.</li>
<li>Annotation timing is used only for scoring. Original MIDI time origins are retained, with no fitted time shift. No independent audio-versus-label verification was performed.</li>
<li>Preferred takes are primary. Alternate take results use the release's documented shared async annotation grid and are saved separately.</li>
<li>Three songs are an exploratory sample. No confidence calibration, learned weighting, streaming-latency claim, or held-out generalization result is established.</li></ul>
<h2>Files and reproduction</h2><p>Run <code>python run_published_matchers.py</code> with NumPy installed. Original pilot files and dataset remain unchanged.</p>
<ul><li><a href="metrics.csv">All metrics</a></li><li><a href="macro.csv">Song-averaged results</a></li><li><a href="by_section.csv">Section breakdown</a></li><li><a href="predictions.csv">Predictions at every timestamp</a></li><li><a href="configuration.json">Exact adaptations and evaluation definitions</a></li><li><a href="inputs.json">Input hashes and MIDI audit</a></li></ul>
<p>Sources: <a href="https://ismir2023program.ismir.net/lbd_365.html">Maezawa 2023</a>; <a href="https://interactiveaudiolab.github.io/assets/papers/pardo-birmingham-cmj02.pdf">Pardo and Birmingham 2002</a>; <a href="https://zenodo.org/records/22943963">JazzSAMBA</a>.</p>
"""
    (OUT/"report.html").write_text(doc)


if __name__ == "__main__":
    main()
