"""Analyze completed melody runs without changing inference or selecting by song.

All metrics are equal-performance macro means within reference/protocol/scope.
The written-versus-past comparison uses the identical written-eligible cohort.
Fusion rankings contain only fixed fused settings; melody-only observation
ablations are retained in the full tables, never relabeled as fusion systems.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats")
GROUP = ("reference", "protocol", "scope", "method")
SCOPES = ("all", "head_in", "head_out", "solo")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def load_completed_run(path, reference):
    path = path.resolve()
    if not (path / "summary.json").exists():
        raise ValueError(f"Run has no completion summary yet: {path}")
    run = json.loads((path / "run.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if not summary.get("complete") or not summary.get("scope_finished") or run.get("is_smoke"):
        raise ValueError(f"Only complete non-smoke runs can be analyzed: {path}")
    for key in ("failures", "source_changes", "input_changes"):
        if summary.get(key):
            raise ValueError(f"Run reports {key}: {path}")
    if run["reference"] != reference:
        raise ValueError(f"Expected {reference} references: {path}")
    with (path / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary["metric_rows"]:
        raise ValueError(f"Metric-row count differs from completion summary: {path}")
    for row in rows:
        row["frames"], row["seed"] = int(row["frames"]), int(row["seed"])
        for key in (*METRICS, "melodic_coverage", "matching_seconds", "inference_seconds"):
            row[key] = float(row[key])
        if row["reference"] != reference or row["seed"] != run["seed"] or row["frames"] <= 0:
            raise ValueError("Reference, seed, or frame count disagrees with run configuration")
        if not np.isfinite([row[k] for k in (*METRICS, "melodic_coverage")]).all():
            raise ValueError("Nonfinite evaluation result")
    identities = [(r["performance_id"], r["scope"], r["method"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated performance/scope/method rows")
    if {r["performance_id"] for r in rows} != set(run["performance_ids"]):
        raise ValueError("Observed performance cohort differs from the declared run")
    expected = {"harmonic_only", "no_observations"} | {
        f"{matcher}__{condition}" for matcher in run["matchers"]
        for condition in ("melodic_updates", "fused1", "fused3")}
    for (_, scope), values in group(rows, ("performance_id", "scope")).items():
        if scope not in SCOPES or {r["method"] for r in values} != expected:
            raise ValueError("Incomplete method coverage in a performance scope")
        if len({r["frames"] for r in values}) != 1:
            raise ValueError("Methods use different evaluation frames in a scope")
    all_rows = [r for r in rows if r["scope"] == "all"]
    if len(all_rows) != len(run["performance_ids"]) * run["expected_methods"]:
        raise ValueError("Missing all-frame controls or candidates")
    # Verify the immutable snapshots; the completion summary separately records
    # the runner's check that its input files did not change during inference.
    for index, (source, digest) in enumerate(run["sources"].items()):
        snapshot = path / "sources" / f"{index:02d}_{Path(source).name}"
        if sha256(snapshot) != digest:
            raise ValueError(f"Source snapshot hash mismatch: {snapshot}")
    audits = json.loads((path / "audits.json").read_text())
    if len(audits) != len(run["performance_ids"]) or not all(a["neutral_reproduction"] for a in audits):
        raise ValueError("Missing neutral-reproduction audit")
    provenance = {"path": str(path), "reference": reference, "performances": len(run["performance_ids"]),
                  "metric_rows": len(rows), "seed": run["seed"], "complete": True,
                  "source_snapshots_verified": True,
                  "real_performance_prefix_checks": sum(a["prefix_verified"] for a in audits),
                  "sha256": {name: sha256(path / name) for name in
                             ("run.json", "summary.json", "metrics.csv", "audits.json")}}
    return rows, run, provenance


def aggregate(rows):
    result = []
    for key, values in sorted(group(rows, GROUP).items()):
        result.append({**dict(zip(GROUP, key)), "performances": len(values),
                       "total_frames": sum(r["frames"] for r in values),
                       **{k: float(np.mean([r[k] for r in values])) for k in METRICS},
                       "melodic_coverage": float(np.mean([r["melodic_coverage"] for r in values]))})
    return result


def paired_rows(rows):
    baselines = {(r["reference"], r["performance_id"], r["scope"]): r
                 for r in rows if r["method"] == "harmonic_only"}
    result = []
    for row in rows:
        baseline = baselines[(row["reference"], row["performance_id"], row["scope"])]
        if row["frames"] != baseline["frames"]:
            raise ValueError("Unmatched harmonic-control frame counts")
        result.append({**row, "bar_gain": row["bar_accuracy"] - baseline["bar_accuracy"],
                       "within2beat_gain": row["within2beat_accuracy"] - baseline["within2beat_accuracy"],
                       "mean_error_change": row["mean_abs_beats"] - baseline["mean_abs_beats"],
                       "median_error_change": row["median_abs_beats"] - baseline["median_abs_beats"]})
    return result


def summarize_pairs(rows):
    result = []
    for key, values in sorted(group(rows, GROUP).items()):
        gains = np.array([r["bar_gain"] for r in values])
        result.append({**dict(zip(GROUP, key)), "performances": len(values),
                       "mean_bar_gain": float(gains.mean()),
                       "mean_within2beat_gain": float(np.mean([r["within2beat_gain"] for r in values])),
                       "mean_error_change": float(np.mean([r["mean_error_change"] for r in values])),
                       "mean_median_error_change": float(np.mean([r["median_error_change"] for r in values])),
                       "improved": int((gains > 1e-12).sum()), "worse": int((gains < -1e-12).sum()),
                       "tied": int((np.abs(gains) <= 1e-12).sum())})
    return result


def coverage(rows):
    unique = {}
    for row in rows:
        if row["matcher"] == "none":
            continue
        key = (row["reference"], row["performance_id"], row["scope"], row["matcher"])
        if key in unique and row["melodic_coverage"] != unique[key]["melodic_coverage"]:
            raise ValueError("Matcher availability differs across observation-strength ablations")
        unique[key] = row
    result = []
    keys = ("reference", "protocol", "scope", "matcher")
    for key, values in sorted(group(unique.values(), keys).items()):
        result.append({**dict(zip(keys, key)), "performances": len(values),
                       "macro_coverage": float(np.mean([r["melodic_coverage"] for r in values])),
                       "frame_weighted_coverage": float(np.average([r["melodic_coverage"] for r in values],
                                                                  weights=[r["frames"] for r in values])),
                       "performances_with_any_evidence": sum(r["melodic_coverage"] > 0 for r in values),
                       "performances_with_zero_evidence": sum(r["melodic_coverage"] == 0 for r in values)})
    return result


def matched_references(past, written):
    ids = {r["performance_id"] for r in written}
    if not ids <= {r["performance_id"] for r in past}:
        raise ValueError("Written cohort is not contained in past-memory cohort")
    matched_past = [r for r in past if r["performance_id"] in ids]
    old = {(r["performance_id"], r["scope"], r["method"]): r for r in matched_past}
    if set(old) != {(r["performance_id"], r["scope"], r["method"]) for r in written}:
        raise ValueError("Written/past scopes or candidates do not match exactly")
    differences = []
    for row in written:
        previous = old[(row["performance_id"], row["scope"], row["method"])]
        if row["frames"] != previous["frames"]:
            raise ValueError("Written/past evaluation frame counts differ")
        if row["method"] in ("harmonic_only", "no_observations"):
            if not np.allclose([row[k] for k in METRICS], [previous[k] for k in METRICS], rtol=0, atol=1e-12):
                raise ValueError("Reference-independent controls differ on the matching cohort")
        differences.append({**row, "bar_gain": row["bar_accuracy"] - previous["bar_accuracy"],
                            "within2beat_gain": row["within2beat_accuracy"] - previous["within2beat_accuracy"],
                            "mean_error_change": row["mean_abs_beats"] - previous["mean_abs_beats"],
                            "median_error_change": row["median_abs_beats"] - previous["median_abs_beats"]})
    comparison = summarize_pairs(differences)
    for row in comparison:
        row["comparison"] = "written_minus_past_same_performances"
    return aggregate(matched_past + written), comparison, sorted(ids)


def fusion_rankings(macros, gains):
    gain_lookup = {tuple(r[k] for k in GROUP): r for r in gains}
    result = []
    fused = [r for r in macros if "__fused" in r["method"]]
    for _, values in sorted(group(fused, ("reference", "protocol", "scope")).items()):
        ordered = sorted(values, key=lambda r: (-r["bar_accuracy"], -r["within2beat_accuracy"], r["method"]))
        for rank, row in enumerate(ordered, 1):
            pair = gain_lookup[tuple(row[k] for k in GROUP)]
            result.append({**row, "rank": rank,
                           **{k: pair[k] for k in ("mean_bar_gain", "mean_error_change", "improved", "worse", "tied")}})
    return result


def extremes(pairs):
    result = []
    # These are individual failures of fixed systems, never per-song choices of
    # which system to deploy. Restrict examples to all-frame fused evaluation.
    selected = [r for r in pairs if r["scope"] == "all" and "__fused" in r["method"]]
    fields = ("performance_id", "song", *METRICS, "bar_gain", "mean_error_change", "melodic_coverage")
    for key, values in sorted(group(selected, GROUP).items()):
        def brief(row):
            return {k: row[k] for k in fields}
        result.append({**dict(zip(GROUP, key)),
                       "largest_gains": [brief(r) for r in sorted(values, key=lambda r: -r["bar_gain"])[:3]],
                       "largest_losses": [brief(r) for r in sorted(values, key=lambda r: r["bar_gain"])[:3]],
                       "largest_absolute_errors": [brief(r) for r in sorted(values, key=lambda r: -r["mean_abs_beats"])[:3]]})
    return result


def negative_findings(rankings, gains, coverage_rows):
    result = []
    for best in [r for r in rankings if r["rank"] == 1]:
        key = {k: best[k] for k in ("reference", "protocol", "scope")}
        if best["mean_bar_gain"] <= 0:
            result.append({**key, "finding": "No fixed fusion candidate improves macro bar accuracy over harmonic_only",
                           "highest_ranked_fusion": best["method"], "mean_bar_gain": best["mean_bar_gain"]})
        if best["mean_error_change"] > 0:
            result.append({**key, "finding": "Highest bar-accuracy fusion increases mean absolute beat error",
                           "highest_ranked_fusion": best["method"], "mean_error_change": best["mean_error_change"]})
    for row in coverage_rows:
        if row["performances_with_zero_evidence"]:
            result.append({**row, "finding": "Some performances have no available melodic potential in this scope"})
    for row in gains:
        if row["scope"] == "all" and "__fused" in row["method"] and row["worse"] > row["improved"]:
            result.append({**row, "finding": "Fusion worsens bar accuracy on more performances than it improves"})
    return result


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in rows]


def markdown(report):
    lines = ["# Melody matching analysis", "",
             "These are exploratory, fixed-setting comparisons of nine-onset matching adaptations. No per-song best system is reported. Protocols and section scopes remain separate; all accuracy summaries weight performances equally.", "",
             "Methods are pitch-class cosine, pitch-class edit distance, signed-interval edit distance, interval/rhythm DTW, and interval/rhythm n-gram overlap. They are simplified algorithm-family adaptations, not complete published-system reproductions. The input has no certified lick annotations: this evaluates phrase/repetition matching, not labeled lick recognition.", "",
             "## Highest-ranked fixed fusion by protocol and scope", "",
             "Rankings include only fused variants. The table reports the highest macro bar-accuracy setting within each declared condition, selected retrospectively across that cohort. Full rankings retain both fixed strengths and all methods in `fusion_rankings.csv`. This is exploratory ranking, not held-out model selection.", ""]
    best = [r for r in report["fusion_rankings"] if r["rank"] == 1]
    lines += table(["Reference", "Protocol", "Scope", "N", "Fixed fusion", "Bar %", "Gain pp", "Within 2 beats %", "Mean error", "Improved / worse"],
                   [[r["reference"], r["protocol"], r["scope"], r["performances"], r["method"],
                     f'{100*r["bar_accuracy"]:.3f}', f'{100*r["mean_bar_gain"]:+.3f}',
                     f'{100*r["within2beat_accuracy"]:.3f}', f'{r["mean_abs_beats"]:.3f}',
                     f'{r["improved"]} / {r["worse"]}'] for r in best])
    lines += ["", "## Harmonic controls and melody-only observation ablations", "",
              "The ablation removes harmonic updates from the evaluated particle filter. Past-memory position tags and all planned-role masks still come from the independent harmonic anchor. It is not a standalone melody-only cold-start system. The complete scope tables are in `aggregate_metrics.csv`.", ""]
    ablations = [r for r in report["aggregate_metrics"] if r["scope"] == "all" and "__fused" not in r["method"]]
    lines += table(["Reference", "Protocol", "Method", "N", "Bar %", "Within 2 beats %", "Mean error"],
                   [[r["reference"], r["protocol"], r["method"], r["performances"], f'{100*r["bar_accuracy"]:.3f}',
                     f'{100*r["within2beat_accuracy"]:.3f}', f'{r["mean_abs_beats"]:.3f}'] for r in ablations])
    lines += ["", "## Written versus past memory: identical cohort", "",
              f'The written comparison contains {len(report["written_cohort_performance_ids"])} performances. Past-memory rows are restricted to exactly those IDs, with identical scope frame counts. Harmonic-only and no-observation control metrics agree to 1e-12. Full values and paired written-minus-past changes are in `matched_reference_metrics.csv` and `matched_reference_differences.csv`; the larger past-memory cohort is not its comparator.', ""]
    matched = [r for r in report["matched_reference_differences"] if r["scope"] == "all" and "__fused" in r["method"]]
    lines += table(["Protocol", "Method", "N", "Written − past bar pp", "Mean-error change", "Improved / worse"],
                   [[r["protocol"], r["method"], r["performances"], f'{100*r["mean_bar_gain"]:+.3f}',
                     f'{r["mean_error_change"]:+.3f}', f'{r["improved"]} / {r["worse"]}'] for r in matched])
    lines += ["", "## Availability and negative findings", "",
              "Coverage is the fraction of evaluation frames with a retained, fresh reference match that produces a potential. It is not correct-match accuracy, cue reliability, or the fraction of particles receiving material evidence. The same matcher has identical coverage across its ablation and two fusion strengths.", ""]
    lines += table(["Reference", "Protocol", "Matcher", "N", "Macro coverage %", "No-evidence performances"],
                   [[r["reference"], r["protocol"], r["matcher"], r["performances"], f'{100*r["macro_coverage"]:.3f}',
                     r["performances_with_zero_evidence"]] for r in report["coverage"] if r["scope"] == "all"])
    visible = [r for r in report["negative_findings"] if r.get("scope") == "all" and
               "highest_ranked_fusion" in r]
    lines += [""] + [f'- {r["reference"]}, {r["protocol"]}: {r["finding"]} ({r["highest_ranked_fusion"]}).' for r in visible]
    lines += ["", "All per-song paired changes are retained in `per_song_gains.csv`. `analysis.json` includes the three largest gains, losses, and absolute errors for every fixed all-frame fusion. Improvements in macro bar accuracy can coexist with individual catastrophic failures; those examples are not combined into an oracle system.", "",
              "## Interpretation limits", ""]
    lines += ["- " + text for text in report["limits"]]
    lines += ["", "## Reproduction", "", "Run `analysis_results.py` only after both run summaries report completion. The script rejects missing methods, duplicate rows, mismatched frame counts, changed source snapshots, or unequal reference-independent controls on the written cohort. Analysis outputs never alter inference settings, source snapshots, or run metrics.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--past-run", type=Path, default=HERE / "runs" / "01_past_all")
    parser.add_argument("--written-run", type=Path, default=HERE / "runs" / "02_written")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    # Finish all validation before creating outputs; partial runs produce none.
    past, past_config, past_provenance = load_completed_run(args.past_run, "past")
    written, written_config, written_provenance = load_completed_run(args.written_run, "written")
    for key in ("seed", "particle_config", "phrase_parameters", "melody_strengths", "matchers", "sources"):
        if past_config[key] != written_config[key]:
            raise ValueError(f"Reference runs have different {key}; separate this confound before comparison")
    rows = past + written
    macros = aggregate(rows)
    pairs = paired_rows(rows)
    gains = summarize_pairs(pairs)
    rankings = fusion_rankings(macros, gains)
    cov = coverage(rows)
    matched, differences, ids = matched_references(past, written)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_source_sha256": sha256(Path(__file__)),
              "provenance": [past_provenance, written_provenance],
              "aggregation": "Equal-performance macro means, separate reference/protocol/scope. Mean song median is not a pooled median.",
              "ranking": "Fixed fused settings only; no per-song oracle choices, no significance or external-validation claim.",
              "aggregate_metrics": macros, "paired_gains": gains, "fusion_rankings": rankings,
              "matched_reference_metrics": matched, "matched_reference_differences": differences,
              "written_cohort_performance_ids": ids, "coverage": cov, "extremes": extremes(pairs),
              "negative_findings": negative_findings(rankings, gains, cov),
              "limits": [
                  "The corpus and algorithms were inspected during exploratory development. These are not untouched external-test results; no statistical-significance claim is made.",
                  "The written cohort is limited by available usable melody files. Its small, selected composition set cannot establish general superiority over past memory.",
                  "There are no certified lick labels. Repetition matching can match heads, accompaniment fragments, transpositions, or accidental short similarities; it does not demonstrate recognition of named jazz licks.",
                  "All matchers use nine-note contexts of variable elapsed duration, with fixed nominal-beat span/gap limits. Timing normalization and feature invariances differ between algorithms.",
                  "The similarity threshold and positive potential are heuristic and uncalibrated across algorithms. Median contrast begins at eight eligible windows, which may overlap and need not represent independent locations. Report coverage alongside accuracy.",
                  "The positive reward leaves uncovered locations neutral. Repeated written measure IDs expand a hit to every matching occurrence, preserving rather than solving chorus ambiguity.",
                  "Past phrases use historical position labels from an independent harmonic follower. Its errors can contaminate memory; removing harmonic updates from the evaluated filter does not remove this harmonic memory dependency.",
                  "Known unfolded measure order, starting position/time, catalog BPM, lead-instrument flags, and planned solo roles are supplied. Solo masks use estimated positions; timed section annotations are used only to evaluate scopes.",
                  "Head/solo scopes may omit performances with no frames in that section; each table gives its actual cohort size. Scope results are diagnostic subdivisions, not separately tuned systems.",
                  "Input is offline AMT-derived MIDI. Audio/MIDI/annotation time-origin alignment remains independently unverified; beat targets are interpolated inside annotated bars.",
                  "Matching and inference timings are batch measurements with shared preprocessing and precomputed controls. They are not live audio-to-decision latency comparisons.",
                  "These primary tables summarize one fixed particle seed. Small gains may be sensitive to resampling; the separate seed_analysis outputs report all repeated conditions with same-seed harmonic controls."]}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in (("aggregate_metrics.csv", macros), ("paired_gains.csv", gains),
                             ("fusion_rankings.csv", rankings), ("matched_reference_metrics.csv", matched),
                             ("matched_reference_differences.csv", differences), ("per_song_gains.csv", pairs),
                             ("coverage.csv", cov)):
        write_csv(out / filename, values)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "analysis.md").write_text(markdown(report))
    print(json.dumps({"output": str(out), "past_performances": past_provenance["performances"],
                      "written_performances": written_provenance["performances"],
                      "metric_rows": len(rows), "aggregate_rows": len(macros),
                      "matched_control_verification": True}, indent=2))


if __name__ == "__main__":
    main()
