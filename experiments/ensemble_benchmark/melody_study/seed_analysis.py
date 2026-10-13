"""Descriptive seed sensitivity with harmonic controls matched within each seed.

This reports every fixed candidate. It never chooses a seed or system per song.
Repeated seeds measure Monte Carlo sensitivity, not independent musical tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import analysis_results as analysis

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = ("01_past_all", "02_written", "03_past_seed1", "04_past_seed2",
                "05_written_seed1", "06_written_seed2")
GAIN_METRICS = ("mean_bar_gain", "mean_within2beat_gain", "mean_error_change", "mean_median_error_change")


def stats(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def summarize_seeds(rows):
    result = []
    for key, values in sorted(analysis.group(rows, analysis.GROUP).items()):
        seeds = sorted(r["seed"] for r in values)
        if len(seeds) != len(set(seeds)) or len({r["performances"] for r in values}) != 1:
            raise ValueError("Repeated seed or inconsistent cohort size")
        metrics = (*analysis.METRICS, "melodic_coverage", *GAIN_METRICS)
        result.append({**dict(zip(analysis.GROUP, key)), "seeds": seeds,
                       "performances_per_seed": values[0]["performances"],
                       **{name: stats([r[name] for r in values]) for name in metrics},
                       "seeds_with_positive_bar_gain": sum(r["mean_bar_gain"] > 1e-12 for r in values),
                       "seeds_with_negative_bar_gain": sum(r["mean_bar_gain"] < -1e-12 for r in values)})
    return result


def flatten(rows):
    output = []
    for row in rows:
        value = {}
        for key, item in row.items():
            if isinstance(item, dict):
                value.update({f"{key}_{stat}": number for stat, number in item.items()})
            elif isinstance(item, list):
                value[key] = ",".join(str(x) for x in item)
            else:
                value[key] = item
        output.append(value)
    return output


def per_song_consistency(rows):
    result = []
    keys = (*analysis.GROUP, "performance_id", "song")
    for key, values in sorted(analysis.group(rows, keys).items()):
        result.append({**dict(zip(keys, key)), "seeds": sorted(r["seed"] for r in values),
                       "bar_accuracy": stats([r["bar_accuracy"] for r in values]),
                       "bar_gain": stats([r["bar_gain"] for r in values]),
                       "mean_abs_beats": stats([r["mean_abs_beats"] for r in values]),
                       "mean_error_change": stats([r["mean_error_change"] for r in values]),
                       "positive_seeds": sum(r["bar_gain"] > 1e-12 for r in values),
                       "negative_seeds": sum(r["bar_gain"] < -1e-12 for r in values)})
    return result


def ranking(rows):
    result = []
    selected = [r for r in rows if "__fused" in r["method"]]
    for _, values in sorted(analysis.group(selected, ("reference", "protocol", "scope")).items()):
        ordered = sorted(values, key=lambda r: (-r["bar_accuracy"]["mean"], r["method"]))
        result.extend({**row, "rank": index} for index, row in enumerate(ordered, 1))
    return result


def markdown(report):
    lines = ["# Melody comparison: seed sensitivity", "",
             "All fixed methods are repeated with seeds 0, 1, and 2. Each gain uses the harmonic-only result for the same seed, performance, and evaluation scope. Means/ranges describe these three complete runs; they are not confidence intervals or independent test samples.", "",
             "## Highest-ranked fixed fusion by mean bar accuracy", "",
             "The setting is ranked across the full cohort and all three seeds, never chosen separately for an individual song. Both strengths and all five matchers remain in the full CSVs.", ""]
    best = [r for r in report["fusion_rankings"] if r["rank"] == 1]
    def percentage(value):
        return f'{100*value["mean"]:.3f} [{100*value["min"]:.3f}, {100*value["max"]:.3f}]'
    lines += analysis.table(["Reference", "Protocol", "Scope", "N", "Fixed fusion", "Bar mean [range] %",
                             "Gain mean [range] pp", "Seeds with positive gain"],
                            [[r["reference"], r["protocol"], r["scope"], r["performances_per_seed"], r["method"],
                              percentage(r["bar_accuracy"]), percentage(r["mean_bar_gain"]),
                              f'{r["seeds_with_positive_bar_gain"]}/{len(r["seeds"])}'] for r in best])
    lines += ["", "## All-frame fusion results", ""]
    values = [r for r in report["seed_summary"] if r["scope"] == "all" and "__fused" in r["method"]]
    lines += analysis.table(["Reference", "Protocol", "Fixed fusion", "Bar mean [range] %",
                             "Gain mean [range] pp", "Mean error [range] beats"],
                            [[r["reference"], r["protocol"], r["method"], percentage(r["bar_accuracy"]),
                              percentage(r["mean_bar_gain"]),
                              f'{r["mean_abs_beats"]["mean"]:.3f} [{r["mean_abs_beats"]["min"]:.3f}, {r["mean_abs_beats"]["max"]:.3f}]']
                             for r in values])
    lines += ["", "## Interpretation", "",
              "- A positive mean gain whose range crosses zero did not improve the macro bar metric in every tested seed. Stable macro signs still do not guarantee consistent gains for individual songs.",
              "- Mean absolute beat error and within-two-beat accuracy are retained alongside bar accuracy in `seed_summary.csv`; a higher bar percentage can coexist with more severe failures.",
              "- The written-minus-past seed comparison restricts both references to the identical written cohort and pairs within seed. It does not compare six written performances against all 75 past-memory performances.",
              "- Matcher availability itself can vary by seed because the independent harmonic anchor controls memory labels and planned-role masks. The uncertainty therefore includes both the anchor and evaluated particle filter.",
              "- No seed was chosen per song. Neither repeated seeds nor retrospective fixed-setting ranking establish held-out generalization, calibrated lick detection, or statistical significance.",
              "- The original analysis limits remain: no certified lick labels, uncalibrated matching thresholds/rewards, variable-duration nine-note contexts, known unfolded plan/start, and unresolved independent AMT/audio/annotation alignment.", "",
              "`seed_analysis.json` contains per-seed macros and matched-control gains, complete mean/range summaries, paired reference comparisons, per-song consistency, and provenance. CSV files preserve all methods, including harmonic controls and melody-only observation ablations. Pure melody ablations are excluded from fusion rankings.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path,
                        default=[HERE / "runs" / name for name in DEFAULT_RUNS])
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    by_run = {}
    configs = {}
    provenance = []
    # No outputs are created unless every requested run has completed and passed
    # the same validations as the main analysis.
    for path in args.runs:
        config = json.loads((path / "run.json").read_text())
        reference = config["reference"]
        rows, config, evidence = analysis.load_completed_run(path, reference)
        key = (reference, config["seed"])
        if key in by_run:
            raise ValueError("More than one run for a reference/seed")
        by_run[key], configs[key] = rows, config
        provenance.append(evidence)
    reference_seeds = {reference: sorted(seed for ref, seed in by_run if ref == reference)
                       for reference in {key[0] for key in by_run}}
    if set(reference_seeds) != {"past", "written"} or reference_seeds["past"] != reference_seeds["written"]:
        raise ValueError("Require the same seeds for past and written references")
    first = next(iter(configs.values()))
    for config in configs.values():
        for name in ("particle_config", "phrase_parameters", "melody_strengths", "matchers", "sources"):
            if config[name] != first[name]:
                raise ValueError(f"Configurations differ beyond seed/reference: {name}")
    for reference, seeds in reference_seeds.items():
        cohorts = [set(configs[(reference, seed)]["performance_ids"]) for seed in seeds]
        if any(cohort != cohorts[0] for cohort in cohorts):
            raise ValueError("Reference cohort changes between seeds")
    per_seed, song_pairs = [], []
    for (reference, seed), rows in sorted(by_run.items()):
        macros = analysis.aggregate(rows)
        pairs = analysis.paired_rows(rows)
        gains = {tuple(r[k] for k in analysis.GROUP): r for r in analysis.summarize_pairs(pairs)}
        for row in macros:
            gain = gains[tuple(row[k] for k in analysis.GROUP)]
            per_seed.append({**row, "seed": seed,
                             **{name: gain[name] for name in (*GAIN_METRICS, "improved", "worse", "tied")}})
        song_pairs.extend(pairs)
    matched_seed_metrics, matched_seed_differences = [], []
    for seed in reference_seeds["past"]:
        metrics, differences, ids = analysis.matched_references(by_run[("past", seed)], by_run[("written", seed)])
        matched_seed_metrics.extend({**r, "seed": seed} for r in metrics)
        matched_seed_differences.extend({**r, "seed": seed} for r in differences)
    # Pair summaries already contain differences, so their seed ranges do not
    # mix musical-composition differences with reference effects.
    matched_summary = []
    for key, rows in sorted(analysis.group(matched_seed_differences, analysis.GROUP).items()):
        matched_summary.append({**dict(zip(analysis.GROUP, key)),
                                "comparison": "written_minus_past_same_performances_and_seed",
                                "performances_per_seed": rows[0]["performances"],
                                "seeds": sorted(r["seed"] for r in rows),
                                **{name: stats([r[name] for r in rows]) for name in GAIN_METRICS}})
    summary = summarize_seeds(per_seed)
    consistency = per_song_consistency(song_pairs)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_source_sha256": analysis.sha256(Path(__file__)),
              "analysis_helper_source_sha256": analysis.sha256(Path(analysis.__file__)),
              "provenance": provenance, "reference_seeds": reference_seeds,
              "paired_baseline": "harmonic_only from the same reference run, seed, performance and scope",
              "aggregation": "Equal-performance macro per protocol/reference/scope, followed by descriptive mean/min/max across complete seeds",
              "per_seed_metrics": per_seed, "seed_summary": summary,
              "fusion_rankings": ranking(summary), "per_song_seed_consistency": consistency,
              "matched_reference_seed_metrics": matched_seed_metrics,
              "matched_reference_seed_differences": matched_seed_differences,
              "matched_reference_seed_summary": matched_summary}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_seed_metrics.csv", per_seed), ("seed_summary.csv", flatten(summary)),
                       ("seed_fusion_rankings.csv", flatten(report["fusion_rankings"])),
                       ("per_song_seed_consistency.csv", flatten(consistency)),
                       ("matched_reference_seed_metrics.csv", matched_seed_metrics),
                       ("matched_reference_seed_differences.csv", matched_seed_differences),
                       ("matched_reference_seed_summary.csv", flatten(matched_summary))):
        analysis.write_csv(out / name, rows)
    (out / "seed_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "seed_analysis.md").write_text(markdown(report))
    print(json.dumps({"output": str(out), "reference_seeds": reference_seeds,
                      "per_seed_metric_rows": len(per_seed), "seed_summary_rows": len(summary),
                      "same_seed_controls_verified": True, "matched_reference_controls_verified": True}, indent=2))


if __name__ == "__main__":
    main()
