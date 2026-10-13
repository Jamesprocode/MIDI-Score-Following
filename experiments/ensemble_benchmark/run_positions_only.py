"""Recompute position results without touching saved chord metrics.

Default scope matches run_benchmark: both schemes/all eligible performances at
factor 1, plus triad/preferred factors .9 and 1.1. Only pooled, raw MIDI and the
one-second harmonic features are computed; Yamaha uses instantaneous activity.
Workers receive precomputed scores, never the profile defaultdicts/lambdas.

Example (full run is deliberate; use a separate output directory for a smoke):
  python run_positions_only.py --limit 1 --workers 3 --output smoke_positions
  python run_positions_only.py --workers 3 --output results
  python run_positions_only.py --broad-tempo --workers 3 --output results

The last command writes separate broad-tempo files and trace directory. It does
not modify primary metrics, traces, config, summary or code-hash records.
Checkpoints preserve completed jobs; this script starts a new run rather than
implicitly resuming a potentially incompatible checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import run_benchmark as benchmark

HERE = Path(__file__).resolve().parent
BROAD_TEMPO_RATIOS = tuple(round(.5 + .1 * i, 10) for i in range(14))
METHODS = ("cosine_onsets", "cosine_duration", "pardo", "yamaha_active", "empirical_profile")
DATA_KEYS = ("times", "bar_starts_beats", "chord_starts_beats", "chord_ends_beats",
             "initial_bpm", "start_time", "score_end_beats", "truth_position_beats", "truth_bar_index")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    if rows:
        benchmark.csv_write(temporary, rows)
    else:
        fields = ["performance_id", "song", "protocol", "take", "composition", "scheme",
                  "initial_tempo_factor", "method", "window_seconds", "frames", "bar_accuracy",
                  "within2beat_accuracy", "mean_abs_beats", "median_abs_beats"]
        temporary.write_text(",".join(fields) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def source_hashes():
    modules = (benchmark, benchmark.common, benchmark.follower,
               benchmark.old, benchmark.pardo, benchmark.yamaha)
    paths = {Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules)}
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def job_specs(entry, broad_tempo):
    if broad_tempo:
        return [("triad", 1.)] if entry["take"] == "preferred" else []
    cases = [("triad", 1.), ("seventh", 1.)]
    if entry["take"] == "preferred":
        cases += [("triad", .9), ("triad", 1.1)]
    return cases


def precomputed_jobs(loaded, profiles, compositions, config, trace_destination,
                     broad_tempo, profile_audit):
    """Build one performance's features once, then yield picklable worker jobs."""
    window = benchmark.POSITION_WINDOW
    for entry, data in loaded:
        specs = job_specs(entry, broad_tempo)
        if not specs:
            continue
        composition = compositions[entry["song"]]
        notes = benchmark.selected_notes(data["notes_by_instrument"], "pooled")
        times = data["times"]
        onsets = benchmark.old.features(notes, times, window, "onsets")
        duration = benchmark.old.features(notes, times, window, "elapsed_duration")
        segments = benchmark.pardo.segment_profiles(notes, times, window)
        active = benchmark.yamaha.piano_roll88(notes, times)
        available_onsets = onsets.sum(axis=1) > 0
        available_duration = duration.sum(axis=1) > 0
        available_segments = segments.sum(axis=1) > 0
        available_active = active.sum(axis=1) > 0
        cases = {}
        for scheme in dict.fromkeys(s for s, _ in specs):
            targets = benchmark.make_targets(data, scheme)
            templates = targets["templates"]
            log_profiles, unseen = benchmark.empirical_templates(
                targets, scheme, "pooled", composition, profiles)
            profile_audit.append({
                "performance_id": entry["performance_id"], "scheme": scheme,
                "input": "pooled", "excluded_composition": composition,
                "unseen_qualities": [label for label, missing in zip(targets["labels"], unseen) if missing],
            })
            scores = {
                "cosine_onsets": (benchmark.old.cosine_scores(onsets, templates), available_onsets),
                "cosine_duration": (benchmark.old.cosine_scores(duration, templates), available_duration),
                "pardo": (benchmark.pardo.score(segments, templates), available_segments),
                "yamaha_active": (benchmark.yamaha.harmonic_log_scores(
                    active, benchmark.yamaha.expand_chroma_templates(templates)), available_active),
                "empirical_profile": (benchmark.old.normalize_rows(onsets) @ log_profiles.T, available_onsets),
            }
            cases[scheme] = ({"scheme": scheme, "vocab": targets["vocab"], "ids": targets["ids"]}, scores)
        worker_entry = {key: entry[key] for key in ("performance_id", "song", "protocol", "take")}
        worker_data = {key: data[key] for key in DATA_KEYS}
        for scheme, factor in specs:
            targets, scores = cases[scheme]
            yield {
                "job_id": f"{entry['performance_id']}|{scheme}|{factor:.6g}",
                "entry": worker_entry, "data": worker_data, "targets": targets,
                "scores_by_method": scores, "config": config, "factors": [factor],
                "destination": trace_destination, "composition": composition,
            }


def position_worker(job):
    """Top-level spawn-compatible callable: no data-trained objects in payload."""
    result = benchmark.run_positions(
        job["entry"], job["data"], job["targets"], job["scores_by_method"],
        job["config"], job["factors"], job["destination"], job["composition"])
    return job["job_id"], result


def update_main_metadata(out, summary, metadata, hashes, *, final=False):
    """Update position fields while retaining chord results and their provenance."""
    combined = read_json(out / "summary.json")
    combined.update({
        "position_complete": summary["complete"], "position_status": summary["status"],
        "position_metric_rows": summary["metric_rows"],
        "position_performances": summary["performances_completed"],
        "position_runtime_seconds": summary["runtime_seconds"],
        "position_runner": "run_positions_only.py", "position_summary_file": "position_summary.json",
    })
    combined["complete"] = bool(combined.get("chord_complete", False)) and summary["complete"]
    atomic_json(out / "summary.json", combined)
    if final:
        config = read_json(out / "config.json")
        config.update({
            "position_config": metadata["filter_overrides"],
            "position_default_config": metadata["filter_defaults"],
            "position_effective_config": metadata["filter_effective"],
            "position_run_metadata": "position_config.json",
            "position_code_hashes_file": "position_code_hashes.json",
            "position_window_seconds": metadata["position_window_seconds"],
            "position_frame_seconds": metadata["position_frame_seconds"],
            "position_status": summary["status"],
        })
        atomic_json(out / "config.json", config)
        # Preserve the previous stage's source snapshot before adding this run.
        prior_path = out / "code_hashes.json"
        preserved = out / "chord_code_hashes.json"
        if prior_path.exists() and not preserved.exists():
            shutil.copyfile(prior_path, preserved)
        current = read_json(prior_path)
        current.update(hashes)
        atomic_json(prior_path, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Zero means all eligible performances; nonzero is an incomplete smoke run")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--broad-tempo", action="store_true",
                        help="Separate sensitivity: preferred/triad/factor1 with 14 tempo ratios from .5 through1.8")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("--limit must be nonnegative and --workers must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode = "broad_tempo" if args.broad_tempo else "primary"
    prefix = "position_broad_tempo" if args.broad_tempo else "position"
    metrics_path = out / ("position_metrics_broad_tempo.csv" if args.broad_tempo else "position_metrics.csv")
    trace_destination = out / "broad_tempo" if args.broad_tempo else out
    started = time.perf_counter()
    started_utc = utc_now()
    hashes_start = source_hashes()
    protected_paths = [out / name for name in ("chord_metrics.csv", "profile_audit.json", "training_profile_counts.npz")]
    if args.broad_tempo:
        protected_paths += [out / name for name in ("position_metrics.csv", "summary.json", "config.json", "code_hashes.json")]
    protected_before = {str(p): file_hash(p) for p in protected_paths}
    benchmark.old.checks(); benchmark.pardo.checks(); benchmark.yamaha.checks()
    benchmark.common.checks(); benchmark.follower.checks()

    manifest = json.loads((HERE / "data" / "manifest.json").read_text())
    compositions = {s["song"]: benchmark.composition_key(s["catalog"]["title"]) for s in manifest["songs"]}
    entries = sorted(manifest["performances"], key=lambda e: (e["take"] != "preferred", e["protocol"], e["song"]))
    # Always train profiles on the full preferred set, even for --limit smoke.
    loaded = [(entry, benchmark.common.load_performance(entry)) for entry in entries]
    profiles = benchmark.build_profiles(loaded, compositions)
    eligible = [(entry, data) for entry, data in loaded if not args.broad_tempo or entry["take"] == "preferred"]
    selected = eligible[:args.limit] if args.limit else eligible
    expected_jobs = sum(len(job_specs(entry, args.broad_tempo)) for entry, _ in selected)
    expected_rows = expected_jobs * (len(METHODS) + 2)
    overrides = {"bins_per_beat": 2, "emission_power_per_second": 1. / benchmark.POSITION_WINDOW}
    if args.broad_tempo:
        overrides["tempo_ratios"] = BROAD_TEMPO_RATIOS
    effective_config = asdict(benchmark.follower.FilterConfig(**overrides))
    metadata = {
        "runner": str(Path(__file__).resolve()), "mode": mode, "started_at_utc": started_utc,
        "workers": args.workers, "multiprocessing_start_method": "spawn",
        "full_training_performances": len(loaded),
        "profile_training": "Preferred takes from the full loaded corpus; exclude every same-composition contribution at query",
        "input": "pooled", "preprocess": "raw", "methods": list(METHODS),
        "schemes": ["triad"] if args.broad_tempo else list(benchmark.SCHEMES),
        "position_window_seconds": benchmark.POSITION_WINDOW, "position_frame_seconds": .2,
        "yamaha_window_seconds": 0., "score_scaling": "run_benchmark.score_probabilities: fixed per-row z-score plus softmax",
        "filter_overrides": overrides, "filter_defaults": asdict(benchmark.follower.FilterConfig()),
        "filter_effective": effective_config,
        "initial_tempo_factors": [1.] if args.broad_tempo else [1., .9, 1.1],
        "stress_scope": "None" if args.broad_tempo else "Factors .9/1.1 only for preferred triad cases",
        "selected_performance_ids": [entry["performance_id"] for entry, _ in selected],
        "eligible_performances": len(eligible), "limit": args.limit,
        "metrics_filename": metrics_path.name, "trace_destination": str(trace_destination),
        "expected_jobs": expected_jobs, "expected_rows": expected_rows,
        "manifest_sha256": file_hash(HERE / "data" / "manifest.json"),
        "source_hashes_file": prefix + "_code_hashes.json",
        "reproduction_scope": "Shared causal baseline with published harmonic-score components; not complete Yamaha/Pardo followers",
        "tempo_sensitivity": "Exploratory .5..1.8 grid; all other filter constants retained" if args.broad_tempo else "Primary grid",
    }
    atomic_json(out / (prefix + "_config.json"), metadata)
    results, profile_audit = {}, []

    def checkpoint(status="running", error=None):
        rows = [row for job_id in sorted(results) for row in results[job_id]]
        atomic_csv(metrics_path, rows)
        hashes_end = source_hashes()
        changed = sorted(path for path in hashes_start if hashes_start[path] != hashes_end.get(path))
        protected_changed = [p for p, value in protected_before.items() if value != file_hash(Path(p))]
        all_done = len(results) == expected_jobs and len(rows) == expected_rows
        complete = status == "finished" and all_done and not args.limit and not changed and not protected_changed
        summary = {
            "mode": mode, "status": status, "complete": complete,
            "selected_scope_finished": status == "finished" and all_done,
            "started_at_utc": started_utc, "updated_at_utc": utc_now(),
            "jobs_completed": len(results), "jobs_expected": expected_jobs,
            "metric_rows": len(rows), "expected_metric_rows": expected_rows,
            "performances_completed": len({row["performance_id"] for row in rows}),
            "performances_selected": len(selected), "full_eligible_performances": len(eligible),
            "runtime_seconds": time.perf_counter() - started, "limit": args.limit,
            "source_files_changed_during_run": changed, "protected_outputs_changed": protected_changed,
            "error": error, "checks": "Module checks passed; runtime output coverage checked",
        }
        atomic_json(out / (prefix + "_summary.json"), summary)
        atomic_json(out / (prefix + "_checkpoint.json"), {"summary": summary, "completed_jobs": sorted(results)})
        atomic_json(out / (prefix + "_code_hashes.json"), {"at_start": hashes_start, "at_checkpoint": hashes_end})
        atomic_json(out / (prefix + "_profile_audit.json"), profile_audit)
        if not args.broad_tempo:
            update_main_metadata(out, summary, metadata, hashes_start, final=status != "running")
        return summary

    checkpoint()
    print(f"Loaded {len(loaded)} performances; {mode}: {len(selected)} selected, {expected_jobs} jobs, {args.workers} workers.", flush=True)
    jobs = iter(precomputed_jobs(selected, profiles, compositions, overrides, trace_destination,
                                 args.broad_tempo, profile_audit))
    pending = {}
    exhausted = False
    last_checkpoint = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * args.workers:
                    try:
                        job = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(position_worker, job)] = job["job_id"]
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    job_id, rows = future.result()
                    if job_id != pending.pop(future) or job_id in results or len(rows) != len(METHODS) + 2:
                        raise RuntimeError("Worker result identity/coverage mismatch")
                    results[job_id] = rows
                if len(results) % 5 == 0 or time.perf_counter() - last_checkpoint >= 10:
                    checkpoint()
                    last_checkpoint = time.perf_counter()
                    print(f"{len(results)}/{expected_jobs} jobs; elapsed={last_checkpoint-started:.1f}s", flush=True)
        final = checkpoint("finished")
        if final["source_files_changed_during_run"] or final["protected_outputs_changed"]:
            raise RuntimeError("Source/protected-output change detected; see position summary before trusting this run")
        if final["jobs_completed"] != expected_jobs or final["metric_rows"] != expected_rows:
            raise RuntimeError("Incomplete position output coverage")
    except BaseException as exc:
        for future in pending:
            future.cancel()
        checkpoint("failed", f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
