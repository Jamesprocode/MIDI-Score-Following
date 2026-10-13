"""Live-band-only, controlled test of initial and ongoing tempo probabilities.

Original data, manifests, follower, and results are read-only. A study copy of
the follower separates its initial weights from its stationary motion weights.
All harmonic and positional settings are otherwise preserved. Run with the
bundled Python runtime and --workers 3. --limit 1 is an incomplete smoke test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
PROJECT = BENCH.parent.parent
sys.path.insert(0, str(BENCH))
import numpy as np
import run_benchmark as benchmark
import position_filter_study as study_filter

VARIANTS = {
    "A_gaussian": (False, False),
    "B_equal_initial": (True, False),
    "C_equal_dynamics": (False, True),
    "D_equal_both": (True, True),
    "no_harmony": (False, False),
}
CASES = ("59-groovin_high", "70-samba_de_orfeu")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def resolve_entry(entry):
    """Resolve relocated inputs in memory; never rewrite the original manifest."""
    value = json.loads(json.dumps(entry))
    value["annotation_path"] = str(BENCH / "data/annotations" / entry["protocol"] / (entry["song"] + ".json"))
    for item in value["midi_files"]:
        item["path"] = str(PROJECT / "Dataset" / "SAMBA dataset" / (entry["protocol"] + "_midi") /
                           "songs" / entry["song"] / entry["take"] / item["name"])
        if not Path(item["path"]).is_file():
            raise FileNotFoundError(item["path"])
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("MIDI differs from original benchmark: " + item["path"])
    if not Path(value["annotation_path"]).is_file():
        raise FileNotFoundError(value["annotation_path"])
    return value


def check_study_filter(config):
    """Check default identity and that initialization and dynamics are separate."""
    times = np.arange(.2, 4.01, .2)
    kwargs = dict(times=times, chord_probabilities=np.tile([.8, .2], (len(times), 1)),
                  available=np.ones(len(times), bool), chord_starts_beats=np.array([0., 4.]),
                  chord_ends_beats=np.array([4., 40.]), chord_ids=np.array([0, 1]),
                  bar_starts_beats=np.arange(0., 40., 4.), initial_bpm=120., start_time=0., config=config)
    old = benchmark.follower.follow_positions(**kwargs)
    new = study_filter.follow_positions(**kwargs)
    for key in ("position_beats", "tempo_bpm", "final_posterior"):
        np.testing.assert_array_equal(old[key], new[key])
    uniform = np.ones(9) / 9
    zero = dict(kwargs, times=np.array([0.]), chord_probabilities=np.array([[.8, .2]]), available=np.array([False]))
    initialized = study_filter.follow_positions(**zero, initial_tempo_weights=uniform)
    np.testing.assert_allclose(initialized["tempo_probabilities"][0], uniform, rtol=0, atol=1e-15)
    original_init = study_filter.follow_positions(**zero)
    dynamics_only = study_filter.follow_positions(**zero, dynamics_tempo_weights=uniform)
    np.testing.assert_array_equal(original_init["tempo_probabilities"], dynamics_only["tempo_probabilities"])
    return {"default_copy_identical": True, "uniform_initial_is_one_ninth": True,
            "changing_dynamics_does_not_change_initial_weights": True}


def frame_diagnostics(data, targets, active, raw, sample):
    """Reference-derived quantities are diagnostics, never inference inputs."""
    selected_target = targets["target"][sample]
    valid = targets["valid"][sample]
    available = active[sample].sum(axis=1) > 0
    selected_scores = raw[sample]
    winners = np.isclose(selected_scores, selected_scores.max(axis=1, keepdims=True), atol=1e-10, rtol=0)
    n_winners = winners.sum(axis=1)
    correct_best = np.zeros(len(sample), bool)
    correct_best[valid] = winners[np.arange(len(sample))[valid], selected_target[valid]]
    times = data["times"][sample]
    durations = np.array([float(b["end_time"]) - float(b["start_time"]) for b in data["bars"]])
    bar_bpm = 60 * data["beats_per_bar"] / durations
    frame_bpm = bar_bpm[data["truth_bar_index"][sample]]
    truth_harmony = np.full(len(sample), np.nan)
    truth_harmony[valid] = selected_scores[np.arange(len(sample))[valid], selected_target[valid]]
    return dict(time=times, available=available, valid_chord=valid, target_chord=selected_target,
                scores=selected_scores, correct_chord_among_best=correct_best, tied_best=n_winners,
                reference_bar_bpm=frame_bpm, active_pitches=active[sample].sum(axis=1),
                reference_chord_score=truth_harmony,
                reference_beats=data["truth_position_beats"][sample],
                reference_bar=data["truth_bar_index"][sample], chord_labels=np.asarray(targets["labels"]))


def run_one(entry, output_dir, config, old_rows):
    started = time.perf_counter()
    data = benchmark.common.load_performance(entry)
    targets = benchmark.make_targets(data, "triad")
    notes = benchmark.selected_notes(data["notes_by_instrument"], "pooled")
    active = benchmark.yamaha.piano_roll88(notes, data["times"])
    raw = benchmark.yamaha.harmonic_log_scores(active, benchmark.yamaha.expand_chroma_templates(targets["templates"]))
    sample = np.arange(0, len(data["times"]), 2)
    times = data["times"][sample]
    probabilities = benchmark.score_probabilities(raw[sample])
    diagnostics = frame_diagnostics(data, targets, active, raw, sample)
    trace_dir = Path(output_dir) / "traces" / entry["song"]
    trace_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_dir / "evidence.npz", **diagnostics,
                        chord_probabilities=probabilities,
                        chord_starts_beats=data["chord_starts_beats"],
                        chord_ends_beats=data["chord_ends_beats"], chord_ids=targets["ids"])
    uniform = np.ones(len(config["tempo_ratios"])) / len(config["tempo_ratios"])
    rows = []
    summaries = {}
    base_kwargs = dict(times=times, chord_probabilities=probabilities, available=diagnostics["available"],
                       chord_starts_beats=data["chord_starts_beats"], chord_ends_beats=data["chord_ends_beats"],
                       chord_ids=targets["ids"], bar_starts_beats=data["bar_starts_beats"],
                       initial_bpm=data["initial_bpm"], start_time=data["start_time"], config=config)
    for method, (equal_initial, equal_dynamics) in VARIANTS.items():
        kwargs = dict(base_kwargs)
        if equal_initial: kwargs["initial_tempo_weights"] = uniform
        if equal_dynamics: kwargs["dynamics_tempo_weights"] = uniform
        if method == "no_harmony": kwargs["available"] = np.zeros(len(times), bool)
        output = study_filter.follow_positions(**kwargs)
        metrics = benchmark.position_metrics(output, data, sample)
        rows.append(dict(performance_id=entry["performance_id"], song=entry["song"], method=method, **metrics))
        np.savez_compressed(trace_dir / (method + ".npz"), time=times,
                            estimated_beats=output["position_beats"], estimated_bar=output["bar_index"],
                            estimated_bpm=output["tempo_bpm"], mean_bpm=output["mean_tempo_bpm"],
                            tempo_probabilities=output["tempo_probabilities"],
                            tempo_values=data["initial_bpm"] * np.asarray(config["tempo_ratios"]))
        if method == "A_gaussian":
            prior = old_rows[entry["performance_id"]]
            for field in ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats"):
                if not np.isclose(metrics[field], float(prior[field]), rtol=0, atol=1e-10):
                    raise AssertionError(f"Original baseline mismatch {entry['song']} {field}")
            old_trace = BENCH / "results/position_traces/sync" / entry["song"] / "triad_yamaha_active.npz"
            saved = np.load(old_trace)
            np.testing.assert_array_equal(output["position_beats"], saved["estimated_beats"])
            np.testing.assert_array_equal(output["tempo_bpm"], saved["estimated_bpm"])
        error = output["position_beats"] - diagnostics["reference_beats"]
        quarters = []
        for indices in np.array_split(np.arange(len(times)), 4):
            quarters.append(dict(start=float(times[indices[0]]), end=float(times[indices[-1]]),
                                 correct_bar=float((output["bar_index"][indices] == diagnostics["reference_bar"][indices]).mean()),
                                 median_signed_beats=float(np.median(error[indices])),
                                 median_estimated_bpm=float(np.median(output["tempo_bpm"][indices]))))
        summaries[method] = dict(**metrics, quarters=quarters, final_signed_error=float(error[-1]))
    clock = benchmark.follower.clock_positions(times, data["bar_starts_beats"], data["initial_bpm"],
                                               start_time=data["start_time"], score_end_beats=data["score_end_beats"])
    rows.append(dict(performance_id=entry["performance_id"], song=entry["song"], method="clock",
                     **benchmark.position_metrics(clock, data, sample)))
    np.savez_compressed(trace_dir / "clock.npz", time=times, estimated_beats=clock["position_beats"],
                        estimated_bar=clock["bar_index"], estimated_bpm=clock["tempo_bpm"])
    valid = diagnostics["valid_chord"]
    valid_available = valid & diagnostics["available"]
    case = dict(song=entry["song"], metadata_bpm=data["initial_bpm"],
                title=entry.get("title", entry["song"]), bars=len(data["bars"]),
                instruments=list(data["notes_by_instrument"]), duration_seconds=float(times[-1]-times[0]),
                chord_metrics=benchmark.chord_metrics(raw, active.sum(axis=1) > 0, targets),
                correct_chord_among_best_available=float(diagnostics["correct_chord_among_best"][valid_available].mean()),
                uniquely_correct_chord_available=float((diagnostics["correct_chord_among_best"] & (diagnostics["tied_best"] == 1))[valid_available].mean()),
                bar_bpm_percentiles=np.quantile(diagnostics["reference_bar_bpm"], [.1, .5, .9]).tolist(),
                time_outside_tempo_range=float(((diagnostics["reference_bar_bpm"] < .7 * data["initial_bpm"]) |
                                                (diagnostics["reference_bar_bpm"] > 1.3 * data["initial_bpm"])).mean()),
                variants=summaries, inference_uses_reference_tempo=False)
    save_json(trace_dir / "diagnostics.json", case)
    return rows, case, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BENCH / "data/manifest.json").read_text())
    catalog = {x["song"]: x["catalog"] for x in manifest["songs"]}
    entries = [resolve_entry(e) for e in manifest["performances"] if e["protocol"] == "sync" and e["take"] == "preferred"]
    for e in entries: e["title"] = catalog[e["song"]]["title"]
    if args.limit: entries = entries[:args.limit]
    config = json.loads((BENCH / "results/position_config.json").read_text())["filter_effective"]
    checks = check_study_filter(config)
    with (BENCH / "results/position_metrics.csv").open() as f:
        old_rows = {r["performance_id"]: r for r in csv.DictReader(f) if r["protocol"] == "sync" and
                    r["take"] == "preferred" and r["scheme"] == "triad" and r["method"] == "yamaha_active" and
                    float(r["initial_tempo_factor"]) == 1.}
    protected = [BENCH / "position_filter.py", BENCH / "data/manifest.json", BENCH / "results/position_metrics.csv"]
    original_hashes = {str(p): digest(p) for p in protected}
    fingerprint = {e["performance_id"]: {"annotation": digest(Path(e["annotation_path"])),
                                         "midi": {x["name"]: x["sha256"] for x in e["midi_files"]}} for e in entries}
    save_json(out / "resolved_inputs.json", entries)
    save_json(out / "config.json", dict(protocol="sync", take="preferred", scheme="triad", input="original pooled bass and horns",
                                        excluded_protocol="async", piano_included=False, update_seconds=.2,
                                        snapshot_history_seconds=0., filter=config, variants=VARIANTS,
                                        case_selection="Best/worst original bar accuracy, selected before new results", cases=CASES,
                                        input_fingerprint=fingerprint, protected_before=original_hashes,
                                        created_utc=datetime.now(timezone.utc).isoformat()))
    rows, cases = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_one, e, str(out), config, old_rows): e["song"] for e in entries}
        for future in as_completed(jobs):
            found, case, seconds = future.result()
            rows.extend(found); cases.append(case)
            save_csv(out / "metrics.csv", sorted(rows, key=lambda r: (r["song"], r["method"])))
            print(f"{len(cases)}/{len(entries)} {case['song']} ({seconds:.1f}s)", flush=True)
    aggregate = []
    for method in (*VARIANTS, "clock"):
        selected = [r for r in rows if r["method"] == method]
        aggregate.append(dict(method=method, songs=len(selected), **{key: float(np.mean([r[key] for r in selected]))
                              for key in ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats")}))
    save_csv(out / "aggregate.csv", aggregate)
    by_song = {r["song"]: {} for r in rows}
    for r in rows: by_song[r["song"]][r["method"]] = r
    deltas = []
    for song, methods in by_song.items():
        deltas.append(dict(song=song, original_bar=methods["A_gaussian"]["bar_accuracy"],
                           equal_initial_bar=methods["B_equal_initial"]["bar_accuracy"],
                           change_percentage_points=100*(methods["B_equal_initial"]["bar_accuracy"]-methods["A_gaussian"]["bar_accuracy"]),
                           equal_both_bar=methods["D_equal_both"]["bar_accuracy"]))
    save_csv(out / "per_song_comparison.csv", sorted(deltas, key=lambda r: r["change_percentage_points"], reverse=True))
    unchanged = all(digest(Path(p)) == h for p,h in original_hashes.items())
    if not unchanged: raise AssertionError("Protected original files changed")
    summary = dict(complete=len(entries)==26, songs=len(entries), runtime_seconds=time.perf_counter()-started,
                   verification=dict(**checks, all_original_traces_reproduced=True, originals_unchanged=unchanged),
                   primary_comparison="A versus B changes initial tempo probabilities only",
                   improved=sum(r["change_percentage_points"] > 1e-9 for r in deltas),
                   worsened=sum(r["change_percentage_points"] < -1e-9 for r in deltas),
                   unchanged=sum(abs(r["change_percentage_points"]) <= 1e-9 for r in deltas),
                   aggregate=aggregate, cases=sorted(cases,key=lambda r:r["song"]))
    save_json(out / "summary.json", summary)
    source_files = [Path(__file__), HERE / "position_filter_study.py", BENCH / "run_benchmark.py", BENCH / "common.py",
                    BENCH.parent / "harmony_pilot/yamaha_matcher.py", BENCH.parent / "harmony_pilot/run_pilot.py"]
    save_json(out / "source_hashes.json", {str(p): digest(p) for p in source_files})
    print(json.dumps(dict(songs=len(entries), aggregate=aggregate, improved=summary["improved"],
                          worsened=summary["worsened"], runtime_seconds=summary["runtime_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
