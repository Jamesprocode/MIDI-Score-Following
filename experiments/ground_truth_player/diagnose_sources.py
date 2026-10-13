"""Small paired diagnostics for the two displayed performances.

Read all existing inputs without changing them. Only diagnostic_controls.json
beside this script is written. Every control uses the existing p02 implementation,
known musical start, supplied catalog tempo, and identical 200 ms frames. This
is a post-hoc two-example diagnostic, not independent benchmark validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BENCHMARK = HERE.parent / "ensemble_benchmark"
STUDY = BENCHMARK / "improvement_study"
sys.path[:0] = [str(STUDY), str(BENCHMARK)]
import run_study
import particle_follower

PERFORMANCES = ["async/07-blue_bossa/preferred",
                "sync/67-on_green_dolphin_street/preferred"]
SEEDS = [0, 1, 2, 3, 4]
PLAN_SHUFFLE_SEED = 20261008


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_metrics(prediction, data, sample, mask):
    if not np.any(mask):
        return None
    subset = {"position_beats": prediction["position_beats"][mask],
              "bar_index": prediction["bar_index"][mask]}
    return run_study.benchmark.position_metrics(subset, data, sample[mask])


def diagnostic(performance_id, manifest):
    entry = copy.deepcopy(next(e for e in manifest["performances"]
                               if e["performance_id"] == performance_id))
    protocol, song, take = performance_id.split("/")
    annotation = BENCHMARK / "data/annotations" / protocol / f"{song}.json"
    entry["annotation_path"] = str(annotation)
    for item in entry["midi_files"]:
        path = PROJECT / "SAMBA dataset" / f"{protocol}_midi" / "songs" / song / take / item["name"]
        assert sha(path) == item["sha256"]
        item["path"] = str(path)
    data = run_study.benchmark.common.load_performance(entry)
    setting = next(s for s in run_study.variants("continuous_tempo")
                   if s["method"] == "p02_particles_slow")
    selected = np.arange(0, len(data["times"]), 2)
    times = data["times"][selected]
    case = run_study.get_features(data, {"triad"})["triad"]
    scores, available, mass = [a[selected] for a in case["features"]["active_bass_root"]]
    probabilities, _ = run_study.observation(scores, available, mass, setting)
    args_tail = (data["chord_starts_beats"], data["chord_ends_beats"],
                 case["targets"]["ids"], data["bar_starts_beats"], data["initial_bpm"])
    all_mask = np.ones(len(times), dtype=bool)
    rows = []

    def report(name, seed, prediction, **extra):
        record = {"method": name, "particle_seed": seed,
                  "all_frames": masked_metrics(prediction, data, selected, all_mask),
                  "bass_active_frames": masked_metrics(prediction, data, selected, available),
                  "bass_inactive_frames": masked_metrics(prediction, data, selected, ~available),
                  "last_estimated_beats": float(prediction["position_beats"][-1]),
                  "out_of_score_fraction": float((prediction["position_beats"] >= data["score_end_beats"]).mean()),
                  **extra}
        rows.append(record)
        print(performance_id, name, seed, record["all_frames"], flush=True)

    clock = run_study.benchmark.follower.clock_positions(
        times, data["bar_starts_beats"], data["initial_bpm"],
        start_time=data["start_time"], score_end_beats=data["score_end_beats"])
    report("fixed_catalog_clock", None, clock)
    shuffled = np.random.default_rng(PLAN_SHUFFLE_SEED).permutation(case["targets"]["ids"])
    assert np.array_equal(np.sort(shuffled), np.sort(case["targets"]["ids"]))
    for seed in SEEDS:
        config = {**setting["config"], "seed": seed}
        for name, avail, ids in (
            ("bass_root_harmony", available, case["targets"]["ids"]),
            ("same_filter_no_harmony", np.zeros_like(available), case["targets"]["ids"]),
            ("bass_root_shuffled_plan", available, shuffled),
        ):
            result = particle_follower.follow_positions(
                times, probabilities, avail, args_tail[0], args_tail[1], ids,
                args_tail[3], args_tail[4], start_time=data["start_time"], config=config)
            report(name, seed, result, resampling_count=result["resampling_count"])

    # A declared one-factor feature control: root has weight 1 instead of 2.
    plain_scores, plain_available, plain_mass = [a[selected] for a in case["features"]["active_bass"]]
    plain_probabilities, _ = run_study.observation(plain_scores, plain_available, plain_mass, setting)
    plain_result = particle_follower.follow_positions(
        times, plain_probabilities, plain_available, *args_tail,
        start_time=data["start_time"], config={**setting["config"], "seed": 0})
    report("bass_plain_triad_no_extra_root_weight", 0, plain_result,
           resampling_count=plain_result["resampling_count"])

    activity = {}
    for name, notes in data["notes_by_instrument"].items():
        active = run_study.benchmark.yamaha.piano_roll88(notes, times).sum(axis=1) > 0
        activity[name] = {"active_frames": int(active.sum()), "frame_count": len(times),
                          "coverage": float(active.mean())}
    target = case["targets"]["target"][selected]
    valid = case["targets"]["valid"][selected]
    winners = np.isclose(scores, scores.max(axis=1, keepdims=True), atol=1e-10, rtol=0)
    hits = winners[np.arange(len(target)), np.maximum(target, 0)] / winners.sum(axis=1)
    chord_active = available & valid
    chord_diagnostic = {
        "framewise_triad_fractional_accuracy_all_valid": float((hits[valid] * available[valid]).mean()),
        "framewise_triad_fractional_accuracy_active_only": float(hits[chord_active].mean()),
        "framewise_triad_tie_rate_active_only": float((winners[chord_active].sum(axis=1) > 1).mean()),
        "candidate_chords": case["targets"]["labels"],
        "note": "This instantaneous triad diagnostic is distinct from position following; ties get fractional credit and unavailable frames count as zero in all-valid accuracy.",
    }
    summary = {}
    for method in sorted({r["method"] for r in rows if r["particle_seed"] is not None}):
        group = [r for r in rows if r["method"] == method]
        if len(group) != len(SEEDS):
            continue
        summary[method] = {metric: {"min": min(r["all_frames"][metric] for r in group),
                                   "median": float(np.median([r["all_frames"][metric] for r in group])),
                                   "max": max(r["all_frames"][metric] for r in group)}
                           for metric in ["bar_accuracy", "within2beat_accuracy", "mean_abs_beats"]}
    return {"performance_id": performance_id, "frame_count": len(times),
            "start_time": data["start_time"], "end_time": data["end_time"],
            "initial_bpm": data["initial_bpm"], "setting": setting,
            "activity": activity, "instantaneous_harmony": chord_diagnostic,
            "controls": rows, "five_seed_summary": summary,
            "shuffled_plan": {"seed": PLAN_SHUFFLE_SEED,
                              "changed_interval_fraction": float((shuffled != case["targets"]["ids"]).mean()),
                              "interval_count": len(shuffled),
                              "original_ids": case["targets"]["ids"].tolist(),
                              "shuffled_ids": shuffled.tolist()},
            "sources": {"annotation": {"path": str(annotation), "sha256": sha(annotation)},
                        "midi": [{"path": item["path"], "sha256": item["sha256"]}
                                 for item in entry["midi_files"]]}}


if __name__ == "__main__":
    started = time.perf_counter()
    manifest_path = BENCHMARK / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = {"scope": "Post-hoc diagnostic on two displayed preferred takes; no selection or parameter tuning.",
              "assumptions": ["Known first musical start and score position", "Catalog initial tempo",
                              "Complete unfolded chord order and score-beat intervals from annotations; future absolute annotation timestamps withheld",
                              "Bass MIDI already separated and transcribed; no audio-to-MIDI pipeline evaluation",
                              "Same 200 ms evaluation frames for every method; all-frame metrics include silence",
                              "Reference beat position is interpolated between annotated bar boundaries, not independently beat-annotated",
                              "Five particle seeds 0..4; one fixed shuffled-plan permutation per song with seed 20261008"],
              "performances": [diagnostic(pid, manifest) for pid in PERFORMANCES],
              "implementation": [{"path": str(p), "sha256": sha(p)} for p in
                                 [Path(__file__), STUDY / "run_study.py", STUDY / "particle_follower.py",
                                  BENCHMARK / "common.py", BENCHMARK / "run_benchmark.py"]]}
    output["elapsed_seconds"] = time.perf_counter() - started
    (HERE / "diagnostic_controls.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Saved diagnostic_controls.json", output["elapsed_seconds"], flush=True)
