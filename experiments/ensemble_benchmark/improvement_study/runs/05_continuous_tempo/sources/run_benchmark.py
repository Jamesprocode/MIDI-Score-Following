"""All available JazzSAMBA MIDI: coarse harmony and shared causal following.

Run with NumPy. Original files are read only. Methods are explicit baseline
adaptations; none is a reproduction of a complete published accompaniment system.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harmony_pilot"))
import run_pilot as old
import pardo_matcher as pardo
import yamaha_matcher as yamaha
import common
import position_filter as follower

WINDOWS = [0.5, 1.0, 2.0, 4.0]
POSITION_WINDOW = 1.0
SCHEMES = ["triad", "seventh"]
INPUTS = ["pooled", "bass", "horns"]
METHODS = ["cosine_onsets", "cosine_duration", "pardo", "yamaha_active", "empirical_profile"]


def composition_key(title):
    return re.sub(r"[^a-z0-9]", "", title.lower())


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_notes(streams, condition):
    selected = [n for name, n in streams.items() if condition == "pooled" or
                (condition == "bass" and name == "bass") or (condition == "horns" and name != "bass")]
    return np.concatenate(selected) if selected else np.empty((0, 3))


def make_targets(data, scheme):
    chord_keys = [common.quality_id(c, scheme) for c in data["chords"]]
    vocab = sorted(set(k for k in chord_keys if k is not None))
    if not vocab:
        raise ValueError("No supported harmonic vocabulary")
    labels = [common.label_string(v) for v in vocab]
    templates = np.stack([common.chord_template(*v) for v in vocab])
    roots = np.array([k[0] for k in vocab])
    ids = np.array([vocab.index(k) if k is not None else -1 for k in chord_keys])
    starts = np.array([float(c["start_time"]) for c in data["chords"]])
    ends = np.array([float(c["end_time"]) for c in data["chords"]])
    idx = np.searchsorted(starts, data["times"], side="right") - 1
    valid = (idx >= 0) & (data["times"] < ends[np.maximum(idx, 0)])
    target = np.full(len(idx), -1, dtype=int)
    target[valid] = ids[idx[valid]]
    valid &= target >= 0
    lengths = data["chord_ends_beats"] - data["chord_starts_beats"]
    chart_prior = np.zeros(len(vocab))
    for label_id, duration in zip(ids, lengths):
        if label_id >= 0:
            chart_prior[label_id] += duration
    majority = int(chart_prior.argmax())
    return {"vocab": vocab, "labels": labels, "templates": templates, "roots": roots,
            "ids": ids, "target": target, "valid": valid, "majority": majority,
            "chart_majority_accuracy": float((target[valid] == majority).mean())}


def chord_metrics(scores, available, targets, winner_mask=None):
    valid = targets["valid"]
    target = targets["target"][valid]
    scores = scores[valid]
    available = available[valid]
    if winner_mask is None:
        winners = np.isclose(scores, scores.max(axis=1, keepdims=True), atol=1e-10, rtol=0)
    else:
        winners = winner_mask[valid].copy()
        empty = ~winners.any(axis=1)
        assert not available[empty].any()
        winners[empty] = True
    nties = winners.sum(axis=1)
    guess = winners.argmax(axis=1)
    correct = (guess == target) & available
    fractional = winners[np.arange(len(target)), target] / nties * available
    balanced = np.mean([correct[target == c].mean() for c in np.unique(target)])
    return {
        "frames": len(target), "coverage": float(available.mean()),
        "chord_accuracy": float(correct.mean()), "fractional_accuracy": float(fractional.mean()),
        "balanced_accuracy": float(balanced),
        "root_accuracy": float(((targets["roots"][guess] == targets["roots"][target]) & available).mean()),
        "tie_rate": float((nties[available] > 1).mean()) if available.any() else 0.,
        "n_chord_labels": len(targets["vocab"]),
        "chart_majority_accuracy": targets["chart_majority_accuracy"],
    }


def score_probabilities(scores):
    """Common, label-free score scaling for the shared tracker, NOT calibration.

    Different harmonic methods have different units. Center and scale each row
    by its standard deviation, then softmax with fixed unit temperature. This
    preserves ranking, is invariant to positive score rescaling, and does not
    claim that row sharpness is calibrated uncertainty. All ties become uniform.
    """
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = scores.std(axis=1, keepdims=True)
    z = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-10)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def build_profiles(performance_data, compositions):
    """Training statistics from preferred takes; all same-composition files excluded at query."""
    totals = defaultdict(lambda: defaultdict(lambda: np.zeros(12)))
    by_composition = defaultdict(lambda: defaultdict(lambda: np.zeros(12)))
    for entry, data in performance_data:
        if entry["take"] != "preferred":
            continue
        comp = compositions[entry["song"]]
        starts = np.array([float(c["start_time"]) for c in data["chords"]])
        ends = np.array([float(c["end_time"]) for c in data["chords"]])
        for scheme in SCHEMES:
            keys = [common.quality_id(c, scheme) for c in data["chords"]]
            for condition in INPUTS:
                notes = selected_notes(data["notes_by_instrument"], condition)
                for onset, _, pitch in notes:
                    i = np.searchsorted(starts, onset, side="right") - 1
                    if i < 0 or onset >= ends[i] or keys[i] is None:
                        continue
                    root, quality = keys[i]
                    relative = (int(pitch) - root) % 12
                    group = (scheme, condition)
                    totals[group][quality][relative] += 1
                    by_composition[(comp, *group)][quality][relative] += 1
    return totals, by_composition


def empirical_templates(targets, scheme, condition, composition, profile_data):
    totals, by_composition = profile_data
    profiles, unseen = [], []
    for root, quality in targets["vocab"]:
        total = totals[(scheme, condition)][quality]
        excluded = by_composition[(composition, scheme, condition)][quality]
        counts = total - excluded
        assert np.all(counts >= 0)
        unseen.append(counts.sum() == 0)
        # Symmetric Dirichlet smoothing: no test labels or fitted hyperparameters.
        relative = (counts + .5) / (counts.sum() + 6.)
        profiles.append(np.roll(relative, root))
    return np.log(np.stack(profiles)), unseen


def position_metrics(output, data, sample):
    error = np.abs(output["position_beats"] - data["truth_position_beats"][sample])
    return {"frames": len(error),
            "bar_accuracy": float((output["bar_index"] == data["truth_bar_index"][sample]).mean()),
            "within2beat_accuracy": float((error <= 2).mean()),
            "mean_abs_beats": float(error.mean()), "median_abs_beats": float(np.median(error))}


def run_positions(entry, data, targets, scores_by_method, config, factors, destination, composition):
    output_rows = []
    sample = np.arange(0, len(data["times"]), 2)  # 200 ms emissions/decisions, all methods identical.
    times = data["times"][sample]
    for factor in factors:
        meta = {"performance_id": entry["performance_id"], "song": entry["song"], "protocol": entry["protocol"],
                "take": entry["take"], "composition": composition, "scheme": targets["scheme"],
                "initial_tempo_factor": factor}
        clock = follower.clock_positions(times, data["bar_starts_beats"], data["initial_bpm"] * factor,
                                          start_time=data["start_time"], score_end_beats=data["score_end_beats"])
        output_rows.append({**meta, "method": "clock", "window_seconds": 0., **position_metrics(clock, data, sample)})
        uniform = follower.follow_positions(times, np.ones((len(times),len(targets["vocab"]))),
                                            np.zeros(len(times),dtype=bool),data["chord_starts_beats"],
                                            data["chord_ends_beats"],targets["ids"],data["bar_starts_beats"],
                                            data["initial_bpm"]*factor,start_time=data["start_time"],config=config)
        output_rows.append({**meta,"method":"uniform_filter","window_seconds":0.,
                            **position_metrics(uniform,data,sample)})
        for method, (scores, available) in scores_by_method.items():
            probabilities = score_probabilities(scores[sample])
            output = follower.follow_positions(times, probabilities, available[sample], data["chord_starts_beats"],
                                               data["chord_ends_beats"], targets["ids"], data["bar_starts_beats"],
                                               data["initial_bpm"] * factor, start_time=data["start_time"], config=config)
            output_rows.append({**meta, "method": method, "window_seconds": 0. if method == "yamaha_active" else POSITION_WINDOW,
                                **position_metrics(output, data, sample)})
            if factor == 1. and entry["take"] == "preferred":
                path = destination / "position_traces" / entry["protocol"] / entry["song"]
                path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path / f"{targets['scheme']}_{method}.npz", time=times,
                                    estimated_beats=output["position_beats"], reference_beats=data["truth_position_beats"][sample],
                                    estimated_bar=output["bar_index"], reference_bar=data["truth_bar_index"][sample],
                                    estimated_bpm=output["tempo_bpm"])
    return output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; zero evaluates every eligible performance")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--skip-position", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old.checks(); pardo.checks(); yamaha.checks()
    if hasattr(common, "checks"):
        common.checks()
    manifest = json.loads((HERE / "data" / "manifest.json").read_text())
    compositions = {song["song"]: composition_key(song["catalog"]["title"]) for song in manifest["songs"]}
    entries = sorted(manifest["performances"], key=lambda e:(e["take"] != "preferred", e["protocol"], e["song"]))
    # Always train cross-composition profiles on all preferred data, even in smoke mode.
    loaded = [(entry, common.load_performance(entry)) for entry in entries]
    profiles = build_profiles(loaded, compositions)
    np.savez_compressed(out / "training_profile_counts.npz", **{
        "__".join([scheme, condition, quality]): counts
        for (scheme, condition), values in profiles[0].items() for quality, counts in values.items()})
    to_run = loaded[:args.limit] if args.limit else loaded
    config = {"bins_per_beat": 2, "emission_power_per_second": 1. / POSITION_WINDOW}
    chord_rows, position_rows, profile_audit = [], [], []
    print(f"Loaded {len(loaded)} performances; evaluating {len(to_run)}. Profile training excludes each query composition.", flush=True)
    for perf_index, (entry, data) in enumerate(to_run):
        comp = compositions[entry["song"]]
        targets_by_scheme = {scheme: {**make_targets(data, scheme), "scheme": scheme} for scheme in SCHEMES}
        for condition in INPUTS:
            raw_notes = selected_notes(data["notes_by_instrument"], condition)
            preprocessing = ["raw", "cap4s"] if condition == "pooled" else ["raw"]
            position_scores = {scheme:{} for scheme in SCHEMES}
            for preprocess in preprocessing:
                notes = raw_notes.copy()
                if preprocess == "cap4s":
                    # Causal duration cap: any note ceases contributing once four seconds have elapsed.
                    notes[:,1] = np.minimum(notes[:,1], notes[:,0]+4.)
                active88 = yamaha.piano_roll88(notes, data["times"])
                active_available = active88.sum(axis=1) > 0
                cached = {}
                for window in WINDOWS:
                    cached[("onsets",window)] = old.features(notes, data["times"], window, "onsets")
                    cached[("duration",window)] = old.features(notes, data["times"], window, "elapsed_duration")
                    cached[("segments",window)] = pardo.segment_profiles(notes, data["times"], window)
                for scheme, targets in targets_by_scheme.items():
                    templates = targets["templates"]
                    log_profiles, unseen = empirical_templates(targets, scheme, condition, comp, profiles)
                    if preprocess == "raw":
                        profile_audit.append({"performance_id":entry["performance_id"], "scheme":scheme, "input":condition,
                                              "excluded_composition":comp,"unseen_qualities":[label for label,flag in zip(targets["labels"],unseen) if flag]})
                    cases = [("yamaha_active",0.,yamaha.harmonic_log_scores(active88,yamaha.expand_chroma_templates(templates)),active_available,None)]
                    for window in WINDOWS:
                        onset, duration, segment = [cached[(key,window)] for key in ("onsets","duration","segments")]
                        pscores = pardo.score(segment, templates)
                        cases += [
                            ("cosine_onsets",window,old.cosine_scores(onset,templates),onset.sum(axis=1)>0,None),
                            ("cosine_duration",window,old.cosine_scores(duration,templates),duration.sum(axis=1)>0,None),
                            ("pardo",window,pscores,segment.sum(axis=1)>0,pardo.root_tie_mask(segment,pscores,targets["roots"])),
                            ("empirical_profile",window,old.normalize_rows(onset)@log_profiles.T,onset.sum(axis=1)>0,None),
                        ]
                    for method,window,scores,available,winners in cases:
                        metadata = {"performance_id":entry["performance_id"],"song":entry["song"],"protocol":entry["protocol"],
                                    "take":entry["take"],"composition":comp,"scheme":scheme,"input":condition,
                                    "preprocess":preprocess,"method":method,"window_seconds":window}
                        chord_rows.append({**metadata,**chord_metrics(scores,available,targets,winners)})
                        if condition == "pooled" and preprocess == "raw" and window in (0.,POSITION_WINDOW):
                            # Pardo's tie rule is a label decision rule; the filter keeps its raw score ambiguities.
                            position_scores[scheme][method] = (scores,available)
            if condition == "pooled" and not args.skip_position:
                for scheme, targets in targets_by_scheme.items():
                    factors = [1.]
                    if scheme == "triad" and entry["take"] == "preferred":
                        factors += [.9,1.1]
                    position_rows += run_positions(entry,data,targets,position_scores[scheme],config,factors,out,comp)
        if perf_index % 5 == 0 or perf_index == len(to_run)-1:
            print(f"{perf_index+1}/{len(to_run)} {entry['performance_id']} elapsed={time.perf_counter()-started:.1f}s",flush=True)
            csv_write(out/"chord_metrics.csv",chord_rows)
            if position_rows:
                csv_write(out/"position_metrics.csv",position_rows)
    csv_write(out/"chord_metrics.csv",chord_rows)
    if position_rows:
        csv_write(out/"position_metrics.csv",position_rows)
    (out/"profile_audit.json").write_text(json.dumps(profile_audit,indent=2)+"\n")
    metadata = {
        "task":"All local eligible MIDI, chart-constrained coarse chord estimation and shared causal following",
        "songs":manifest["protocol_song_counts"],"performances":len(to_run),"full_eligible_performances":len(loaded),
        "preferred_is_primary":True,"sync_alternate_excluded":"No distinct timing annotations available",
        "extra_takes_excluded":True,"instruments":"all available pitched bass/horn MIDI; percussion excluded; piano MIDI absent",
        "chord_schemes":{
            "triad":"root+major/minor/diminished/augmented/sus2/sus4/power; ignore sevenths, extensions, alterations, inversions",
            "seventh":"root+triad and seventh family; ignore 6/9/11/13 additions, alterations and inversions"},
        "chord_task":"Choose among unique reduced labels occurring in this known chart, not unconstrained chord recognition",
        "chord_frame_seconds":.1,"position_frame_seconds":.2,
        "chord_accuracy":"top1 correct reduced label / all non-N.C. labeled frames; unavailable evidence counts incorrect",
        "tie_rules":"Pardo uses root-weight tie rule, then vocabulary order; others vocabulary order. Fractional1/k accuracy also reported.",
        "balanced_accuracy":"mean recall over each distinct reference label within a performance, standard class-balanced accuracy",
        "root_accuracy":"correct root regardless of quality; silence counts incorrect",
        "majority_baseline":"constant chord with largest total duration in untimed score beats; uses no performed timestamps",
        "no_chord":"excluded from chord accuracy; neutral harmonic likelihood at N.C. plan positions; bar following still evaluated",
        "method_scope":{
            "cosine_onsets":"binary chord-tone cosine on trailing onset counts",
            "cosine_duration":"binary chord-tone cosine on trailing elapsed note durations",
            "pardo":"published P-(M+N), minimal-segment counts and root tie rule; causal windows/reduced chart vocabulary are adaptations",
            "yamaha_active":"published log likelihood x^TT; binary88-key roll/templates and no melody are reconstruction choices",
            "empirical_profile":"leave-one-composition-out root-relative pitch-class log likelihood; symmetric0.5 pseudocount per class; other preferred takes only"},
        "windows_seconds":WINDOWS,"position_window_seconds":POSITION_WINDOW,
        "raw_is_primary":True,"cap4s":"pooled-only sensitivity; common causal max4-second contribution per note, no label-based cleaning",
        "position_filter":"shared position-tempo grid Bayesian baseline, NOT complete Yamaha/Pardo followers; uniform_filter runs identical dynamics with no harmonic updates",
        "position_initialization":"known first musical position and synchronized start time; approximate catalog BPM",
        "position_plan":"unfolded chord/bar order and musical durations from bar/fraction coordinates; no timed reference after initialization",
        "position_score_scaling":"per-row z-score+softmax, fixedunittemperature, same for all methods; uncalibrated compatibility potentials",
        "position_config":config,"position_default_config":follower.FilterConfig().__dict__,
        "position_stress":"triad preferred only: initial BPM multiplied0.9 or1.1, same raw MIDI; controlled initialization error",
        "position_metrics":{
            "bar_accuracy":"fraction of200ms frames at correct unfolded plan bar; MIDI adaptation of measure-number alignment accuracy",
            "within2beat_accuracy":"fraction within2 score beats; MIDI-frame adaptation of alignment-rate/tolerance evaluation",
            "mean_median_abs":"all-frame absolute error, no exclusion of errors>2 beats; differs from Matchmaker conditional summaries",
            "reference_beats":"interpolated within annotated bars; not independently annotated beat onsets"},
        "evaluation_status":"exploratory all-song sweep; no held-out best-setting claim; learned profiles exclude same composition",
        "runtime_seconds_not_streaming_latency":time.perf_counter()-started,
    }
    (out/"config.json").write_text(json.dumps(metadata,indent=2)+"\n")
    summary = {"complete":not bool(args.limit) and not args.skip_position,
               "chord_complete":not bool(args.limit),"position_complete":not bool(args.limit) and not args.skip_position,
               "performances":len(to_run),"chord_metric_rows":len(chord_rows),
               "position_metric_rows":len(position_rows),"runtime_seconds":time.perf_counter()-started,
               "methods":METHODS,"schemas":SCHEMES,"checks":"module analytical/causality checks passed; full output review separate"}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    paths=[Path(__file__),HERE/"common.py",HERE/"position_filter.py",HERE.parent/"harmony_pilot/pardo_matcher.py",HERE.parent/"harmony_pilot/yamaha_matcher.py"]
    (out/"code_hashes.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
