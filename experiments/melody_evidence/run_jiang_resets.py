"""Jiang, Maezawa & Xia (ICASSP 2025) reset ratio for our followers: how often would the follower have to be reset?

The paper (sec. III-C) feeds the performed notes in onset order and *hard-resets* the system whenever a note's
position is off by more than the correction threshold, |s - s~| > 0.5 beats (s~: where the system puts the note,
s: the true score position). A reset puts the system back at the true position and tempo and clears its state. With
N notes and R resets in a piece, the reset ratio is R / N; lower is better. Resets cannot be read off saved frames
(a follower that is never reset stays lost and fails at every later note), so this runs the followers again with
the resets inside the loop.

Our choices, where our setup differs from the paper's:
  - notes: every onset of the piano (recorded MIDI) and the horns (CREPE transcriptions, SAMBA_CLEANED_MIDI), the
    parts our followers listen to. The paper follows one piano player's notes.
  - our follower is time-driven (a step every 50 ms) and its estimate does not jump at a note, so s~ for a note is
    the follower's position at the first step at or after the onset, and s the true position at that step. Notes
    that first arrive at the same step share one check, so a step resets at most once.
  - the true tempo at a reset is the annotated tempo of the bar being played (60 * beats per bar / bar length).
  - a reset clears the posterior, re-centres the tempo grid on the true tempo and sets the fed-back tempo to it.
  - the follower starts at the true place with the played tempo, as in the other runs (the paper's systems also
    start from the true tempo).

--mode whole follows the whole take from the top (run_melody_follower.py's B5 and melody variants, plus "none":
counting with the tempo model only, our stand-in for the paper's "const." controller). --mode heads follows each
head on its own from its first instant, as run_melody_heads.py, with that runner's variants; only notes inside the
heads count.

Output: one JSON per take in OUT/resets, {variant: {"notes": N, "resets": R, "seconds": followed time, "reset_times":
[...]}}. Summarize with jiang_metrics.py.

  SAMBA_CLEANED_MIDI=... ../yamaha_timing/.venv/bin/python run_jiang_resets.py --mode heads --samba ... \\
      --checkpoint ../yamaha_timing/runs/lakh_r3b/best_model.pt --protocols sync async --workers 10 --shard 0 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

import run_melody_follower  # noqa: F401  (sets up the import paths)
from run_melody_follower import VARIANTS as WHOLE_VARIANTS
from run_melody_heads import VARIANTS as HEAD_VARIANTS, head_stretches
from melody_evidence import MelodyConfig, MelodyEvidence
from note_hmm import NoteHMMConfig, NoteHMMEvidence, horn_parts
import follower_data
import run_study
from evidence import NetworkPhase
from run_follow_b import STEP, played_bpm
from yamaha_filter import YamahaConfig
from sweep_fixed_weights import _init
import sweep_fixed_weights
from weighted_filter import (_advance, _beat_kernel, _beat_transition, _tempo_grid, _tempo_predict, _walk,
                             bar_indices)

HERE = Path(__file__).resolve().parent
THRESHOLD = .5  # beats, the paper's correction threshold


def follow_with_resets(times, chord_probabilities, available, chord_starts_beats, chord_ends_beats, chord_ids,
                       bar_starts_beats, beats_per_bar, initial_bpm, weights, *, start_time, truth_position, truth_bpm,
                       notes_at, threshold=THRESHOLD, network=None, config=YamahaConfig(), melody=None,
                       start_position=0.):
    """weighted_filter.follow_weighted, line for line, plus the hard reset: at a step where a note first arrives
    (notes_at > 0), if the position is more than ``threshold`` beats from ``truth_position`` (per step), the
    follower starts again there with the tempo ``truth_bpm`` (per step). With threshold = inf it is follow_weighted."""
    times = np.asarray(times, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(times), 2 if melody is None else 3))
    probability = np.asarray(chord_probabilities, dtype=float)
    probability = probability / np.maximum(probability.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    starts, ends, ids = map(np.asarray, (chord_starts_beats, chord_ends_beats, chord_ids))
    bars = np.asarray(bar_starts_beats, dtype=float)
    step = 1. / config.bins_per_beat
    positions = np.arange(int(np.ceil(ends[-1] / step)) + 1, dtype=float) * step
    interval = np.searchsorted(starts, positions, side="right") - 1
    valid = (interval >= 0) & (positions < ends[np.maximum(interval, 0)])
    state_chords = np.full(len(positions), -1, dtype=int)
    state_chords[valid] = ids[interval[valid]]
    ratios = _tempo_grid(config)
    tempi = initial_bpm * ratios
    prior = np.exp(-.5 * (np.log(ratios) / config.initial_tempo_log_std) ** 2)
    prior /= prior.sum()
    posterior = np.zeros((len(tempi), len(positions)))
    start = int(round(start_position / step))
    posterior[:, start] = prior
    taps, kernel = _beat_kernel(config)
    transition = _beat_transition(config, ratios) if config.beat_tempo_reversion else None
    n_classes = probability.shape[1]
    reachable, hint, previous = start, float(initial_bpm), float(start_time)
    out = {key: np.empty(len(times)) for key in ("position_beats", "tempo_bpm", "mean_tempo_bpm", "bpm_condition",
                                                 "position_confidence")}
    out["reset"] = np.zeros(len(times), bool)
    for frame, now in enumerate(times):
        elapsed = now - previous
        w_h, w_d = weights[frame][:2]
        w_m = weights[frame][2] if melody is not None else 0.
        if elapsed:
            beats_moved = tempi * elapsed / 60.
            reachable += int(np.ceil(beats_moved.max() / step))
            if reachable + 2 > len(positions):
                pad = max(reachable + 2, len(positions) + 256) - len(positions)
                posterior = np.pad(posterior, ((0, 0), (0, pad)))
                state_chords = np.pad(state_chords, (0, pad), constant_values=-1)
                positions = np.arange(len(positions) + pad, dtype=float) * step
            if config.tempo_model == "continuous":
                posterior = _tempo_predict(posterior, elapsed * config.continuous_rate_per_second, prior)
            posterior = _advance(posterior, beats_moved / step)
            if config.tempo_model == "beat":
                fraction = np.mod(positions + 1e-9, 1.0)
                crossed = posterior * (fraction[None, :] < beats_moved[:, None] - 1e-9)
                moved = _walk(crossed, taps, kernel) if transition is None else transition @ crossed
                posterior = posterior - crossed + moved
            known = state_chords >= 0
            log_weight = np.zeros(len(positions))
            if available[frame] and w_h:
                smooth = (1 - config.probability_floor_mix) * probability[frame] + config.probability_floor_mix / n_classes
                observation = np.full(len(positions), 1. / n_classes)
                observation[known] = smooth[state_chords[known]]
                log_h = np.log(observation)
                log_weight += (log_h - log_h.max()) * elapsed * w_h
            if network is not None and w_d:
                log_d = network.bar_log_probs(network.frame_at(now), hint)
                phase = np.mod(positions, beats_per_bar) / beats_per_bar
                log_pd = log_d[np.minimum((phase * len(log_d)).astype(int), len(log_d) - 1)]
                log_weight += (log_pd - log_pd.max()) * elapsed * w_d
            if melody is not None and w_m:
                log_m = melody.log_likelihood(now, hint, len(positions))
                if log_m is not None:
                    log_weight += (log_m - log_m.max()) * elapsed * w_m
            posterior *= np.exp(log_weight)[None, :]
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("follower lost all probability mass")
        posterior /= total
        position_marginal, tempo_marginal = posterior.sum(axis=0), posterior.sum(axis=1)
        best = position_marginal.argmax()
        if notes_at[frame] and abs(positions[best] - truth_position[frame]) > threshold:  # the hard reset
            start = int(round(truth_position[frame] / step))
            if start + 2 > len(positions):
                pad = max(start + 2, len(positions) + 256) - len(positions)
                posterior = np.pad(posterior, ((0, 0), (0, pad)))
                state_chords = np.pad(state_chords, (0, pad), constant_values=-1)
                positions = np.arange(len(positions) + pad, dtype=float) * step
            tempi = float(truth_bpm[frame]) * ratios
            posterior[:] = 0.
            posterior[:, start] = prior
            reachable, hint = max(reachable, start), float(truth_bpm[frame])
            position_marginal, tempo_marginal = posterior.sum(axis=0), posterior.sum(axis=1)
            best = start
            out["reset"][frame] = True
        out["position_beats"][frame] = positions[best]
        out["position_confidence"][frame] = position_marginal[best]
        out["tempo_bpm"][frame] = tempi[tempo_marginal.argmax()]
        out["mean_tempo_bpm"][frame] = tempo_marginal @ tempi
        out["bpm_condition"][frame] = hint
        if config.feedback and elapsed:
            hint += min(1., elapsed / config.feedback_time_constant) * (out["mean_tempo_bpm"][frame] - hint)
        previous = now
    out["bar_index"] = bar_indices(out["position_beats"], bars, ends[-1])
    return out


def note_onsets(data, samba):
    """Every onset the followers listen to: piano (recorded) and horns (transcribed), sorted, one per note."""
    horns, _ = horn_parts(samba, data["performance_id"])
    return np.sort(np.concatenate([data["notes_by_instrument"]["piano"][:, 0]] + [on for on, _ in horns]))


def bar_tempo(data, times):
    """Annotated tempo of the bar being played at each time, quarter notes per minute (as played_bpm)."""
    start = np.array([float(b["start_time"]) for b in data["bars"]])
    end = np.array([float(b["end_time"]) for b in data["bars"]])
    k = np.clip(np.searchsorted(start, times, side="right") - 1, 0, len(start) - 1)
    return 60. * data["beats_per_bar"] / (end[k] - start[k])


def variant_spec(mode, name):
    """(weights, melody kind or None): kind is "v1" (lead-sheet tune) or "v2" (note HMM)."""
    if mode == "heads":
        (w_h, w_d, w_m), cue = HEAD_VARIANTS[name]
        return (w_h, w_d) if cue is None else (w_h, w_d, w_m), cue
    if name == "none":
        return (0., 0.), None
    w_m, config = WHOLE_VARIANTS[name]
    if config is None:
        return (1., 1.), None
    if config == "note_hmm":
        return (1., 1., w_m), "v2"
    assert config.reference == "written", name  # the lead-sheet v1 only
    return (1., 1., w_m), "v1"


def run_variant(data, take, mode, name, samba, lead_sheets, params, threshold=THRESHOLD):
    weights, cue = variant_spec(mode, name)
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    times = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(sweep_fixed_weights._MODEL, take) if weights[1] else None
    if cue == "v1":
        melody = MelodyEvidence(data, samba, MelodyConfig(reference="written", lead_sheets=str(lead_sheets)))
    elif cue == "v2":
        melody = NoteHMMEvidence(data, samba, NoteHMMConfig(params=str(params), lead_sheets=str(lead_sheets)))
    else:
        melody = None
    truth = np.interp(times, data["times"], data["truth_position_beats"])
    tempo = bar_tempo(data, times)
    onsets = note_onsets(data, samba)
    if mode == "heads":
        spans = [tuple(np.searchsorted(times, [eval_times[a] - 1e-6, eval_times[b - 1] + 1e-6]))
                 for a, b in head_stretches(samba, data["performance_id"], eval_times)]
    else:
        spans = [(0, len(times))]
    notes = resets = 0
    seconds, reset_times = 0., []
    for lo, hi in spans:
        if hi - lo < 2:
            continue
        # a note is checked at the first step at or after its onset, inside this stretch
        inside = onsets[(onsets > times[lo]) & (onsets <= times[hi - 1])]
        notes_at = np.bincount(np.searchsorted(times[lo:hi], inside, side="left"), minlength=hi - lo)
        start = truth[lo] if mode == "heads" else 0.
        start_time = times[lo] if mode == "heads" else data["start_time"]
        out = follow_with_resets(times[lo:hi], probabilities[lo:hi], available[lo:hi], data["chord_starts_beats"],
                                 data["chord_ends_beats"], case["targets"]["ids"], data["bar_starts_beats"],
                                 data["beats_per_bar"], played_bpm(data), weights, start_time=start_time,
                                 truth_position=truth[lo:hi], truth_bpm=tempo[lo:hi], notes_at=notes_at,
                                 threshold=threshold, network=network, config=YamahaConfig(), melody=melody,
                                 start_position=start)
        notes += int(len(inside))
        resets += int(out["reset"].sum())
        seconds += float(times[hi - 1] - times[lo])
        reset_times += times[lo:hi][out["reset"]].round(3).tolist()
    return {"performance_id": data["performance_id"], "variant": name,
            "result": {"notes": notes, "resets": resets, "seconds": round(seconds, 3), "reset_times": reset_times}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("whole", "heads"), required=True)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--variants", nargs="*", default=None,
                        help="default: whole: none B5 B5+W0.5 B5+H0.5; heads: all of run_melody_heads.py's")
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    parser.add_argument("--lead-sheets", default=MelodyConfig.lead_sheets)
    parser.add_argument("--params", default=str(HERE / "params" / "crepe_all"), help="for the v2 variants")
    parser.add_argument("--protocols", nargs="+", choices=("sync", "async"), default=["sync"])
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="beats; the paper's is 0.5")
    parser.add_argument("--limit", type=int, default=0, help="only the first N takes (quick look)")
    args = parser.parse_args()
    variants = args.variants or (["none", "B5", "B5+W0.5", "B5+H0.5"] if args.mode == "whole" else list(HEAD_VARIANTS))
    (args.out / "resets").mkdir(parents=True, exist_ok=True)
    takes = follower_data.samba_follow_takes(args.samba, protocols=tuple(args.protocols))
    if args.limit:
        takes = takes[:args.limit]
    k, n = args.shard
    if n > 1:  # greedy by take length, deterministic
        load, owner = [0.0] * n, {}
        for data, _ in sorted(takes, key=lambda t: (-len(t[0]["times"]), t[0]["performance_id"])):
            i = load.index(min(load))
            owner[data["performance_id"]] = i
            load[i] += len(data["times"]) ** (2 if args.mode == "whole" else 1)
        takes = [t for t in takes if owner[t[0]["performance_id"]] == k]
    done = {f.stem.replace("__", "/") for f in (args.out / "resets").glob("*.json")}
    takes = [t for t in takes if t[0]["performance_id"] not in done]  # resume: skip takes already saved
    takes.sort(key=lambda t: -len(t[0]["times"]))
    print(json.dumps({"mode": args.mode, "takes": len(takes), "skipped_done": len(done), "variants": variants,
                      "workers": args.workers, "shard": [k, n], "params": args.params}), flush=True)
    pending = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.checkpoint),)) as pool:
        jobs = [pool.submit(run_variant, data, take, args.mode, name, args.samba, args.lead_sheets, args.params,
                            args.threshold) for data, take in takes for name in variants]
        for count, job in enumerate(as_completed(jobs), 1):
            r = job.result()
            parts = pending.setdefault(r["performance_id"], {})
            parts[r["variant"]] = r["result"]
            if len(parts) == len(variants):
                (args.out / "resets" / f"{r['performance_id'].replace('/', '__')}.json").write_text(
                    json.dumps({v: parts[v] for v in variants}))
                pending.pop(r["performance_id"])
                print(json.dumps({"done_take": r["performance_id"], "tasks_done": count, "of": len(jobs),
                                  "resets": {v: parts[v]["resets"] for v in variants}}), flush=True)


if __name__ == "__main__":
    main()
