"""Logged, causal position-following experiments; original benchmark is read-only.

Every run stores its candidate settings BEFORE inference, source hashes, progress,
per-performance metrics, and full traces. Test annotations are passed to metrics
only. No method may select a setting from its own reference labels at inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import run_benchmark as benchmark
import adaptive_grid

BROAD = tuple(round(.5+.1*i, 10) for i in range(14))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def log_event(out, event, **values):
    with (out/"events.jsonl").open("a") as handle:
        handle.write(json.dumps({"time":now(),"event":event,**values},allow_nan=False)+"\n")


def hashes(wave="diagnostics"):
    files = [Path(__file__), HERE/"adaptive_grid.py", BASE/"common.py", BASE/"run_benchmark.py",
             BASE/"position_filter.py", Path(benchmark.old.__file__), Path(benchmark.yamaha.__file__)]
    if wave == "rhythm_sequence":
        files += [HERE/"rhythm.py", HERE/"sequence_follower.py"]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def variants(wave):
    base = {"engine":"grid", "feature":"active_pooled", "scaling":"zscore", "reliability":"none",
            "scheme":"triad", "tempo_factor":1., "config":{}}
    def item(name, reason, **changes):
        return {**base, "method":name, "reason":reason, **changes}
    wave1 = [
        item("g00_reproduce", "Exact reproduction control for the prior Maezawa-style primary result"),
        item("g01_bar_decision", "Maximize posterior mass per bar rather than a single half-beat bin", config={"output_rule":"bar"}),
        item("g02_weak_harmony", "Reduce harm from unreliable harmonic updates", config={"emission_power_per_second":.25}),
        item("g03_weak_bar", "Combine weaker observations with a bar decision", config={"emission_power_per_second":.25,"output_rule":"bar"}),
        item("g04_bass_active", "Remove improvised horn pitches from harmonic observations", feature="active_bass", config={"output_rule":"bar"}),
        item("g05_balanced_active", "Prevent dense instrument transcriptions from dominating", feature="active_balanced",config={"output_rule":"bar"}),
        item("g06_bass_root", "Give bass-root correspondence additional weight",feature="active_bass_root",config={"output_rule":"bar"}),
        item("g07_onsets", "One-second onset context with posterior bar decision",feature="onset_pooled",config={"output_rule":"bar"}),
        item("g08_onset_delay", "Compare a trailing window with past candidate harmony while reporting current position",feature="onset_pooled",config={"output_rule":"bar","harmonic_delay_seconds":.5}),
        item("g09_bass_onset_delay", "Combine bass-only evidence with trailing-window age correction",feature="onset_bass",config={"output_rule":"bar","harmonic_delay_seconds":.5}),
        item("g10_raw_contrast", "Preserve absolute match contrast rather than standardizing every frame",scaling="fraction4",config={"output_rule":"bar"}),
        item("g11_confidence_gate", "Temper observations by information in their unstandardized scores",reliability="entropy",config={"output_rule":"bar"}),
        item("g12_broad_prior", "Allow catalog-tempo mismatch with broader tempo support and initial uncertainty",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3}),
        item("g13_finer_grid", "Reduce position rounding diffusion with quarter-beat bins",config={"output_rule":"bar","bins_per_beat":4}),
        item("g14_seventh", "Retain seventh quality while still ignoring higher extensions",scheme="seventh",config={"output_rule":"bar"}),
    ]
    if wave == "diagnostics":
        return wave1
    if wave == "rhythm_sequence":
        return [
            item("r00_rhythm_only", "Causal MIDI-onset timing without harmonic observations",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"emission_power_per_second":0.,"rhythm_strength":3.}),
            item("r01_joint", "Combine continuous harmonic and onset-phase evidence",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"rhythm_strength":3.}),
            item("r02_joint_weak_harmony", "Rely more on onset timing when harmonic evidence is weak",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"emission_power_per_second":.25,"rhythm_strength":3.}),
            item("r03_joint_gate", "Context-dependent harmony strength from match entropy with rhythm evidence",reliability="entropy",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"rhythm_strength":3.}),
            item("r04_joint_bass", "Use bass for harmony and weighted ensemble onsets for timing",feature="active_bass",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"rhythm_strength":3.}),
            item("r05_joint_delayed", "Use delayed harmonic context with onset timing",feature="onset_pooled",config={"output_rule":"bar","tempo_ratios":BROAD,"initial_tempo_ratio_std":.3,"harmonic_delay_seconds":.5,"rhythm_strength":3.}),
            item("s00_beam", "Continuous-position online Viterbi beam removes grid-rounding diffusion",engine="beam"),
            item("s01_beam_bass", "Beam sequence following with bass-only harmony",engine="beam",feature="active_bass"),
            item("s02_beam_onsets", "Beam sequence following with recent onset harmony",engine="beam",feature="onset_pooled"),
        ]
    raise ValueError(wave)


def get_features(data, schemes):
    times = data["times"]
    streams = data["notes_by_instrument"]
    active = {name:benchmark.yamaha.piano_roll88(notes,times) for name,notes in streams.items()}
    pooled = np.maximum.reduce(list(active.values()))
    bass = active.get("bass",np.zeros_like(pooled))
    balanced = sum(a/np.maximum(a.sum(axis=1,keepdims=True),1.) for a in active.values())
    pooled_notes = benchmark.selected_notes(streams,"pooled")
    bass_notes = benchmark.selected_notes(streams,"bass")
    onset = {"pooled":benchmark.old.features(pooled_notes,times,1.,"onsets"),
             "bass":benchmark.old.features(bass_notes,times,1.,"onsets")}
    result = {}
    for scheme in schemes:
        targets = benchmark.make_targets(data,scheme)
        templates = targets["templates"]
        t88 = benchmark.yamaha.expand_chroma_templates(templates)
        root_templates = templates.copy()
        root_templates[np.arange(len(templates)),targets["roots"]] += 1.
        result[scheme] = {"targets":targets,"features":{
            "active_pooled":(pooled@t88.T,pooled.sum(axis=1)>0,pooled.sum(axis=1)),
            "active_bass":(bass@t88.T,bass.sum(axis=1)>0,bass.sum(axis=1)),
            "active_balanced":(balanced@t88.T,balanced.sum(axis=1)>0,balanced.sum(axis=1)),
            "active_bass_root":(bass@benchmark.yamaha.expand_chroma_templates(root_templates).T,bass.sum(axis=1)>0,2*bass.sum(axis=1)),
            "onset_pooled":(benchmark.old.cosine_scores(onset["pooled"],templates),onset["pooled"].sum(axis=1)>0,np.ones(len(times))),
            "onset_bass":(benchmark.old.cosine_scores(onset["bass"],templates),onset["bass"].sum(axis=1)>0,np.ones(len(times))),
        }}
    return result


def observation(scores, available, mass, setting):
    fraction = scores/np.maximum(mass[:,None],1.)
    logits = 4.*fraction
    raw_p = np.exp(logits-logits.max(axis=1,keepdims=True))
    raw_p /= raw_p.sum(axis=1,keepdims=True)
    probabilities = benchmark.score_probabilities(scores) if setting["scaling"] == "zscore" else raw_p
    if setting["reliability"] == "entropy":
        k = scores.shape[1]
        reliability = np.zeros(len(scores)) if k == 1 else np.clip(1.+(raw_p*np.log(np.maximum(raw_p,1e-30))).sum(axis=1)/np.log(k),0.,1.)
    else:
        reliability = np.ones(len(scores))
    reliability[~available] = 0.
    return probabilities, reliability


def run_performance(entry, settings, out):
    started = time.perf_counter()
    # Always load on the original 100ms grid and subsample exactly as baseline.
    data = benchmark.common.load_performance(entry)
    sample = np.arange(0,len(data["times"]),2)
    times = data["times"][sample]
    features = get_features(data,set(s["scheme"] for s in settings))
    rhythm_cache = {}
    rows = []
    for setting in settings:
        before = time.perf_counter()
        scheme = setting["scheme"]
        case = features[scheme]
        scores,available,mass = [a[sample] for a in case["features"][setting["feature"]]]
        probabilities,reliability = observation(scores,available,mass,setting)
        config = setting["config"]
        bpm = data["initial_bpm"]*setting["tempo_factor"]
        args = (times, probabilities, available, data["chord_starts_beats"],data["chord_ends_beats"],
                case["targets"]["ids"],data["bar_starts_beats"],bpm)
        if setting["engine"] == "grid":
            rhythm = None
            if config.get("rhythm_strength",0):
                from rhythm import prepare_rhythm
                tempo_grid = tuple(bpm*np.asarray(config.get("tempo_ratios",adaptive_grid.FilterConfig().tempo_ratios)))
                key = (tempo_grid,bpm)
                if key not in rhythm_cache:
                    rhythm_cache[key] = prepare_rhythm(data["notes_by_instrument"],times,np.array(tempo_grid),bpm)
                rhythm = rhythm_cache[key]
            prediction = adaptive_grid.follow_positions(*args,start_time=data["start_time"],config=config,
                                                        rhythm=rhythm,harmonic_reliability=reliability)
        else:
            import sequence_follower
            prediction = sequence_follower.follow_positions(*args,start_time=data["start_time"],config=config)
        metrics = benchmark.position_metrics(prediction,data,sample)
        row = {"performance_id":entry["performance_id"],"song":entry["song"],"protocol":entry["protocol"],
               "take":entry["take"],"composition":benchmark.composition_key(entry.get("title",entry["song"])),
               "scheme":scheme,"initial_tempo_factor":setting["tempo_factor"],"method":setting["method"],
               "window_seconds":1. if setting["feature"].startswith("onset") else 0.,
               **metrics,"inference_seconds":time.perf_counter()-before,
               "mean_harmonic_reliability":float(reliability.mean()),
               "out_of_score_rate":float((prediction["position_beats"]>=data["score_end_beats"]).mean())}
        rows.append(row)
        path = out/"traces"/entry["protocol"]/entry["song"]/entry["take"]
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/f"{setting['method']}.npz",time=times,estimated_beats=prediction["position_beats"],
                            reference_beats=data["truth_position_beats"][sample],estimated_bar=prediction["bar_index"],
                            reference_bar=data["truth_bar_index"][sample],estimated_bpm=prediction["tempo_bpm"],
                            harmonic_reliability=reliability)
    return rows,time.perf_counter()-started


def checks():
    adaptive_grid.checks()
    # Default experimental grid must preserve the baseline exactly.
    times = np.arange(0.,15.,.2)
    starts = np.arange(0.,40.,4.)
    probabilities = np.tile([.1,.7,.2],(len(times),1))
    args = (times,probabilities,np.ones(len(times),bool),starts,starts+4,np.arange(len(starts))%3,starts,120.)
    original = benchmark.follower.follow_positions(*args,start_time=0.)
    copied = adaptive_grid.follow_positions(*args,start_time=0.)
    assert np.array_equal(original["position_beats"],copied["position_beats"])
    for rule in ("bar","mean"):
        config={"output_rule":rule,"harmonic_delay_seconds":.5}
        full=adaptive_grid.follow_positions(*args,start_time=0.,config=config)
        short_args=(times[:30],probabilities[:30],np.ones(30,bool),*args[3:])
        short=adaptive_grid.follow_positions(*short_args,start_time=0.,config=config)
        assert np.array_equal(full["position_beats"][:30],short["position_beats"])
    print("Reproduction and causal-prefix checks passed.",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wave",choices=["diagnostics","rhythm_sequence"],default="diagnostics")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--takes",choices=["preferred","all","alternate"],default="preferred")
    parser.add_argument("--methods",help="Comma-separated subset of predefined variant IDs")
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"run.json").exists():
        raise ValueError("Use a new run directory; completed or failed run records are never overwritten")
    checks()
    settings=variants(args.wave)
    if args.methods:
        requested=args.methods.split(",")
        settings=[s for s in settings if s["method"] in requested]
        assert len(settings)==len(requested), "Unknown or repeated method"
    manifest=json.loads((BASE/"data/manifest.json").read_text())
    entries=[e for e in manifest["performances"] if args.takes=="all" or e["take"]==args.takes]
    entries=sorted(entries,key=lambda e:(e["protocol"],e["song"],e["take"]))
    if args.limit:
        entries=entries[:args.limit]
    config={"started_at":now(),"wave":args.wave,"settings":settings,"scope":args.takes,"limit":args.limit,
            "performance_ids":[e["performance_id"] for e in entries],"workers":args.workers,
            "source_hashes":hashes(args.wave),"manifest_sha256":hashlib.sha256((BASE/"data/manifest.json").read_bytes()).hexdigest(),
            "selection":"Exploratory candidate comparison; no per-song oracle settings at inference. Later setting selection excludes the evaluated song.",
            "initialization":"Known musical start and position, catalog BPM; later reference times only for metrics",
            "metrics":"Same 200ms exact unfolded-bar, within2beats, all-frame mean/median beat error as preserved baseline",
            "reference":"Beat targets interpolated inside annotated bars; AMT timestamp origin not independently verified",
            "runtime":"Batch causal simulation; measured wall times are not live input/output latency"}
    write_json(out/"run.json",config)
    log_event(out,"started",settings=len(settings),performances=len(entries))
    started=time.perf_counter()
    rows=[]
    def checkpoint(status,error=None):
        ordered=sorted(rows,key=lambda r:(r["performance_id"],r["method"]))
        if ordered:
            benchmark.csv_write(out/"metrics.csv",ordered)
        current=hashes(args.wave)
        changed=[p for p,digest in config["source_hashes"].items() if current.get(p)!=digest]
        result={"status":status,"complete":status=="finished" and len(rows)==len(entries)*len(settings) and not changed and not args.limit,
                "scope_finished":status=="finished" and len(rows)==len(entries)*len(settings),"metric_rows":len(rows),
                "expected_rows":len(entries)*len(settings),"performances":len(entries),"settings":len(settings),
                "source_changes":changed,"seconds":time.perf_counter()-started,"error":error}
        write_json(out/"summary.json",result)
        return result
    checkpoint("running")
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            pending={pool.submit(run_performance,e,settings,out):e["performance_id"] for e in entries}
            for i,future in enumerate(as_completed(pending)):
                new_rows,elapsed=future.result()
                rows.extend(new_rows)
                log_event(out,"performance_finished",performance_id=pending[future],rows=len(new_rows),seconds=elapsed)
                if (i+1)%5==0 or i+1==len(entries):
                    checkpoint("running")
                    print(f"{i+1}/{len(entries)} performances, {len(rows)} rows, {time.perf_counter()-started:.1f}s",flush=True)
        summary=checkpoint("finished")
        log_event(out,"finished",**summary)
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        checkpoint("failed",str(exc))
        log_event(out,"failed",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__=="__main__":
    main()
