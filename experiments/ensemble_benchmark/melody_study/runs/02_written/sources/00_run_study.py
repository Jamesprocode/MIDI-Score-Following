"""Logged comparison of causal phrase-memory and written-melody observations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "improvement_study"))
import run_benchmark as benchmark
import particle_follower as original
import follower
import matchers
import pipeline

P02 = {"initial_tempo_ratio_std": .3, "tempo_diffusion_per_sqrt_second": .003}


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def hashes():
    sources = [Path(__file__), HERE/"pipeline.py", HERE/"matchers.py", HERE/"follower.py",
               Path(original.__file__), BASE/"run_benchmark.py", BASE/"common.py",
               Path(benchmark.old.__file__), Path(benchmark.yamaha.__file__)]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def lead_streams(entry, data, catalog):
    streams = data["notes_by_instrument"]
    chosen = {}
    for instrument, notes in streams.items():
        if instrument == "trumpet" and str(catalog.get("is_trumpet_lead")).lower() == "true":
            chosen[instrument] = notes
        elif "saxophone" in instrument and str(catalog.get("is_saxophone_lead")).lower() == "true":
            chosen[instrument] = notes
    # This fallback is recorded. It uses instrument identity, not test accuracy.
    if not chosen:
        chosen = {k:v for k,v in streams.items() if k != "bass"}
    return chosen


def harmony(data, sample):
    targets = benchmark.make_targets(data, "triad")
    templates = targets["templates"].copy()
    templates[np.arange(len(templates)), targets["roots"]] += 1.
    bass = data["notes_by_instrument"].get("bass", np.empty((0,3)))
    roll = benchmark.yamaha.piano_roll88(bass, data["times"][sample])
    scores = roll @ benchmark.yamaha.expand_chroma_templates(templates).T
    return targets, benchmark.score_probabilities(scores), roll.sum(axis=1)>0


def scopes(data, sample):
    t = data["times"][sample]
    result = {"all":np.ones(len(t),bool), "head_in":np.zeros(len(t),bool),
              "head_out":np.zeros(len(t),bool), "solo":np.zeros(len(t),bool)}
    for s in data["sections"]:
        mask = (t >= float(s["start_time"])) & (t < float(s["end_time"]))
        for name, flag in [("head_in","is_head_in"),("head_out","is_head_out"),("solo","is_solo")]:
            if str(s.get(flag,"False")).lower() == "true":
                result[name] |= mask
    return result


def run_one(entry, catalog, reference_path, out, seed, prefix_check=False):
    data = benchmark.common.load_performance(entry)
    sample = np.arange(0,len(data["times"]),2)
    times = data["times"][sample]
    target, probabilities, available = harmony(data,sample)
    args = (times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
            target["ids"],data["bar_starts_beats"],data["initial_bpm"])
    cfg = {**P02, "seed":seed}
    anchor = original.follow_positions(*args,start_time=data["start_time"],config=cfg)
    reproduced = follower.follow_positions(*args,start_time=data["start_time"],config=cfg)
    assert np.array_equal(anchor["position_beats"],reproduced["position_beats"])
    plan = pipeline.untimed_plan(data)
    streams = lead_streams(entry,data,catalog)
    written = None
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        assert reference["status"] == "usable"
        notes = reference["notes"]
        # Separate score-onset order from its aligned plan coordinate if a
        # reference exporter provides both; otherwise aligned beats supply both.
        raw = np.array([[float(n.get("score_onset_beats",n["onset_beats"])), int(n["pitch"]),
                         float(n["onset_beats"])] for n in notes])
        raw = raw[np.argsort(raw[:,0],kind="stable")]
        written = np.array([raw[raw[:,0] == t][np.argmax(raw[raw[:,0] == t,1])]
                            for t in np.unique(raw[:,0])])
    evidence = pipeline.prepare_all(streams,times,anchor,plan,written_notes=written)
    prefix_verified = False
    if prefix_check:
        n = min(len(times), max(30,len(times)//2))
        short_streams = {k:v[v[:,0] <= times[n-1]] for k,v in streams.items()}
        short_anchor = {"position_beats":anchor["position_beats"][:n], "tempo_bpm":anchor["tempo_bpm"][:n]}
        short = pipeline.prepare_all(short_streams,times[:n],short_anchor,plan,written_notes=written)
        positions = np.linspace(0,data["score_end_beats"],97)
        tempi = np.full(len(positions),data["initial_bpm"])
        for name in matchers.METHODS:
            assert np.array_equal(evidence[name].available[:n],short[name].available)
            for frame in range(n):
                assert np.array_equal(evidence[name](frame,times[frame],positions,tempi),
                                      short[name](frame,times[frame],positions,tempi))
        prefix_verified = True

    cases = [("harmonic_only", None, {**cfg}, reproduced),
             ("no_observations",None,{**cfg,"emission_power_per_second":0.},None)]
    for name in matchers.METHODS:
        cases += [(f"{name}__melodic_updates",name,{**cfg,"emission_power_per_second":0.,"melodic_power_per_second":1.},None),
                  (f"{name}__fused1",name,{**cfg,"melodic_power_per_second":1.},None),
                  (f"{name}__fused3",name,{**cfg,"melodic_power_per_second":3.},None)]
    rows = []
    section_masks = scopes(data,sample)
    directory = out/"traces"/entry["protocol"]/entry["song"]
    directory.mkdir(parents=True,exist_ok=True)
    for name, matcher, config, prediction in cases:
        before = time.perf_counter()
        if prediction is None:
            prediction = follower.follow_positions(*args,start_time=data["start_time"],config=config,
                                                  melodic_logscore=evidence[matcher] if matcher else None)
        inference = time.perf_counter()-before
        for scope, mask in section_masks.items():
            if not mask.any():
                continue
            selection = {"position_beats":prediction["position_beats"][mask], "bar_index":prediction["bar_index"][mask]}
            row = {"performance_id":entry["performance_id"],"song":entry["song"],"protocol":entry["protocol"],
                   "take":entry["take"],"reference":"written" if reference_path else "past", "method":name,
                   "matcher":matcher or "none", "scope":scope, "seed":seed,
                   **benchmark.position_metrics(selection,data,sample[mask]),
                   "melodic_coverage":float(evidence[matcher].available[mask].mean()) if matcher else 0.,
                   "matching_seconds":evidence[matcher].scoring_seconds if matcher else 0.,
                   "inference_seconds":inference}
            rows.append(row)
        np.savez_compressed(directory/f"{name}.npz",time=times,
            estimated_beats=prediction["position_beats"],estimated_bar=prediction["bar_index"],
            reference_beats=data["truth_position_beats"][sample],reference_bar=data["truth_bar_index"][sample],
            estimated_bpm=prediction["tempo_bpm"],
            melody_available=evidence[matcher].available if matcher else np.zeros(len(times),bool),
            **{f"scope_{k}":v for k,v in section_masks.items()})
    audit = {"performance_id":entry["performance_id"],"lead_instruments":list(streams),
             "source_note_counts":{k:len(v) for k,v in streams.items()},"reference_file":reference_path,
             "reference_note_count":len(written) if written is not None else None,
             "prefix_verified":prefix_verified,"neutral_reproduction":True,
             "query_updates":evidence[matchers.METHODS[0]].query_updates,
             "reference_comparisons":evidence[matchers.METHODS[0]].reference_comparisons}
    json_write(directory/"audit.json",audit)
    return rows,audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",choices=["past","written"],required=True)
    parser.add_argument("--written-root",type=Path,default=HERE/"references"/"written"/"aligned")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--prefix-check",action="store_true")
    args = parser.parse_args()
    manifest = json.loads((BASE/"data"/"manifest.json").read_text())
    entries = [e for e in manifest["performances"] if e["take"] == "preferred"]
    catalogs = {(s["protocol"],s["song"]):s["catalog"] for s in manifest["songs"]}
    written = {}
    if args.reference == "written":
        for path in args.written_root.glob("*.json"):
            value = json.loads(path.read_text())
            if value.get("status") == "usable":
                written[(value["protocol"],value["song"])] = str(path.resolve())
        entries = [e for e in entries if (e["protocol"],e["song"]) in written]
    eligible = len(entries)
    if args.limit:
        # Deterministic mix of the two protocols for smoke checks.
        entries = [e for group in zip([e for e in entries if e["protocol"]=="async"],
                                     [e for e in entries if e["protocol"]=="sync"]) for e in group][:args.limit]
    if not entries:
        raise SystemExit("No eligible performance/reference pairs; do not create fabricated written references")
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source_hashes = hashes()
    (out/"sources").mkdir()
    for index,(path,digest) in enumerate(source_hashes.items()):
        (out/"sources"/f"{index:02d}_{Path(path).name}").write_bytes(Path(path).read_bytes())
    input_paths = {str(BASE/"data"/"manifest.json"),str(BASE/"data"/"catalog.json")}
    for e in entries:
        input_paths.add(e["annotation_path"])
        input_paths.update(m["path"] for m in e["midi_files"])
        if args.reference == "written":input_paths.add(written[(e["protocol"],e["song"])])
    input_hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(input_paths)}
    run = {"created_utc":datetime.now(timezone.utc).isoformat(),"reference":args.reference,
           "performances":len(entries),"eligible_performances":eligible,"is_smoke":bool(args.limit),
           "expected_methods":17,"seed":args.seed,"particle_config":P02,"phrase_parameters":pipeline.PARAMETERS,
           "melody_strengths":[1.,3.],"matchers":list(matchers.METHODS),"sources":source_hashes,
           "input_hashes":input_hashes,"performance_ids":[e["performance_id"] for e in entries],
           "inference_boundary":"No timed annotations; reference memory uses past harmonic-anchor estimates. Planned solo roles are filtered using the anchor estimate, never actual section timestamps.",
           "melodic_only_boundary":"Melodic updates only: the past-memory coordinate anchors and role masks still depend on the independent harmonic follower.",
           "score_scaling":"Uncalibrated [0,1] similarity; reward above .75, or max(.75, reference median) with at least8 references, normalized by remaining range. Up to8 distinct recurring locations; Gaussian sigma .75beat. Positive relative potential, uncovered positions neutral.",
           "metrics":"Same 200ms frame position metrics as prior study; head/solo scopes are evaluation-only. Equal song macro within protocol."}
    json_write(out/"run.json",run)
    rows, audits, failures = [], [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_one,e,catalogs[(e["protocol"],e["song"])],
                               written.get((e["protocol"],e["song"])) if args.reference=="written" else None,
                               out,args.seed,args.prefix_check and i<2):e for i,e in enumerate(entries)}
        for future in as_completed(pending):
            entry = pending[future]
            try:
                result,audit = future.result()
                rows.extend(result);audits.append(audit)
                benchmark.csv_write(out/"metrics.csv",sorted(rows,key=lambda r:(r["performance_id"],r["method"],r["scope"])))
                event = {"performance_id":entry["performance_id"],"status":"complete","seconds":time.perf_counter()-start}
            except Exception as error:
                event = {"performance_id":entry["performance_id"],"status":"failed","error":repr(error)}
                failures.append(event)
            with (out/"events.jsonl").open("a") as handle:handle.write(json.dumps(event)+"\n")
            print(f'{len(audits)+len(failures)}/{len(entries)} performances; {len(rows)} rows; {time.perf_counter()-start:.1f}s',flush=True)
    changes = [p for p,digest in source_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
    input_changes = [p for p,digest in input_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
    summary = {"complete":not args.limit and not failures and not changes and not input_changes,
               "scope_finished":not failures,"performances":len(audits),"eligible_performances":eligible,
               "metric_rows":len(rows),"tracking_runs":sum(r["scope"]=="all" for r in rows),
               "seconds":time.perf_counter()-start,"source_changes":changes,"input_changes":input_changes,"failures":failures}
    json_write(out/"summary.json",summary);json_write(out/"audits.json",audits)
    print(json.dumps(summary),flush=True)
    if failures or changes or input_changes:raise SystemExit(1)


if __name__ == "__main__":
    main()
