"""Independently audit saved melody-study runs without changing experiment code.

Recompute every metric from its trace, reconstruct evaluation targets/scopes from
the frozen annotations, check source/input provenance, reproduce saved original
p02 controls for each seed, and compare common-cohort controls across references.
Smoke runs may be scope-finished without claiming complete dataset coverage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DEFAULT_RUNS = ("01_past_all", "02_written", "03_past_seed1", "04_past_seed2",
                "05_written_seed1", "06_written_seed2")
ORIGINAL = {0:"05_continuous_tempo", 1:"08_particles_seed1", 2:"09_particles_seed2"}
METRICS = ("bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats")
TRACE_KEYS = ("time", "estimated_beats", "estimated_bar", "reference_beats", "reference_bar", "estimated_bpm")
SCOPES = ("all", "head_in", "head_out", "solo")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


class Audit:
    def __init__(self):
        self.errors = []
        self.checked = 0
        self.hash_cache = {}

    def check(self, condition, description):
        self.checked += 1
        if not condition:
            self.errors.append(description)

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.hash_cache:
            self.hash_cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.hash_cache[path]

    def check_hash(self, path, expected, label):
        path = Path(path)
        self.check(path.is_file(), f"{label}: missing {path}")
        if path.is_file():
            self.check(self.digest(path) == expected, f"{label}: hash mismatch {path}")


def targets(entry):
    """Reconstruct frame coverage, unfolded q, and scope masks without MIDI IO."""
    annotation = read_json(entry["annotation_path"])
    bars = sorted((r for r in annotation["bars"] if float(r["end_time"]) > float(r["start_time"])),
                  key=lambda r:(float(r["start_time"]),int(r["bar"])))
    first = next(i for i,r in enumerate(bars) if int(r["measure"]) != 0)
    bars = bars[first:]
    start, end = float(bars[0]["start_time"]), float(bars[-1]["end_time"])
    t = np.arange(int(np.ceil(start/.1-1e-9))*.1, end, .1)
    starts = np.array([float(r["start_time"]) for r in bars])
    ends = np.array([float(r["end_time"]) for r in bars])
    bar = np.searchsorted(starts,t,side="right")-1
    valid = (bar >= 0) & (t < ends[np.maximum(bar,0)])
    t, bar = t[valid][::2], bar[valid][::2]
    numerator, denominator = (int(x) for x in entry["time_signature"].split("/"))
    bpb = numerator*4./denominator
    q = (bar+(t-starts[bar])/(ends[bar]-starts[bar]))*bpb
    masks = {"all":np.ones(len(t),bool), **{s:np.zeros(len(t),bool) for s in SCOPES[1:]}}
    for section in annotation.get("sections",[]):
        within = (t >= float(section["start_time"])) & (t < float(section["end_time"]))
        for scope in SCOPES[1:]:
            if str(section.get("is_"+scope,"False")).lower() == "true":
                masks[scope] |= within
    return t, bar, q, masks


def validate_run(path, audit, manifest, original_rows):
    config, summary = read_json(path/"run.json"), read_json(path/"summary.json")
    errors_before = len(audit.errors)
    ids = config["performance_ids"]
    audit.check(len(ids) == len(set(ids)) == config["performances"], f"{path.name}: performance IDs")
    audit.check(summary["scope_finished"] and not summary["failures"], f"{path.name}: unfinished/failed scope")
    audit.check(not summary["source_changes"] and not summary["input_changes"], f"{path.name}: recorded changes")
    if config["is_smoke"]:
        audit.check(not summary["complete"], f"{path.name}: smoke incorrectly marked complete")
    else:
        expected = 75 if config["reference"] == "past" else 6
        audit.check(summary["complete"] and len(ids) == config["eligible_performances"] == expected,
                    f"{path.name}: full cohort completeness")
    for index,(source,digest) in enumerate(config["sources"].items()):
        audit.check_hash(source,digest,f"{path.name} current source")
        audit.check_hash(path/"sources"/f"{index:02d}_{Path(source).name}",digest,f"{path.name} source snapshot")
    for source,digest in config["input_hashes"].items():
        audit.check_hash(source,digest,f"{path.name} input")

    methods = {"harmonic_only","no_observations"} | {
        f"{m}__{suffix}" for m in config["matchers"] for suffix in ("melodic_updates","fused1","fused3")}
    audit.check(len(methods) == config["expected_methods"] == 17,f"{path.name}: expected methods")
    rows = read_csv(path/"metrics.csv")
    row_index = {(r["performance_id"],r["method"],r["scope"]):r for r in rows}
    audit.check(len(rows) == len(row_index) == summary["metric_rows"],f"{path.name}: unique metric rows")
    audit.check({r["performance_id"] for r in rows} == set(ids),f"{path.name}: metric cohort")
    audits = read_json(path/"audits.json")
    audit.check(len(audits) == len(ids) and {r["performance_id"] for r in audits} == set(ids),f"{path.name}: audit cohort")
    audit.check(all(r["neutral_reproduction"] for r in audits),f"{path.name}: neutral reproduction flags")
    verified = {r["performance_id"] for r in audits if r["prefix_verified"]}
    if config["seed"] == 0:
        audit.check(verified == set(ids[:2]),f"{path.name}: first-two prefix audit flags")
    else:
        audit.check(not verified,f"{path.name}: unexpected claimed seed prefix checks")
    for r in audits:
        reference = r["reference_file"]
        if config["reference"] == "written":
            audit.check(reference in config["input_hashes"],f"{path.name}: written reference not frozen")
            record = read_json(reference)
            audit.check(record["status"] == "usable" and len(record["notes"]) > 0,f"{path.name}: unusable written reference")
        else:
            audit.check(reference is None,f"{path.name}: unexpected written input")

    trace_count = 0
    expected_rows = 0
    baseline_count = 0
    max_metric_error = 0.
    controls = {}
    seed = config["seed"]
    old_run = BASE/"improvement_study"/"runs"/ORIGINAL[seed]
    if seed not in original_rows:
        original_rows[seed] = {r["performance_id"]:r for r in read_csv(old_run/"metrics.csv")
                              if r["method"] == "p02_particles_slow" and r["take"] == "preferred"
                              and r["scheme"] == "triad" and float(r["initial_tempo_factor"]) == 1.}
    for performance_id in ids:
        entry = manifest[performance_id]
        protocol,song,take = performance_id.split("/")
        directory = path/"traces"/protocol/song
        found_methods = {f.stem for f in directory.glob("*.npz")}
        audit.check(found_methods == methods,f"{path.name}/{performance_id}: unique trace methods")
        t, reference_bar, reference_q, scope_masks = targets(entry)
        for method in sorted(methods):
            tracepath = directory/f"{method}.npz"
            with np.load(tracepath,allow_pickle=False) as archive:
                trace = {k:archive[k] for k in archive.files}
            trace_count += 1
            for key in (*TRACE_KEYS,"melody_available",*("scope_"+s for s in SCOPES)):
                audit.check(trace[key].shape == t.shape and np.isfinite(trace[key]).all(),f"{path.name}/{performance_id}/{method}: invalid {key}")
            for key,expected in (("time",t),("reference_bar",reference_bar),("reference_beats",reference_q)):
                audit.check(np.array_equal(trace[key],expected),f"{path.name}/{performance_id}/{method}: reference {key}")
            for scope,mask in scope_masks.items():
                audit.check(np.array_equal(trace["scope_"+scope],mask),f"{path.name}/{performance_id}/{method}: scope {scope}")
                if not mask.any():
                    audit.check((performance_id,method,scope) not in row_index,f"{path.name}: empty scope has row")
                    continue
                expected_rows += 1
                row = row_index[(performance_id,method,scope)]
                error = np.abs(trace["estimated_beats"][mask]-reference_q[mask])
                recomputed = dict(frames=len(error),bar_accuracy=float((trace["estimated_bar"][mask]==reference_bar[mask]).mean()),
                                  within2beat_accuracy=float((error <= 2.).mean()),mean_abs_beats=float(error.mean()),
                                  median_abs_beats=float(np.median(error)),melodic_coverage=float(trace["melody_available"][mask].mean()))
                for key,value in recomputed.items():
                    delta = abs(float(row[key])-value)
                    max_metric_error = max(max_metric_error,delta)
                    audit.check(delta <= 1e-12,f"{path.name}/{performance_id}/{method}/{scope}: metric {key}")
                audit.check(row["reference"] == config["reference"] and int(row["seed"]) == seed and row["take"] == take,
                            f"{path.name}/{performance_id}/{method}/{scope}: row metadata")
            if method in ("harmonic_only","no_observations"):
                controls[(performance_id,method)] = {k:trace[k] for k in TRACE_KEYS}
            if method == "harmonic_only":
                original_path = old_run/"traces"/protocol/song/take/"p02_particles_slow.npz"
                with np.load(original_path,allow_pickle=False) as old:
                    for key in TRACE_KEYS:
                        audit.check(np.array_equal(trace[key],old[key]),f"{path.name}/{performance_id}: original p02 trace {key}")
                previous = original_rows[seed][performance_id]
                current = row_index[(performance_id,method,"all")]
                for key in ("frames",*METRICS):
                    audit.check(float(current[key]) == float(previous[key]),f"{path.name}/{performance_id}: original p02 metric {key}")
                baseline_count += 1
    audit.check(trace_count == summary["tracking_runs"] == len(ids)*17,f"{path.name}: trace counts")
    audit.check(len(list((path/"traces").rglob("*.npz"))) == trace_count,f"{path.name}: unexpected trace files")
    audit.check(expected_rows == len(rows),f"{path.name}: expected scope rows")
    result = {"run":str(path.resolve()),"passed":len(audit.errors)==errors_before,"is_smoke":config["is_smoke"],
              "reference":config["reference"],"seed":seed,"performances":len(ids),"traces":trace_count,
              "metric_rows":len(rows),"maximum_metric_recomputation_difference":max_metric_error,
              "source_files":len(config["sources"]),"input_files":len(config["input_hashes"]),
              "prefix_verified_performance_ids":sorted(verified),"original_p02_exact_comparisons":baseline_count,
              "original_p02_run":str(old_run.resolve()),"errors":audit.errors[errors_before:]}
    return result,controls


def written_provenance(audit):
    path = HERE/"references"/"written"/"manifest.json"
    source = read_json(path)
    before = len(audit.errors)
    for p,k in ((source["source_path"],"source_sha256"),(BASE/"data/manifest.json","catalog_manifest_sha256"),
                (HERE/"prepare_written.py","script_sha256")):
        audit.check_hash(p,source[k],"written provenance")
    usable = []
    for item in source["songs"]:
        for stem in ("xml","event","annotation","aligned"):
            if stem+"_sha256" in item:
                audit.check_hash(item[stem+"_path"],item[stem+"_sha256"],"written artifact")
        if item["status"] == "usable":
            usable.append(item["protocol"]+"/"+item["song"]+"/preferred")
    audit.check(len(usable) == source["usable_songs"] == 6,"written usable cohort")
    audit.check(source["query_data_used"] is False and source["annotation_timestamp_fields_used"] is False,
                "written provenance claims query/timestamp use")
    return {"passed":len(audit.errors)==before,"usable_performance_ids":sorted(usable),"manifest_sha256":audit.digest(path),
            "source_parquet_sha256":source["source_sha256"],"errors":audit.errors[before:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs",nargs="*",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"validation.json")
    parser.add_argument("--preservation-manifest",type=Path,default=BASE/"improvement_study"/"data_fingerprint.json")
    args = parser.parse_args()
    paths = args.runs or [HERE/"runs"/r for r in DEFAULT_RUNS]
    started = time.perf_counter()
    audit = Audit()
    manifest = {r["performance_id"]:r for r in read_json(BASE/"data/manifest.json")["performances"]}
    result = {"status":"running","created_utc":datetime.now(timezone.utc).isoformat(),"runs":[],"errors":[],
              "validator_source_sha256":audit.digest(Path(__file__))}
    write_json(args.output,result)
    controls,old_rows = {},{}
    try:
        for path in paths:
            entry,c = validate_run(path,audit,manifest,old_rows)
            result["runs"].append(entry)
            controls[(entry["seed"],entry["reference"])] = c
            result["errors"] = audit.errors
            write_json(args.output,result)
            print(f'{path.name}: {entry["traces"]} traces, {entry["metric_rows"]} rows; passed={entry["passed"]}',flush=True)
        paired = 0
        for seed in ORIGINAL:
            past,written = controls.get((seed,"past")),controls.get((seed,"written"))
            if past is None or written is None:
                continue
            for key,trace in written.items():
                audit.check(key in past,f"seed{seed}: written control missing from past cohort {key}")
                if key in past:
                    for field in TRACE_KEYS:
                        audit.check(np.array_equal(trace[field],past[key][field]),f"seed{seed}: cohort control differs {key}/{field}")
                    paired += 1
        result["same_cohort_control_trace_pairs"] = paired
        result["written_source_provenance"] = written_provenance(audit)
        if args.preservation_manifest:
            frozen = read_json(args.preservation_manifest)
            if "files" in frozen:
                frozen = frozen["files"]
            elif "hashes" in frozen:
                frozen = frozen["hashes"]
            for path,digest in frozen.items():
                audit.check_hash(path,digest,"preserved prior artifact")
            result["prior_preservation_manifest"] = str(args.preservation_manifest.resolve())
            result["prior_preserved_files_checked"] = len(frozen)
        result["total_traces"] = sum(r["traces"] for r in result["runs"])
        result["total_metric_rows"] = sum(r["metric_rows"] for r in result["runs"])
        result["original_p02_exact_comparisons"] = sum(r["original_p02_exact_comparisons"] for r in result["runs"])
        if {p.name for p in paths} == set(DEFAULT_RUNS):
            audit.check(result["total_traces"] == 4131,"all-six trace total")
            audit.check(result["total_metric_rows"] == 16524,"all-six metric total")
            audit.check(paired == 36,"all-six common-cohort controls")
        result["causality_audit"] = {
            "method":"read-only code review plus saved seed0 first-two prefix audit flags",
            "past_memory":"Earlier nonoverlapping phrases use independent causal harmonic-anchor q; no truth q tagging.",
            "role_mask":"Planned score-coordinate solo roles, selected using anchor-estimated position; timed sections are evaluation scopes only.",
            "known_inputs":"Unfolded plan/roles, catalog BPM, annotation-derived synchronized initial boundary and evaluation horizon are supplied assumptions.",
            "melodic_updates_limit":"Past-memory coordinates and both reference conditions' role gating depend on the harmonic anchor; melodic_updates is not a fully harmony-free system.",
            "score_limit":"[0,1] similarities and positive heuristic position rewards are uncalibrated; five methods do not have matched null score distributions.",
            "timing_limit":"harmonic_only inference_seconds is cached-result retrieval time, not its actual inference cost; timing columns must not be compared as total end-to-end latency."}
    except Exception as error:
        audit.errors.append(f"Validator exception: {type(error).__name__}: {error}")
    result.update(status="passed" if not audit.errors else "failed",complete=True,checks=audit.checked,
                  unique_files_hashed=len(audit.hash_cache),seconds=time.perf_counter()-started,errors=audit.errors)
    write_json(args.output,result)
    print(json.dumps({k:result.get(k) for k in ("status","checks","total_traces","total_metric_rows","original_p02_exact_comparisons","seconds","errors")},indent=2))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
