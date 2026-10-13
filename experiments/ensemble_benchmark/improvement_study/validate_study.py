"""Artifact-level completion checks for the logged improvement study."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import run_study as study
import rhythm
import sequence_follower
import particle_follower

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("runs",type=Path,nargs="+");parser.add_argument("--output",type=Path,default=study.HERE/"validation.json");args=parser.parse_args()
    study.checks(); rhythm_result=rhythm.checks(); sequence_result=sequence_follower.checks(); particle_result=particle_follower.checks()
    dataset=json.loads((study.HERE/"data_fingerprint.json").read_text())
    for path,digest in dataset["files"].items(): assert sha(Path(path))==digest, f"Changed data: {path}"
    results=[]; all_rows=[]
    for run in args.runs:
        summary=json.loads((run/"summary.json").read_text()); config=json.loads((run/"run.json").read_text())
        assert summary["complete"] and summary["source_changes"]==[],run
        for path,digest in config["source_hashes"].items():
            assert sha(run/"sources"/Path(path).name)==digest, f"Unrecoverable executed source: {run}/{path}"
        with (run/"metrics.csv").open() as f: rows=list(csv.DictReader(f))
        assert len(rows)==len(config["settings"])*len(config["performance_ids"])
        keys={(r["performance_id"],r["method"]) for r in rows}
        expected={(perf,s["method"]) for perf in config["performance_ids"] for s in config["settings"]}
        assert keys==expected and len(keys)==len(rows)
        for row in rows:
            path=run/"traces"/row["protocol"]/row["song"]/row["take"]/(row["method"]+".npz")
            a=np.load(path); error=np.abs(a["estimated_beats"]-a["reference_beats"])
            assert len(error)==int(row["frames"])
            assert np.isfinite(error).all() and np.all(np.diff(a["time"])>0)
            direct={"bar_accuracy":float(np.mean(a["estimated_bar"]==a["reference_bar"])),
                    "within2beat_accuracy":float(np.mean(error<=2)),"mean_abs_beats":float(error.mean()),"median_abs_beats":float(np.median(error))}
            for metric,value in direct.items():assert abs(value-float(row[metric]))<1e-11,(row["performance_id"],row["method"],metric)
        by={(r["performance_id"],r["method"]):r for r in rows}
        no_drums_checked=0
        for r in rows:
            if r["protocol"]=="async" and r["method"] in ("r08_joint_drums","r09_joint_drums_broad"):
                pair="r06_joint_bass_root" if r["method"]=="r08_joint_drums" else "r07_joint_bass_root_broad"
                other=by[(r["performance_id"],pair)]
                for metric in ("bar_accuracy","within2beat_accuracy","mean_abs_beats","median_abs_beats"):
                    assert float(r[metric])==float(other[metric]),"Absent drums changed prediction"
                no_drums_checked+=1
        results.append({"run":str(run.resolve()),"rows":len(rows),"traces_verified":len(rows),"absent_drum_controls":no_drums_checked})
        all_rows+=rows
    with (study.BASE/"results/position_metrics.csv").open() as f: baseline=list(csv.DictReader(f))
    baseline={r["performance_id"]:r for r in baseline if r["take"]=="preferred" and r["method"]=="yamaha_active" and r["scheme"]=="triad" and float(r["initial_tempo_factor"])==1.}
    reproduction=[r for r in all_rows if r["method"]=="g00_reproduce"]
    assert len(reproduction)==75
    for r in reproduction:
        for metric in ("bar_accuracy","within2beat_accuracy","mean_abs_beats","median_abs_beats"):
            assert float(r[metric])==float(baseline[r["performance_id"]][metric])
    result={"passed":True,"data_files_unchanged":len(dataset["files"]),"baseline_reproductions":75,
            "rhythm_checks":rhythm_result,"sequence_checks":sequence_result,"particle_checks":particle_result,"runs":results,
            "claim_boundary":"Checks establish causal implementation and artifact consistency, not independent AMT annotation alignment, live latency, or external generalization."}
    study.write_json(args.output,result);print(json.dumps(result,indent=2))

if __name__=="__main__": main()
