"""Compare complete runs, record non-oracle setting selection, and write HTML.

Leave-one-song-out selection chooses a setting using OTHER songs' scores. This
is internal exploratory validation, not an untouched external test. Candidates
are fixed at evaluation time but later designs were informed by this corpus.
Neither selection rule feeds test annotations to a follower at inference.
Per-song oracle maxima are deliberately not reported as systems.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import html
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
METRICS=("bar_accuracy","within2beat_accuracy","mean_abs_beats","median_abs_beats")

def read(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in METRICS:
            r[k]=float(r[k])
    return rows

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def grouped(rows,keys):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,values in sorted(groups.items()):
        result.append({**dict(zip(keys,key)),"songs":len(values),**{k:float(np.mean([r[k] for r in values])) for k in METRICS}})
    return result

def selection(rows,condition_on_protocol):
    by_song=defaultdict(dict)
    for r in rows: by_song[r["song"]][r["method"]]=r
    candidates=sorted(set.intersection(*(set(v) for v in by_song.values())))
    if not candidates: raise ValueError("No complete shared candidate cohort")
    output=[]
    for song,models in sorted(by_song.items()):
        protocol=next(iter(models.values()))["protocol"]
        train=[r for r in rows if r["song"]!=song and (not condition_on_protocol or r["protocol"]==protocol)]
        means=grouped(train,["method","protocol"])
        score=defaultdict(list)
        for row in means:
            if row["method"] in candidates: score[row["method"]].append(row["bar_accuracy"])
        chosen=min(candidates,key=lambda name:(-np.mean(score[name]),name))
        row=dict(models[chosen])
        row["selected_method"]=chosen
        row["method"]="loo_protocol" if condition_on_protocol else "loo_global"
        row["selection_training_songs"]=len({r["song"] for r in train})
        row["selection_objective"]="mean of protocol mean bar accuracies on other compositions"
        output.append(row)
    return output

def paired(rows,baseline):
    lookup={(r["song"],r["protocol"]):r for r in baseline}
    groups=defaultdict(list)
    for r in rows:
        old=lookup[(r["song"],r["protocol"])]
        groups[(r["method"],r["protocol"])].append((r,old))
    rng=np.random.default_rng(81231)
    result=[]
    for (method,protocol),pairs in sorted(groups.items()):
        delta=np.array([new["bar_accuracy"]-old["bar_accuracy"] for new,old in pairs])
        boot=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
        result.append({"method":method,"protocol":protocol,"songs":len(delta),"mean_bar_gain":float(delta.mean()),
                       "bootstrap_low":float(np.quantile(boot,.025)),"bootstrap_high":float(np.quantile(boot,.975)),
                       "songs_improved":int((delta>1e-10).sum()),"songs_worse":int((delta < -1e-10).sum()),
                       "songs_tied":int((abs(delta)<=1e-10).sum())})
    return result

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def main():
    p=argparse.ArgumentParser();p.add_argument("runs",type=Path,nargs="+");p.add_argument("--output",type=Path,default=HERE/"comparison");a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    rows=[];settings={};provenance=[]
    for run in a.runs:
        summary=json.loads((run/"summary.json").read_text());config=json.loads((run/"run.json").read_text())
        if not summary["complete"] or config["scope"]!="preferred":raise ValueError("Only complete preferred-take runs can be selected")
        rows+=read(run/"metrics.csv")
        settings.update({s["method"]:s for s in config["settings"]})
        provenance.append({"run":str(run.resolve()),"summary":summary})
    identities=[(r["song"],r["method"]) for r in rows]
    if len(identities)!=len(set(identities)):raise ValueError("Repeated song/setting")
    cohorts=defaultdict(set)
    for r in rows:cohorts[r["method"]].add(r["song"])
    if len({tuple(sorted(v)) for v in cohorts.values()})!=1:raise ValueError("Unequal candidate song coverage")
    selected=selection(rows,False)+selection(rows,True)
    write(out/"selected_metrics.csv",selected)
    macro=grouped(rows+selected,["method","protocol"])
    write(out/"aggregate_metrics.csv",macro)
    baseline=[r for r in rows if r["method"]=="g00_reproduce"]
    gain=paired(rows+selected,baseline)
    write(out/"paired_gains.csv",gain)
    meta={"runs":provenance,"candidate_settings":settings,"songs":len(cohorts[next(iter(cohorts))]),
          "candidate_count":len(cohorts),"selection":"Leave-one-song-out setting selection, never oracle per-song choice",
          "interpretation":"Exploratory internal validation; corpus has already been inspected; no external test or significance claim",
          "bootstrap":"2000 paired song bootstrap draws within protocol, fixed RNG81231, percentile95% interval; exploratory selection uncertainty not accounted for"}
    (out/"comparison.json").write_text(json.dumps(meta,indent=2)+"\n")
    doc=['<!doctype html><meta charset="utf-8"><title>Bar-following improvement study</title><style>body{font:16px/1.55 system-ui;color:#172536;max-width:1200px;margin:40px auto;padding:0 24px}h1,h2{line-height:1.2}.notice{background:#fff5df;padding:16px;border-left:4px solid #b57616}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d8e1e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eff4f8}summary{cursor:pointer;font-weight:650}code{font-size:13px}</style><h1>Improving bar following</h1>']
    doc+=['<p class="notice">Exploratory causal MIDI experiment. All methods receive a known unfolded chord plan, starting position/time, and approximate catalog BPM. Piano MIDI is absent. These are experimental implementations, not published full-system reproductions. Independent MIDI/audio/annotation time-origin verification remains outstanding.</p>',
           '<p>Every table averages per-song metrics equally and separates async from sync. A better exact-bar percentage may coexist with larger catastrophic errors; inspect within-two-beat accuracy and absolute error too. Beat reference positions are interpolated inside annotated bars.</p>']
    controls=read(HERE.parent/"results/position_metrics.csv")
    controls=[r for r in controls if r["take"]=="preferred" and r["scheme"]=="triad" and float(r["initial_tempo_factor"])==1. and r["method"] in ("clock","uniform_filter")]
    control_summary=grouped(controls,["protocol","method"])
    doc.append('<h2>Timing-only controls</h2><p>The nearly perfect async clock relies on a known start and the correct fixed nominal tempo. The setting-selection tables below compare the experimental followers; these two controls are shown separately, not silently treated as worse baselines.</p>'+table(["Protocol","Control","Songs","Correct bar","Within2beats","Mean error"],[[r["protocol"],r["method"],r["songs"],f'{100*r["bar_accuracy"]:.1f}%',f'{100*r["within2beat_accuracy"]:.1f}%',f'{r["mean_abs_beats"]:.2f}'] for r in control_summary]))
    for protocol in ("async","sync"):
        sub=sorted([r for r in macro if r["protocol"]==protocol],key=lambda r:-r["bar_accuracy"])
        doc.append('<h2>'+protocol+' · all settings</h2>'+table(["Setting","Songs","Correct bar","Within2 beats","Mean error (beats)","Mean song median"],[[r["method"],r["songs"],f'{100*r["bar_accuracy"]:.1f}%',f'{100*r["within2beat_accuracy"]:.1f}%',f'{r["mean_abs_beats"]:.2f}',f'{r["median_abs_beats"]:.2f}'] for r in sub]))
    doc.append('<h2>Setting selection and consistency</h2><p><b>loo_global</b> selects a setting on all other songs, weighting the two protocol means equally. <b>loo_protocol</b> uses other songs with the same known recording protocol. Neither chooses the best setting using the evaluated song\'s labels. These are retrospective internal checks, not an untouched test set; protocol-conditioned selection is only applicable when recording conditions are known.</p>')
    for protocol in ("async","sync"):
        sub=sorted([r for r in gain if r["protocol"]==protocol],key=lambda r:-r["mean_bar_gain"])
        doc.append('<h3>'+protocol+' · paired change from original harmonic grid</h3>'+table(["Setting","Mean gain (pp)","Exploratory95% interval (pp)","Songs improved","Songs worse"],[[r["method"],f'{100*r["mean_bar_gain"]:+.1f}',f'{100*r["bootstrap_low"]:+.1f} to {100*r["bootstrap_high"]:+.1f}',r["songs_improved"],r["songs_worse"]] for r in sub]))
    doc.append('<p>The bootstrap resamples paired songs within protocol. Its intervals do not account for candidate selection or all exploratory decisions.</p><h2>What each setting changes</h2>')
    doc.append(table(["Setting","Reason","Feature","Configuration"],[[k,v["reason"],v["feature"],json.dumps(v["config"])] for k,v in sorted(settings.items())]))
    doc.append('<h2>Reproduce and inspect</h2><p>Each run retains <code>run.json</code>, <code>events.jsonl</code>, <code>summary.json</code>, <code>metrics.csv</code>, and per-song prediction traces. Failed variants remain in the comparison when execution completed. Sources and configuration are hashed before every run.</p><ul>')
    for name in ("aggregate_metrics.csv","selected_metrics.csv","paired_gains.csv","comparison.json"):
        doc.append(f'<li><a href="{name}">{name}</a></li>')
    doc.append('</ul>')
    (out/"report.html").write_text(''.join(doc))
    print(json.dumps({"report":str((out/"report.html").resolve()),"rows":len(rows),"selected_rows":len(selected),"candidates":len(cohorts),"songs":meta["songs"]}))

if __name__=="__main__": main()
