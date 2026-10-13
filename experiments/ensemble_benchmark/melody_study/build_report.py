"""Build a compact report; aggregate each song first, then three random seeds."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = ["01_past_all", "02_written", "03_past_seed1", "04_past_seed2", "05_written_seed1", "06_written_seed2"]
METRICS = ["bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats", "melodic_coverage"]
NAMES = {"pc_cosine":"Pitch-class histogram (ignores order)", "pc_edit":"Pitch-class edit distance",
         "interval_edit":"Pitch-interval edit distance", "interval_dtw":"Interval + rhythm DTW",
         "interval_ngrams":"Short interval/rhythm n-grams", "harmonic_only":"Bass harmony only",
         "no_observations":"Same particle filter without observations"}


def display_method(method):
    if "__" not in method:
        return NAMES[method]
    stem, condition = method.split("__")
    return NAMES[stem] + {"fused1":" + harmony, strength 1", "fused3":" + harmony, strength 3",
                         "melodic_updates":"; melodic updates only"}[condition]


def table(frame):
    frame = frame.copy()
    for col in frame:
        if col in ["bar_accuracy","within2beat_accuracy","melodic_coverage","bar_min","bar_max"]:
            frame[col] = frame[col].map(lambda x:f"{100*x:.2f}%")
        elif col in ["mean_abs_beats","median_abs_beats"]:
            frame[col] = frame[col].map(lambda x:f"{x:.3f}")
        elif col == "gain_pp":
            frame[col] = frame[col].map(lambda x:f"{x:+.2f}")
        elif col == "method":
            frame[col] = frame[col].map(display_method)
    frame = frame.rename(columns={"method":"Method","protocol":"Protocol","scope":"Evaluated section",
        "reference":"Reference","bar_accuracy":"Correct bar","within2beat_accuracy":"Within 2 beats",
        "mean_abs_beats":"Mean error (beats)","median_abs_beats":"Mean of song median errors",
        "melodic_coverage":"Melody evidence available","gain_pp":"Bar gain (pp)",
        "bar_min":"Lowest seed","bar_max":"Highest seed"})
    return '<div class="scroll">'+frame.to_html(index=False,border=0,escape=True)+'</div>'


def main():
    frames, ledger = [], []
    for run in RUNS:
        p = HERE/"runs"/run
        summary = json.loads((p/"summary.json").read_text())
        assert summary["complete"] and not summary["failures"]
        d = pd.read_csv(p/"metrics.csv");frames.append(d)
        ledger.append({"Run":run,"Performances":summary["performances"],"Tracking runs":summary["tracking_runs"],"Status":"Complete"})
    data = pd.concat(frames,ignore_index=True)
    key = ["reference","protocol","method","scope"]
    seed = data.groupby([*key,"seed"],as_index=False)[METRICS].mean()
    macro = seed.groupby(key,as_index=False)[METRICS].mean()
    ranges = seed.groupby(key).bar_accuracy.agg(bar_min="min",bar_max="max").reset_index()
    macro = macro.merge(ranges,on=key,validate="one_to_one")
    bass = macro[macro.method == "harmonic_only"][["reference","protocol","scope","bar_accuracy"]].rename(columns={"bar_accuracy":"baseline_bar"})
    macro = macro.merge(bass,on=["reference","protocol","scope"],validate="many_to_one")
    macro["gain_pp"] = 100*(macro.bar_accuracy-macro.baseline_bar)
    ids = set(data[data.reference=="written"].performance_id)
    matched = data[data.performance_id.isin(ids)].groupby(["reference","method","scope","seed"],as_index=False)[METRICS].mean()
    matched = matched.groupby(["reference","method","scope"],as_index=False)[METRICS].mean()
    primary = macro[(macro.scope=="all") & (macro.method.str.endswith("__fused1") | (macro.method=="harmonic_only"))]
    strong = macro[(macro.scope=="all") & macro.method.str.endswith("__fused3")]
    chosen = primary[primary.method != "harmonic_only"].sort_values("bar_accuracy",ascending=False).groupby(["reference","protocol"]).head(1)
    out = HERE/"results";out.mkdir(exist_ok=True)
    macro.to_csv(out/"aggregate_three_seeds.csv",index=False)
    seed.to_csv(out/"per_seed_aggregate.csv",index=False)
    matched.to_csv(out/"matched_six_songs.csv",index=False)
    summary = {"scope":{"past_songs":75,"written_songs":6,"written_async":4,"written_sync":2,
                        "seeds":[0,1,2],"matchers":5,"tracking_runs":sum(x["Tracking runs"] for x in ledger)},
               "primary_strength1":primary.to_dict("records"),"strength3":strong.to_dict("records"),
               "best_strength1_by_protocol":chosen.to_dict("records"),
               "matched_six_songs":matched.to_dict("records"),"ledger":ledger,
               "boundary":"Exploratory simplified matching variants with uncalibrated score scales; not labeled lick detection or an external generalization result."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    html = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Melodic matching pilot</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px 60px;background:#fbfcfa;color:#22323a}h1{font-size:32px}h2{margin-top:36px;font-size:23px}h3{font-size:18px}a{color:#176477}p,li{max-width:100ch}table{border-collapse:collapse;font-size:14px;background:white;width:100%}td,th{padding:9px 12px;border-bottom:1px solid #dce4e2;text-align:left;font-variant-numeric:tabular-nums}th{background:#edf3f1}.scroll{overflow:auto;margin:18px 0}.note{background:#eef5f3;padding:16px 20px;border-left:4px solid #43796d}pre{background:#edf1f0;padding:18px;overflow:auto}code{font-size:.9em}</style>
<h1>Melodic matching pilot</h1><p>6 October 2026 · five matchers · two reference conditions · three particle-filter seeds.</p>
<p class="note"><strong>Finding:</strong> written melody helps on the six-song subset, while earlier-phrase memory adds only small and inconsistent gains across all 75 songs. With a common evidence strength of 1, written-melody DTW raises mean bar accuracy from 69.17% to 77.98%; the order-free histogram reaches 77.94%. This does not establish a unique advantage for recognizing note order or licks.</p>
<p>The best earlier-memory results at strength 1 are 68.26% versus 67.42% for async (pitch-class edit distance), and 73.06% versus 72.78% for sync (interval edit distance). These small exploratory gains need further validation. With stronger influence, the written histogram and pitch-class edit methods both reach about 78.9%, while stronger earlier-memory matching can hurt.</p>
<h2>What was compared?</h2><ul>
<li><strong>Pitch-class histogram:</strong> compare the pitch content of nine notes while ignoring order. This is the control for whether sequence information adds value.</li>
<li><strong>Pitch-class edit distance:</strong> compare the ordered notes, allowing insertions, deletions and substitutions. Octave differences are ignored.</li>
<li><strong>Interval edit distance:</strong> compare successive pitch movements. A transposed phrase can still match.</li>
<li><strong>Interval/rhythm DTW:</strong> allow flexible alignment between pitch movements and relative inter-onset timing.</li>
<li><strong>Short n-grams:</strong> compare repeated four-note units represented by pitch intervals and relative rhythm.</li>
</ul><p>These are simplified implementations of established algorithm families. They are not exact reproductions of the cited papers. Nine-note windows contain only notes already received, and use completed onset intervals rather than future note durations. Written and performed rhythms can differ through swing and ornamentation; no swing correction was fitted.</p>
<p>For the primary comparison each melodic matcher uses the same evidence strength of 1. Strength 3 is a sensitivity check. All variants use the same p02 particle settings, bass-harmony feature and starting information. The table means average songs equally within protocol, then average seeds 0, 1 and 2. This isolates adding melodic observations to one particle backbone; it is not a new comparison against the previous async beam (91.4%) or the correctly initialized fixed clock (99.9%).</p>
<h2>Earlier-performance phrase memory: all 75 songs</h2>
<p>The system records earlier non-solo horn phrases and associates them with the independent harmonic follower's past position estimates. A query never matches itself or an overlapping window: reference material must end at least two nominal beats before the query begins. Repeated chart measures support all corresponding occurrences, leaving chorus identity to the temporal tracker.</p>''']
    for protocol in ["async","sync"]:
        d = primary[(primary.reference=="past") & (primary.protocol==protocol)]
        html += [f'<h3>{protocol} ({49 if protocol=="async" else 26} songs)</h3>',
                 table(d[["method","bar_accuracy","gain_pp","bar_min","bar_max","mean_abs_beats","melodic_coverage"]])]
    html += ['''<p>The earlier-memory gains are small and depend on method, strength and protocol. The reference positions can inherit mistakes from the harmonic anchor. A repeated phrase can also fit several chart locations. This experiment does not establish a robust standalone lick tracker.</p>
<h2>Written melody: compare the same six songs</h2>
<p>Written references were extracted from the existing local <a href="https://huggingface.co/datasets/PRAIG/JAZZMUS">JAZZMUS dataset</a>. Twelve titles overlap; six pass the declared chart checks: Freddie Freeloader, Impressions, St. Thomas, Take Five, In a Sentimental Mood and Nardis. Key and position are aligned using untimed chord sequences only, with no performance notes or timestamps.</p>
<p>The tables below compare both reference types on exactly these six songs (four async, two sync). These means are not directly comparable with the full 75-song means. Different chart editions may still differ in melody even when their chord sequences agree.</p>''']
    d = matched[(matched.scope=="all") & (matched.method.str.endswith("__fused1") | (matched.method=="harmonic_only"))]
    html += [table(d[["reference","method","bar_accuracy","within2beat_accuracy","mean_abs_beats","melodic_coverage"]]),
             '<h3>Where does written melody help?</h3>']
    d = matched[(matched.reference=="written") & matched.method.isin(["harmonic_only","pc_cosine__fused1","pc_edit__fused1","interval_dtw__fused1"]) & (matched.scope!="all")]
    html += [table(d[["scope","method","bar_accuracy","mean_abs_beats","melodic_coverage"]]),
             '''<p>Head-in/head-out/solo labels are used only to analyze outputs. Better solo-section accuracy can reflect an improved position inherited from the preceding head; it does not prove the written melody directly matches an improvised solo. An order-free pitch histogram is competitive here, so these results do not establish that note ordering or a lick-specific representation provides a unique benefit.</p>
<h2>Stronger melodic influence</h2><p>The following sensitivity uses strength 3. No per-song best method is selected. A ranking change or a larger mean error is evidence of a tradeoff, not a reason to hide that result.</p>''',
             table(strong[["reference","protocol","method","bar_accuracy","gain_pp","mean_abs_beats"]]),
             '''<h2>Input selection and claim boundaries</h2><ul>
<li>The local band MIDI contains bass and horn transcriptions; piano MIDI is absent. Bass remains the fixed harmonic input. Catalog lead-horn tracks provide melodic queries.</li>
<li>Planned solo roles come from the untimed performance plan. A horn is excluded when the independent harmonic follower estimates that its planned solo is happening. The actual timed solo annotation is never used for live selection. Incorrect position estimates can therefore cause incorrect masking.</li>
<li>Memory-based melodic-update-only ablations still use a harmonic follower for memory anchors and role selection. They are not independent melody-only cold-start followers. Written-query role gating also uses that fixed anchor.</li>
<li>Each window has nine notes, a maximum span of 32 nominal beats and maximum internal gap of eight. Evidence expires two nominal beats after the latest onset. This is variable-duration context with no future lookahead.</li>
<li>Similarity scores are uncalibrated. A positive location reward is issued above 0.75; with at least eight references, the median reference score can raise the threshold. Up to eight distinct locations receive Gaussian support with 0.75-beat width. Common numerical thresholds do not guarantee equal false-match rates across algorithms.</li>
<li>Interval methods allow transposition but do not jointly infer transposed harmonic contexts. Harmonic compatibility is handled by the separate bass branch.</li>
<li>The dataset has no verified lick-occurrence labels. We measure position-following outcomes and evidence availability, not lick-detection precision or recall. Evidence availability does not mean the match is correct.</li>
<li>Known starting position/time and approximate tempo are supplied. The async fixed clock remains a strong control. No new claim is made about unseen-song generalization, unknown starts, unplanned jumps or live end-to-end latency.</li>
<li>Independent MIDI/audio/annotation time-origin verification remains outstanding. Automatic transcription errors and written-edition differences can affect these provisional results.</li>
</ul>
<h2>Metrics</h2><p>Every 200 ms, compare predicted and annotated unfolded bar occurrence. Correct-bar percentage measures time in the correct bar, not the percentage of perfectly followed whole bars. Within-two-beat rate and all-frame mean/median absolute beat errors accompany it. Fractional reference beats are interpolated inside annotated bars; they are not separately annotated beat onsets. No large errors are discarded.</p>
<h2>Reproduce and inspect</h2>''', pd.DataFrame(ledger).to_html(index=False,border=0),
             '''<p>Every run preserves pre-execution configuration/source hashes, exact source snapshots, input hashes, per-song metrics, prediction traces, event logs and completion status. Smoke runs remain separately identified. Existing datasets, baseline studies and proposal text were not changed. Final validation recomputed all 4,131 traces and 16,524 scope-metric rows exactly, reproduced 243 earlier harmonic-only results, and verified that all 462 previously fingerprinted input files remain unchanged.</p>
<p>Runtime fields cannot compare baseline speed with the new methods: its prediction is reused after the anchor pass, so its inference_seconds field omits computation already performed. No live-latency or relative-speed claim is made.</p><ul>
<li><a href="summary.json">Summary JSON</a> · <a href="aggregate_three_seeds.csv">All three-seed aggregates</a> · <a href="per_seed_aggregate.csv">Per-seed results</a> · <a href="matched_six_songs.csv">Matched six-song comparison</a></li>
<li><a href="../validation.json">Validation</a> · <a href="../analysis/analysis.md">Detailed seed-0 analysis</a> · <a href="../analysis/seed_analysis.json">Three-seed paired analysis</a></li>
<li><a href="../references/written/README.md">Written-reference audit and rejected matches</a> · <a href="../references/written/manifest.json">Source provenance</a></li>
<li><a href="../references.json">Algorithm literature and implementation boundaries</a> · <a href="../../improvement_study/report.html">Previous harmonic/temporal study</a></li>
</ul><p>From the workspace root, using the bundled Python environment with NumPy:</p>
<pre>python3 'Research Proposal/experiments/ensemble_benchmark/melody_study/run_study.py' --reference past --output /tmp/melody-past-new-run --workers 3
python3 'Research Proposal/experiments/ensemble_benchmark/melody_study/run_study.py' --reference written --output /tmp/melody-written-new-run --workers 3
# Add --seed 1 or --seed 2 for the repeat checks.</pre>
</html>''']
    (out/"report.html").write_text("\n".join(html))
    print(json.dumps({"report":str(out/"report.html"),"tracking_runs":summary["scope"]["tracking_runs"]}))


if __name__ == "__main__":
    main()
