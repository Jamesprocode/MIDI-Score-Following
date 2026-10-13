"""Build the final human-readable study summary from verified saved results."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = ["01_diagnostics", "02_rhythm_sequence", "05_continuous_tempo"]
RUNS = ["01_diagnostics", "02_rhythm_sequence", "03_bass_root_alternate",
        "04_bass_root_tempo_stress", "05_continuous_tempo", "06_finalists_alternate",
        "07_finalists_tempo_stress", "08_particles_seed1", "09_particles_seed2"]
METRICS = ["bar_accuracy", "within2beat_accuracy", "mean_abs_beats", "median_abs_beats"]
LABELS = {"g00_reproduce": "Original pooled-harmony grid",
          "g06_bass_root": "Bass/root-weighted grid",
          "s06_beam_fine_tempo": "Fine-tempo sequence beam (s06)",
          "p02_particles_slow": "Slow-diffusion particles (p02, seed 0)",
          "loo_global": "Internal LOO: one setting across protocols",
          "loo_protocol": "Internal LOO: setting selected by protocol",
          "clock": "Catalog-tempo clock", "uniform_filter": "Grid without harmony"}


def table(frame, renames=None):
    frame = frame.copy()
    for col in frame:
        if col in ("bar_accuracy", "within2beat_accuracy"):
            frame[col] = frame[col].map(lambda v: f"{100*v:.2f}%")
        elif col in ("mean_abs_beats", "median_abs_beats"):
            frame[col] = frame[col].map(lambda v: f"{v:.3f}")
    frame = frame.rename(columns=renames or {})
    return '<div class="scroll">' + frame.to_html(index=False, escape=True, border=0) + '</div>'


def main():
    validation = json.loads((HERE / "validation.json").read_text())
    assert validation["passed"]
    ledger, frames = [], {}
    for run in RUNS:
        path = HERE / "runs" / run
        summary = json.loads((path / "summary.json").read_text())
        assert summary["complete"] and not summary["source_changes"]
        frame = pd.read_csv(path / "metrics.csv")
        assert len(frame) == summary["expected_rows"]
        frames[run] = frame
        ledger.append({"Run": run, "Performances": summary["performances"],
                       "Settings": summary["settings"], "Evaluations": len(frame), "Status": "Complete"})
    total = sum(len(frame) for frame in frames.values())
    assert total == sum(row["traces_verified"] for row in validation["runs"])

    aggregate = pd.read_csv(HERE / "comparison" / "aggregate_metrics.csv")
    wanted = ["g00_reproduce", "g06_bass_root", "s06_beam_fine_tempo", "p02_particles_slow"]
    fixed = aggregate[aggregate.method.isin(wanted)].copy()
    baseline = pd.read_csv(HERE.parent / "results" / "position_metrics.csv")
    controls = baseline[(baseline["take"] == "preferred") & (baseline.scheme == "triad") &
                        (baseline.initial_tempo_factor == 1) & baseline.method.isin(["clock", "uniform_filter"])]
    controls = controls.groupby(["method", "protocol"], as_index=False)[METRICS].mean()

    alternate = frames["06_finalists_alternate"].groupby(["method", "protocol"], as_index=False)[METRICS].mean()
    stress = frames["07_finalists_tempo_stress"].groupby(["method", "protocol", "initial_tempo_factor"], as_index=False)[METRICS].mean()
    seeds = []
    for seed, run in [(0, "05_continuous_tempo"), (1, "08_particles_seed1"), (2, "09_particles_seed2")]:
        values = frames[run][frames[run].method == "p02_particles_slow"].groupby("protocol", as_index=False)[METRICS].mean()
        values["seed"] = seed
        seeds.append(values)
    seeds = pd.concat(seeds, ignore_index=True)
    selected = aggregate[aggregate.method.isin(["loo_global", "loo_protocol"])].copy()
    seed_sync = seeds[seeds.protocol == "sync"]

    tune_tables = []
    for folder, method, protocol in [("tune_analysis_final_beam", "s06_beam_fine_tempo", "async"),
                                     ("tune_analysis_final_particles", "p02_particles_slow", "sync")]:
        provenance = json.loads((HERE / folder / "analysis_provenance.json").read_text())
        assert not provenance["warnings"]
        tunes = pd.read_csv(HERE / folder / "high_low_tunes.csv")
        tunes = tunes[(tunes.method == method) & (tunes.protocol == protocol)]
        tune_tables.append((protocol, method, tunes[["group", "title", "bar_accuracy", "within2beat_accuracy", "genre"]]))

    findings = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "scope": {"songs": 75, "async": 49, "sync": 26, "primary_settings": 42,
                          "song_method_evaluations": total},
                "fixed_primary": fixed.to_dict("records"),
                "internal_selection": selected.to_dict("records"),
                "alternate": alternate.to_dict("records"), "tempo_stress": stress.to_dict("records"),
                "particle_seeds": seeds.to_dict("records"), "ledger": ledger,
                "validation": "validation.json",
                "interpretation": "Exploratory corpus-informed development; not an external generalization test."}
    (HERE / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    renames = {"method": "Method", "protocol": "Protocol", "songs": "Songs",
               "bar_accuracy": "Time in correct bar", "within2beat_accuracy": "Within 2 beats",
               "mean_abs_beats": "Mean error (beats)", "median_abs_beats": "Mean of song medians (beats)",
               "initial_tempo_factor": "Initial tempo factor", "seed": "Seed"}
    for frame in (fixed, selected, controls, alternate):
        frame["method"] = frame.method.map(lambda m: LABELS.get(m, m))

    html = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIDI lead-sheet following: evaluation and findings</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#202c34;background:#fafbf9;max-width:1120px;margin:40px auto;padding:0 24px 70px}h1{font-size:32px;line-height:1.2}h2{margin-top:38px;font-size:23px}h3{font-size:18px}p,li{max-width:95ch}a{color:#175b70}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #d9e0e0}th{background:#edf2f2}td{font-variant-numeric:tabular-nums}code{font-size:.9em}.scroll{overflow-x:auto;margin:18px 0}.note{padding:15px 20px;border-left:4px solid #487a86;background:#edf4f4}pre{overflow:auto;background:#eef1ef;padding:18px;font-size:13px}.small{font-size:14px;color:#52616a}</style>
<h1>MIDI lead-sheet following: evaluation and findings</h1>
<p>Exploratory study, 6 October 2026. All 75 annotated preferred performances were evaluated: 49 async and 26 sync songs. Forty-two fixed settings were compared, followed by alternate-take, initial-tempo and random-seed checks.</p>
<p class="note"><strong>Main result:</strong> using bass/root evidence and changing the temporal tracker improves the original baseline. The fine-tempo beam reaches 91.41% exact-bar accuracy on async; the particle setting reaches 72.96% on sync at seed 0. These are different settings. The async catalog-tempo clock already reaches 99.87%, so the async result does not demonstrate a better follower than that control.</p>
<nav><a href="#metrics">Metric definitions</a> · <a href="#results">Results</a> · <a href="#robustness">Robustness</a> · <a href="#tunes">Tune properties</a> · <a href="#records">Full records</a></nav>

<h2 id="metrics">What is being measured?</h2>
<p>The tracker receives MIDI events in time order, an untimed unfolded chord plan, meter, the correct initial position/start time, and an approximate catalog tempo. Later reference timestamps are used only to score its outputs. Every 0.2 seconds, its estimated position is compared with the annotated performance.</p>
<ol>
<li><strong>Exact-bar accuracy:</strong> the percentage of evaluated 200-ms frames assigned to the correct occurrence of a bar in the unfolded plan. Bar 5 in chorus 2 is different from bar 5 in chorus 3. A 90% score means 90% of sampled time is in the correct bar; it does not mean 90% of bars are followed perfectly throughout.</li>
<li><strong>Within-two-beat rate:</strong> the percentage of those same frames with absolute position error at most two quarter-note beats. An estimate of beat 42.5 when the reference is 44 has error 1.5 and passes; an estimate of 41 has error 3 and fails. This is a tolerance-based position measure, not beat-onset detection F-measure.</li>
<li><strong>Mean absolute position error:</strong> average distance between estimated and reference positions, in quarter-note beats. Every frame contributes, including catastrophic errors. Smaller is better.</li>
<li><strong>Median absolute position error:</strong> the middle frame error within one performance. This describes a typical frame but can hide large failures; it is always reported with the mean. Summary tables average each song's median, rather than computing one pooled median.</li>
</ol>
<p>For a simple error sequence of 0, 1, 2 and 9 beats, the within-two-beat rate is 75%, mean error is 3 beats, and median error is 1.5 beats. Exact-bar accuracy requires the actual bar labels and cannot be recovered from these four error magnitudes alone. Near a boundary, a small error can cross into the wrong bar; within a long bar, a larger error can still remain in the correct bar.</p>
<p><strong>Reference limitation:</strong> the dataset supplies bar boundaries. Fractional beat position is linearly interpolated between those boundaries; individual beat onsets are not independently annotated. Exact-bar accuracy uses the bar labels directly. The beat-error measures therefore use an approximate reference for timing within a bar.</p>
<p><strong>Aggregation:</strong> calculate each metric per song, then average songs equally within async and sync separately. Long songs do not dominate the corpus summary. Silence remains part of the position evaluation; the tracker must keep time when no useful pitch evidence arrives.</p>
<h3>Chord recognition is a separate evaluation</h3>
<p>The original benchmark also checks chord labels every 0.1 seconds. The primary label is root plus base quality: major, minor, diminished, augmented or the annotated suspension. It ignores sevenths, extensions, alteration flags and slash bass. For example, C7 and Cmaj7 both reduce to C major, while Cm7 reduces to C minor. A separate seventh-level analysis retains seventh type.</p>
<p>Candidate chord labels are restricted to those in the supplied chart. This measures chart-conditioned harmonic recognition, not unrestricted chord transcription. The saved chord metrics include exact root/quality accuracy, root-only accuracy, balanced accuracy (equal weight per reference chord label), coverage, tie rate, and fractional tie credit. Missing evidence counts as incorrect on valid chord frames; annotated no-chord regions and annotation gaps are excluded from chord accuracy. Ties use a fixed ordering after any algorithm-specific tie rule; fractional accuracy awards 1/k when the correct chord is among k tied predictions.</p>
<p>The new improvement study optimizes and evaluates position following. It does not claim that its 91.41% or 72.96% scores are chord-label accuracy. Batch runtime is recorded for reproducibility; it is not measured live MIDI latency.</p>

<h2 id="results">Which changes helped?</h2>
<p>The original common grid integrates tempo predictions with pooled harmonic scores. The stronger harmonic representation uses the identified bass track and assigns template weight 2 to the chord root, 1 to other base chord tones and 0 to other pitches. The beam retains several likely position/tempo paths; the particle filter retains continuous position/tempo samples and updates their weights from harmonic compatibility. Both are online, non-neural prototypes.</p>''']
    html += [table(fixed, renames), "<h3>Controls</h3>", table(controls, renames),
             '''<p>On async, the fine-tempo beam improves all 49 songs relative to the original pooled-harmony grid; 37/49 exceed 90% bar accuracy. On sync, the seed-0 particle configuration improves 24/26 songs over that grid and 23/26 over the bass/root-weighted grid. The tested onset-phase heuristic and simple confidence gate did not produce the leading results.</p>
<p>The control without harmony uses the original grid's dynamics. The final beam and particle settings do not yet have matched full-corpus no-harmony ablations, so their gains do not isolate the causal contribution of harmonic evidence from the choice of temporal model.</p>
<h3>Internal selection check</h3>
<p>For each target song, the setting can be selected using the other songs' mean scores. Selecting separately by known protocol chooses s06 for async and p02 for sync; selecting one setting across both protocols chooses s06. No per-song oracle selects the best result from that song's labels. However, later candidate designs were informed by this inspected corpus: this internal leave-one-song-out check is not an untouched external test or a trained adaptive selector.</p>''', table(selected, renames),
             '<h2 id="robustness">Alternate takes, tempo errors and random seeds</h2>',
             '<p>Alternate-take evaluation covers 49 async songs only. Matching annotations were unavailable for sync alternate takes. The original pooled-harmony baseline obtains 47.19% bar accuracy and 8.395 beats mean error on this same alternate cohort.</p>',
             table(alternate, renames),
             '<p>The beam retains 84.33% on alternate async takes. The particle result has catastrophic outliers: its bar accuracy is higher than the original baseline, but its mean error rises to 48.16 beats. Reporting bar accuracy alone would hide this problem.</p>',
             '<p>Initial-tempo stress changes only the supplied starting BPM by −10% or +10%; the MIDI and reference stay unchanged. It tests initialization error, not a new artificially time-warped performance.</p>',
             table(stress, renames),
             '<p>The async beam falls from 91.41% at nominal tempo to about 51–52% under the perturbed initial tempo. The sync particle setting stays around 72–73%. These results argue against calling the beam universally robust.</p>',
             '<h3>Particle repeatability</h3>', table(seeds[["protocol", "seed", *METRICS]], renames),
             f'<p>Across the three seeds, sync bar accuracy averages {100*seed_sync.bar_accuracy.mean():.2f}% and ranges from {100*seed_sync.bar_accuracy.min():.2f}% to {100*seed_sync.bar_accuracy.max():.2f}%. Mean beat error varies from {seed_sync.mean_abs_beats.min():.2f} to {seed_sync.mean_abs_beats.max():.2f}; stable bar averages do not imply identical failure severity.</p>',
             '''<h2 id="tunes">What do the more accurately followed tunes have in common?</h2>
<p>For the sync particle setting, agreement between the exported bass pitches and annotated chord roots is associated with better bar accuracy. The fraction of non-silent frames containing a bass root has Spearman correlation 0.593 with accuracy. The highest five songs have median root-onset share 41.1%, compared with 28.3% for the lowest five. Their base chord-tone shares are 67.4% and 56.8%, respectively.</p>
<p>These are descriptive, reference-dependent measurements made after tracking; the follower never receives them. They depend on transcription quality and annotation alignment. They do not establish causation, a genre rule, or a feature available to an online weighting model. Sparse bass can work well on async because the supplied fixed clock is already accurate; the same interpretation does not transfer to sync.</p>''']
    for protocol, method, tunes in tune_tables:
        html += [f'<h3>{escape(protocol)} · {escape(method)}</h3>', table(tunes, renames)]
    html += ['''<p>Several async songs tie at 100%; the five shown are a deterministic subset of tied outcomes. The full per-song tables preserve every song. Sync failures such as Samba de Orfeu persist despite a small catalog-tempo mismatch, so initial tempo alone does not explain the failures.</p>
<p>During sync solo-role sections, particle bar accuracy is 83.3% for trumpet, 79.8% for piano, 63.4% for bass and 52.3% for drums. The piano label describes the annotated solo role; no piano MIDI was supplied to the tracker. The role subsets contain different songs and inherit earlier tracking drift, so this is not a controlled comparison of roles.</p>
<h2>What the study establishes, and what remains open</h2>
<p>The experiments identify workable, better-performing baseline configurations and useful failure cases. They do not yet establish an adaptive evidence-weighting model, unrestricted repeat/jump handling, unknown-start alignment, or a real-time end-to-end accompaniment system. Yamaha- and Pardo-inspired components are adaptations within common experimental followers, not reproductions of the complete published systems.</p>
<p>The inputs are automatic transcriptions of bass and horns, with drum MIDI used only in designated timing variants. They are not clean piano performances. Bass track identity is supplied by metadata. The reference is an unfolded performance plan with a synchronized known start. Independent verification of the audio/MIDI/annotation time origin remains outstanding, so the empirical scores are provisional. Local mixture duration differences alone cannot establish an offset.</p>
<p>A useful next controlled study would verify this alignment, add matched no-harmony controls for the final temporal models, and freeze settings before testing new compositions. Only then should an adaptive weighting experiment test whether causal measures of ambiguity or recent consistency predict when harmony helps correct tempo-driven drift.</p>
<h2 id="records">Experiment ledger and reproducibility</h2>''', table(pd.DataFrame(ledger)),
             f'<p>Validation passed: {total:,} prediction traces recomputed against saved metrics; 75 exact baseline reproductions; 462 input files unchanged; 98 absent-drum negative controls; causal-prefix, silence, overflow, tempo and random-seed checks. These checks validate implementation consistency, not independent annotation alignment or live latency.</p>',
             '''<ul>
<li><a href="comparison/report.html">All 42 settings, aggregate results and internal selection</a></li>
<li><a href="tune_analysis_final_particles/analysis.md">Particle tune analysis</a> · <a href="tune_analysis_final_beam/analysis.md">Beam tune analysis</a></li>
<li><a href="findings.json">Machine-readable findings</a> · <a href="validation.json">Validation results</a> · <a href="completion_audit.json">Completion audit</a></li>
<li><a href="study_plan.json">Study plan</a> · <a href="wave03_rationale.json">Final-wave rationale</a> · <a href="environment.json">Runtime environment</a> · <a href="data_fingerprint.json">Input hashes</a></li>
<li><a href="secondary_validation.md">Independent secondary-results review</a></li>
</ul>
<p>Each run folder contains <code>run.json</code> (settings saved before execution), <code>events.jsonl</code>, <code>summary.json</code>, <code>metrics.csv</code>, prediction traces, and source snapshots matching the execution hashes. Early smoke runs are explicitly incomplete development checks and are excluded from the main evaluation. Original MIDI, annotations, proposal text and baseline outputs were preserved.</p>
<p>To run a new comparison from the workspace root, use the Python interpreter recorded in <code>environment.json</code> and a new output directory:</p>
<pre>python3 'Research Proposal/experiments/ensemble_benchmark/improvement_study/run_study.py' \
  --wave continuous_tempo \
  --methods s06_beam_fine_tempo,p02_particles_slow \
  --output /tmp/follower-finalists-new-run

# Optional alternate takes: --takes alternate
# Initial-tempo stress: --tempo-factors 0.9,1.1
# Particle replication: --methods p02_particles_slow --particle-seed 1</pre>
<p class="small">Metric implementation: <code>../run_benchmark.py:position_metrics</code>; annotation targets: <code>../common.py:load_performance</code>. The chord and position metrics are different outputs and should not be conflated.</p>
</html>''']
    (HERE / "report.html").write_text("\n".join(html))
    audit = {"objective": json.loads((HERE / "study_plan.json").read_text())["objective"],
             "status": "complete", "requirements": {
                 "all_song_method_comparison": "42 settings across 75 songs; comparison/ and runs/",
                 "causality_and_reproduction": "validation.json: passed; 75 exact baseline reproductions",
                 "selection_and_robustness": "comparison/selected_metrics.csv and runs/03,04,06,07,08,09",
                 "tune_characteristics": "tune_analysis_final_particles/ and tune_analysis_final_beam/",
                 "documentation": "report.html, findings.json, source snapshots, traces and run logs"},
             "verified_evaluations": total,
             "remaining_research_limitations": ["Independent audio/MIDI/annotation alignment",
                 "Untouched external compositions", "Live end-to-end latency", "Adaptive learned fusion",
                 "Matched no-harmony ablations for final beam and particle configurations"]}
    (HERE / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"report": str(HERE / "report.html"), "evaluations": total}))


if __name__ == "__main__":
    main()
