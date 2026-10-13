# All-song MIDI harmony and following benchmark

This experiment compares harmonic matching methods on the locally available JazzSAMBA transcriptions. It evaluates chord estimates separately from the positions produced by a shared causal follower. It does **not** reproduce the complete Yamaha or Pardo accompaniment systems.

## Data and task

- 75 local songs: 49 async and 26 sync.
- 124 evaluated performances: preferred and alternate async takes, plus preferred sync takes. Distinct sync alternate timing annotations were unavailable, so those takes are inventoried but not scored.
- Preferred takes are the primary comparison. Alternate async takes are a secondary check on the same compositions.
- Harmony inputs contain available bass and horn MIDI. Local piano MIDI is absent; drums are excluded from harmonic evidence. These are automatic transcriptions, not clean MIDI keyboard performances.
- The original files remain unchanged. A separate four-second note-duration cap tests sensitivity to excessively long transcription notes.

Each chord scorer chooses among the reduced chord labels present in the supplied chart. The primary label is root plus base quality: major, minor, diminished, augmented, suspended, or power chord. Thus C, C7, Cmaj7, and C9 reduce to C major. A second evaluation retains seventh families while still ignoring higher extensions, alterations, and inversions. Explicit no-chord intervals are excluded from chord classification.

## Compared scorers

1. Cosine similarity against chord-tone templates, using recent note onsets.
2. The same similarity, using elapsed note durations.
3. Pardo's `P - (M + N)` matching rule and root tie-break, adapted to causal windows and the chart's reduced vocabulary.
4. Maezawa's harmonic dot-product likelihood, using an instantaneous binary 88-key piano roll and explicitly chosen binary chord templates. The original numerical templates, timing network, melody term, and full follower are not reproduced.
5. A statistical pitch-class profile learned from other compositions' preferred takes. All performances of the evaluated composition are excluded from its training counts. This is a simple learned harmonic baseline, not a neural model.

The window sweep uses 0.5, 1, 2, and 4 seconds. The fixed-context comparison uses one second, except the instantaneous Maezawa-style scorer. Best-window results are exploratory selections on the reported songs, not held-out performance estimates.

## Following experiment

All five methods supply harmonic scores to the same position-and-tempo grid filter. Additional baselines use a constant-tempo clock and the same filter without harmonic observations. The follower receives a known starting position/time, catalog tempo, and an unfolded, untimed chord/bar plan. Later annotation timestamps are used only for evaluation. No neural beat tracker or adaptive cue-weighting layer is evaluated here.

The filter updates every 200 ms, with a half-beat position grid and a declared tempo range. A separate initialization test multiplies the supplied tempo by 0.9 or 1.1. The position grid grows causally beyond the score endpoint; overrun predictions count as out of score rather than accumulating at the last bar. Unplanned repeats, jumps, unknown starts, and real-time execution latency are not evaluated.

## Metrics and interpretation

- **Chord accuracy:** percentage of labeled frames with the correct reduced root and quality. Missing evidence counts as an incorrect/abstaining prediction. Root-only, class-balanced, fractional-tie, and coverage scores are also saved.
- **Constant-chord baseline:** always predicts the chord occupying the most beats in the untimed chart. This checks whether a matcher improves on a simple chart prior.
- **Exact bar accuracy:** percentage of evaluation frames assigned to the correct unfolded bar, including the correct chorus occurrence. This adapts measure-number alignment accuracy to MIDI.
- **Within-two-beat rate:** percentage of frames within two score beats of the annotated position. Reference beat positions are interpolated within annotated bars, not independently annotated beat onsets.
- **Absolute position error:** mean and median error in score beats over all evaluated frames. Errors above two beats are retained; these are not Matchmaker's conditional error summaries.

Protocol tables average per-song scores equally. Async and sync are separate song collections, so their difference is not a controlled effect of recording protocol. Low chord accuracy need not imply equally low following accuracy: timing history constrains the candidate positions. Conversely, a strong result on a correctly initialized fixed-tempo recording may mostly reflect the clock.

## Files and reproduction

- `data/manifest.json`: exact eligible inputs and annotation paths.
- `data/sources.json`: downloaded annotation provenance and hashes.
- `audit/README.md`: all-file inventory, transcription issues, and timing metadata.
- `results/chord_metrics.csv`: per-performance chord results.
- `results/position_metrics.csv`: per-performance following results.
- `results/config.json`, `summary.json`, and `code_hashes.json`: settings, completion status, and implementation hashes.
- `results/report.html`: readable tables generated from the saved metrics.

Python requires NumPy; the report generator otherwise uses the standard library. From this directory:

```sh
python run_benchmark.py --skip-position
python run_positions_only.py --workers 3
python report_benchmark.py
```

The broader-tempo sensitivity can be run separately with `python run_positions_only.py --broad-tempo --workers 3`. It writes separate results and does not replace the primary following experiment.
