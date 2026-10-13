# Melody matching analysis

These are exploratory, fixed-setting comparisons of nine-onset matching adaptations. No per-song best system is reported. Protocols and section scopes remain separate; all accuracy summaries weight performances equally.

Methods are pitch-class cosine, pitch-class edit distance, signed-interval edit distance, interval/rhythm DTW, and interval/rhythm n-gram overlap. They are simplified algorithm-family adaptations, not complete published-system reproductions. The input has no certified lick annotations: this evaluates phrase/repetition matching, not labeled lick recognition.

## Highest-ranked fixed fusion by protocol and scope

Rankings include only fused variants. The table reports the highest macro bar-accuracy setting within each declared condition, selected retrospectively across that cohort. Full rankings retain both fixed strengths and all methods in `fusion_rankings.csv`. This is exploratory ranking, not held-out model selection.

| Reference | Protocol | Scope | N | Fixed fusion | Bar % | Gain pp | Within 2 beats % | Mean error | Improved / worse |
|---|---|---|---|---|---|---|---|---|---|
| past | async | all | 49 | pc_edit__fused1 | 68.487 | +1.882 | 74.382 | 7.417 | 27 / 21 |
| past | async | head_in | 49 | interval_ngrams__fused1 | 61.605 | -0.009 | 66.876 | 3.231 | 3 / 3 |
| past | async | head_out | 49 | interval_edit__fused3 | 71.243 | +9.712 | 79.941 | 13.511 | 27 / 19 |
| past | async | solo | 49 | pc_edit__fused1 | 70.142 | +1.482 | 76.085 | 4.593 | 22 / 21 |
| past | sync | all | 26 | interval_edit__fused3 | 73.571 | +0.611 | 79.750 | 3.594 | 15 / 10 |
| past | sync | head_in | 26 | interval_ngrams__fused1 | 84.529 | +0.033 | 91.885 | 0.902 | 1 / 0 |
| past | sync | head_out | 26 | interval_edit__fused3 | 69.421 | +1.514 | 74.930 | 7.269 | 11 / 12 |
| past | sync | solo | 26 | interval_edit__fused1 | 72.033 | +0.525 | 77.921 | 3.429 | 11 / 8 |
| written | async | all | 4 | pc_edit__fused3 | 76.135 | +14.337 | 81.007 | 1.279 | 4 / 0 |
| written | async | head_in | 4 | interval_dtw__fused3 | 92.175 | +18.746 | 96.131 | 0.647 | 4 / 0 |
| written | async | head_out | 4 | interval_dtw__fused3 | 87.703 | +33.863 | 94.864 | 0.684 | 4 / 0 |
| written | async | solo | 4 | pc_edit__fused3 | 71.681 | +9.729 | 76.353 | 1.412 | 4 / 0 |
| written | sync | all | 2 | pc_cosine__fused3 | 84.912 | +0.970 | 92.519 | 0.655 | 2 / 0 |
| written | sync | head_in | 2 | pc_cosine__fused3 | 89.048 | +3.702 | 96.945 | 0.476 | 2 / 0 |
| written | sync | head_out | 2 | interval_edit__fused3 | 93.693 | +0.385 | 100.000 | 0.315 | 1 / 0 |
| written | sync | solo | 2 | pc_cosine__fused1 | 82.182 | +0.788 | 89.987 | 0.762 | 2 / 0 |

## Harmonic controls and melody-only observation ablations

The ablation removes harmonic updates from the evaluated particle filter. Past-memory position tags and all planned-role masks still come from the independent harmonic anchor. It is not a standalone melody-only cold-start system. The complete scope tables are in `aggregate_metrics.csv`.

| Reference | Protocol | Method | N | Bar % | Within 2 beats % | Mean error |
|---|---|---|---|---|---|---|
| past | async | harmonic_only | 49 | 66.605 | 72.368 | 8.325 |
| past | async | interval_dtw__melodic_updates | 49 | 14.209 | 15.282 | 130.199 |
| past | async | interval_edit__melodic_updates | 49 | 13.751 | 14.964 | 126.686 |
| past | async | interval_ngrams__melodic_updates | 49 | 4.767 | 5.076 | 136.823 |
| past | async | no_observations | 49 | 3.671 | 3.862 | 138.309 |
| past | async | pc_cosine__melodic_updates | 49 | 14.805 | 15.883 | 135.748 |
| past | async | pc_edit__melodic_updates | 49 | 16.854 | 18.285 | 120.720 |
| past | sync | harmonic_only | 26 | 72.960 | 79.380 | 3.578 |
| past | sync | interval_dtw__melodic_updates | 26 | 11.260 | 12.156 | 118.788 |
| past | sync | interval_edit__melodic_updates | 26 | 12.176 | 12.816 | 105.354 |
| past | sync | interval_ngrams__melodic_updates | 26 | 5.243 | 5.395 | 114.447 |
| past | sync | no_observations | 26 | 4.246 | 4.360 | 120.025 |
| past | sync | pc_cosine__melodic_updates | 26 | 12.858 | 13.920 | 109.201 |
| past | sync | pc_edit__melodic_updates | 26 | 13.939 | 14.637 | 92.016 |
| written | async | harmonic_only | 4 | 61.798 | 65.627 | 2.219 |
| written | async | interval_dtw__melodic_updates | 4 | 62.032 | 62.738 | 25.008 |
| written | async | interval_edit__melodic_updates | 4 | 55.560 | 56.143 | 59.376 |
| written | async | interval_ngrams__melodic_updates | 4 | 20.094 | 20.672 | 80.884 |
| written | async | no_observations | 4 | 3.884 | 3.785 | 112.072 |
| written | async | pc_cosine__melodic_updates | 4 | 65.867 | 67.898 | 16.677 |
| written | async | pc_edit__melodic_updates | 4 | 56.625 | 57.365 | 47.542 |
| written | sync | harmonic_only | 2 | 83.943 | 91.716 | 0.716 |
| written | sync | interval_dtw__melodic_updates | 2 | 48.069 | 50.189 | 43.882 |
| written | sync | interval_edit__melodic_updates | 2 | 47.423 | 49.133 | 50.946 |
| written | sync | interval_ngrams__melodic_updates | 2 | 13.355 | 13.213 | 120.955 |
| written | sync | no_observations | 2 | 13.355 | 13.213 | 120.955 |
| written | sync | pc_cosine__melodic_updates | 2 | 40.910 | 43.477 | 5.035 |
| written | sync | pc_edit__melodic_updates | 2 | 41.854 | 43.925 | 41.815 |

## Written versus past memory: identical cohort

The written comparison contains 6 performances. Past-memory rows are restricted to exactly those IDs, with identical scope frame counts. Harmonic-only and no-observation control metrics agree to 1e-12. Full values and paired written-minus-past changes are in `matched_reference_metrics.csv` and `matched_reference_differences.csv`; the larger past-memory cohort is not its comparator.

| Protocol | Method | N | Written − past bar pp | Mean-error change | Improved / worse |
|---|---|---|---|---|---|
| async | interval_dtw__fused1 | 4 | +6.232 | -36.527 | 2 / 2 |
| async | interval_dtw__fused3 | 4 | +5.423 | -0.398 | 4 / 0 |
| async | interval_edit__fused1 | 4 | +6.957 | -31.054 | 3 / 1 |
| async | interval_edit__fused3 | 4 | +3.836 | -0.106 | 4 / 0 |
| async | interval_ngrams__fused1 | 4 | -0.520 | -0.010 | 1 / 1 |
| async | interval_ngrams__fused3 | 4 | -0.442 | -0.008 | 2 / 1 |
| async | pc_cosine__fused1 | 4 | +5.145 | -0.238 | 3 / 0 |
| async | pc_cosine__fused3 | 4 | +8.980 | -15.522 | 4 / 0 |
| async | pc_edit__fused1 | 4 | +7.148 | -26.955 | 2 / 2 |
| async | pc_edit__fused3 | 4 | +9.716 | -8.133 | 4 / 0 |
| sync | interval_dtw__fused1 | 2 | -0.358 | +0.010 | 1 / 1 |
| sync | interval_dtw__fused3 | 2 | +0.770 | -0.012 | 1 / 1 |
| sync | interval_edit__fused1 | 2 | -0.507 | +0.036 | 1 / 1 |
| sync | interval_edit__fused3 | 2 | +0.566 | -0.010 | 1 / 1 |
| sync | interval_ngrams__fused1 | 2 | +0.000 | -0.001 | 0 / 0 |
| sync | interval_ngrams__fused3 | 2 | +0.092 | -0.004 | 1 / 0 |
| sync | pc_cosine__fused1 | 2 | +0.005 | +0.007 | 1 / 1 |
| sync | pc_cosine__fused3 | 2 | +2.609 | -0.120 | 2 / 0 |
| sync | pc_edit__fused1 | 2 | +0.738 | -0.042 | 2 / 0 |
| sync | pc_edit__fused3 | 2 | +0.417 | -0.011 | 1 / 1 |

## Availability and negative findings

Coverage is the fraction of evaluation frames with a retained, fresh reference match that produces a potential. It is not correct-match accuracy, cue reliability, or the fraction of particles receiving material evidence. The same matcher has identical coverage across its ablation and two fusion strengths.

| Reference | Protocol | Matcher | N | Macro coverage % | No-evidence performances |
|---|---|---|---|---|---|
| past | async | interval_dtw | 49 | 17.135 | 0 |
| past | async | interval_edit | 49 | 5.417 | 0 |
| past | async | interval_ngrams | 49 | 0.912 | 24 |
| past | async | pc_cosine | 49 | 16.509 | 0 |
| past | async | pc_edit | 49 | 10.635 | 0 |
| past | sync | interval_dtw | 26 | 21.888 | 0 |
| past | sync | interval_edit | 26 | 5.976 | 0 |
| past | sync | interval_ngrams | 26 | 0.717 | 17 |
| past | sync | pc_cosine | 26 | 21.046 | 0 |
| past | sync | pc_edit | 26 | 11.754 | 0 |
| written | async | interval_dtw | 4 | 16.974 | 0 |
| written | async | interval_edit | 4 | 6.573 | 0 |
| written | async | interval_ngrams | 4 | 1.633 | 2 |
| written | async | pc_cosine | 4 | 16.755 | 0 |
| written | async | pc_edit | 4 | 9.838 | 0 |
| written | sync | interval_dtw | 2 | 13.062 | 0 |
| written | sync | interval_edit | 2 | 3.718 | 0 |
| written | sync | interval_ngrams | 2 | 0.000 | 2 |
| written | sync | pc_cosine | 2 | 11.247 | 0 |
| written | sync | pc_edit | 2 | 7.033 | 0 |

- past, sync: Highest bar-accuracy fusion increases mean absolute beat error (interval_edit__fused3).

All per-song paired changes are retained in `per_song_gains.csv`. `analysis.json` includes the three largest gains, losses, and absolute errors for every fixed all-frame fusion. Improvements in macro bar accuracy can coexist with individual catastrophic failures; those examples are not combined into an oracle system.

## Interpretation limits

- The corpus and algorithms were inspected during exploratory development. These are not untouched external-test results; no statistical-significance claim is made.
- The written cohort is limited by available usable melody files. Its small, selected composition set cannot establish general superiority over past memory.
- There are no certified lick labels. Repetition matching can match heads, accompaniment fragments, transpositions, or accidental short similarities; it does not demonstrate recognition of named jazz licks.
- All matchers use nine-note contexts of variable elapsed duration, with fixed nominal-beat span/gap limits. Timing normalization and feature invariances differ between algorithms.
- The similarity threshold and positive potential are heuristic and uncalibrated across algorithms. Median contrast begins at eight eligible windows, which may overlap and need not represent independent locations. Report coverage alongside accuracy.
- The positive reward leaves uncovered locations neutral. Repeated written measure IDs expand a hit to every matching occurrence, preserving rather than solving chorus ambiguity.
- Past phrases use historical position labels from an independent harmonic follower. Its errors can contaminate memory; removing harmonic updates from the evaluated filter does not remove this harmonic memory dependency.
- Known unfolded measure order, starting position/time, catalog BPM, lead-instrument flags, and planned solo roles are supplied. Solo masks use estimated positions; timed section annotations are used only to evaluate scopes.
- Head/solo scopes may omit performances with no frames in that section; each table gives its actual cohort size. Scope results are diagnostic subdivisions, not separately tuned systems.
- Input is offline AMT-derived MIDI. Audio/MIDI/annotation time-origin alignment remains independently unverified; beat targets are interpolated inside annotated bars.
- Matching and inference timings are batch measurements with shared preprocessing and precomputed controls. They are not live audio-to-decision latency comparisons.
- These primary tables summarize one fixed particle seed. Small gains may be sensitive to resampling; the separate seed_analysis outputs report all repeated conditions with same-seed harmonic controls.

## Reproduction

Run `analysis_results.py` only after both run summaries report completion. The script rejects missing methods, duplicate rows, mismatched frame counts, changed source snapshots, or unequal reference-independent controls on the written cohort. Analysis outputs never alter inference settings, source snapshots, or run metrics.
