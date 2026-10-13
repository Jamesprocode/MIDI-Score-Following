# Tune characteristics and bar-tracking outcomes

This is a descriptive analysis of the 75 preferred performances, with async and sync kept separate. Chord recognition and position following are different tasks: a stronger chord classifier need not give a better tracking trajectory. The fixed primary view uses triad targets, nominal initial tempo, and the saved baseline settings. No per-song oracle chooses a winning method. Additional runs remain separately identified.

Reference-derived tempo and harmonic clarity, note-duration summaries, complete-song structure, and accuracy fields are explanatory measurements. They are **not causal inference inputs** and must not be supplied to an online selector as if available in advance.

## Focused configuration: p02_particles_slow

Recorded source: `run3_05_continuous_tempo`. Engine: `particle`. Harmonic input: `active_bass_root`. Constrain tempo diffusion while keeping broad initial uncertainty.

This configuration receives the dataset's **bass instrument label** and uses that stream for harmonic evidence. It is not role-free inference. Reference harmonic clarity is measured afterward; the tracker does not receive those diagnostic values.

## Baseline distributions

### async

| Method | Songs | Mean exact bar | Median | Middle 50% | ≥80% songs | ≥90% songs | Within 2 beats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clock | 49 | 99.9% | 100.0% | 100.0%–100.0% | 49 | 49 | 100.0% |
| cosine_duration | 49 | 43.6% | 45.4% | 30.9%–57.8% | 0 | 0 | 48.7% |
| cosine_onsets | 49 | 50.9% | 50.3% | 38.7%–63.5% | 1 | 0 | 56.9% |
| empirical_profile | 49 | 52.3% | 52.3% | 42.6%–65.0% | 1 | 0 | 57.7% |
| pardo | 49 | 43.1% | 42.9% | 32.2%–58.4% | 0 | 0 | 47.2% |
| uniform_filter | 49 | 91.2% | 93.3% | 86.1%–95.5% | 48 | 29 | 99.5% |
| yamaha_active | 49 | 51.9% | 51.6% | 37.2%–68.6% | 4 | 0 | 57.7% |

### sync

| Method | Songs | Mean exact bar | Median | Middle 50% | ≥80% songs | ≥90% songs | Within 2 beats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clock | 26 | 7.7% | 3.3% | 1.9%–6.2% | 0 | 0 | 8.5% |
| cosine_duration | 26 | 44.7% | 50.6% | 26.4%–58.2% | 0 | 0 | 47.7% |
| cosine_onsets | 26 | 53.0% | 57.4% | 41.4%–68.5% | 1 | 0 | 58.3% |
| empirical_profile | 26 | 51.2% | 53.4% | 35.6%–68.7% | 1 | 0 | 55.5% |
| pardo | 26 | 43.9% | 50.3% | 25.6%–59.6% | 0 | 0 | 46.6% |
| uniform_filter | 26 | 7.5% | 3.0% | 1.8%–7.0% | 0 | 0 | 8.0% |
| yamaha_active | 26 | 54.1% | 55.7% | 42.3%–72.0% | 3 | 0 | 60.4% |

Async uses a fixed timing plan and a synchronized known start, so near-perfect clock tracking is expected. Clock performance therefore does not establish robust score following. The uniform-evidence filter controls for the shared filter's own dynamics.

### Scale of reference timing variation

| Protocol | Median BPM CV | BPM CV range | Median catalog mismatch | Maximum catalog mismatch |
| --- | --- | --- | --- | --- |
| async | 0.000 | 0.000000–0.000391 | 0.0% | 0.0% |
| sync | 0.028 | 0.014385–0.130647 | 10.0% | 45.5% |

The tiny async variation is consistent with timestamp rounding on a fixed grid and must not be interpreted as measured expressive tempo drift. Rank correlations can look large even when a feature's physical range is negligible; inspect these ranges and `feature_ranges.csv`.

## Additional run: run1_01_diagnostics

These rows are paired with baseline outcomes for the same songs. If the additional run covers a subset, its percentages are not directly comparable to the full-cohort baseline mean; paired deltas use only its evaluated songs.

### async

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g00_reproduce | 49 | 51.9% | 51.6% | 4 | -48.0% | -39.3% | -0.4% |
| g01_bar_decision | 49 | 52.1% | 51.7% | 3 | -47.7% | -39.0% | -0.2% |
| g02_weak_harmony | 49 | 60.5% | 61.1% | 8 | -39.4% | -30.7% | 8.2% |
| g03_weak_bar | 49 | 59.0% | 59.3% | 8 | -40.9% | -32.1% | 6.7% |
| g04_bass_active | 49 | 56.9% | 58.0% | 5 | -43.0% | -34.3% | 4.6% |
| g05_balanced_active | 49 | 53.0% | 50.1% | 2 | -46.8% | -38.1% | 0.7% |
| g06_bass_root | 49 | 62.0% | 62.7% | 6 | -37.9% | -29.2% | 9.6% |
| g07_onsets | 49 | 51.6% | 51.3% | 1 | -48.3% | -39.6% | -0.7% |
| g08_onset_delay | 49 | 44.2% | 44.1% | 0 | -55.7% | -47.0% | -8.2% |
| g09_bass_onset_delay | 49 | 42.6% | 43.1% | 0 | -57.2% | -48.5% | -9.7% |
| g10_raw_contrast | 49 | 49.1% | 49.2% | 1 | -50.8% | -42.1% | -3.3% |
| g11_confidence_gate | 49 | 57.7% | 57.3% | 7 | -42.1% | -33.4% | 5.4% |
| g12_broad_prior | 49 | 42.9% | 42.1% | 0 | -57.0% | -48.3% | -9.5% |
| g13_finer_grid | 49 | 54.3% | 52.7% | 5 | -45.6% | -36.8% | 2.0% |

### sync

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g00_reproduce | 26 | 54.1% | 55.7% | 3 | 46.4% | 46.7% | 2.9% |
| g01_bar_decision | 26 | 54.1% | 55.7% | 3 | 46.4% | 46.7% | 2.9% |
| g02_weak_harmony | 26 | 51.3% | 53.7% | 4 | 43.6% | 43.9% | 0.1% |
| g03_weak_bar | 26 | 49.2% | 50.1% | 3 | 41.5% | 41.7% | -2.0% |
| g04_bass_active | 26 | 61.0% | 65.0% | 5 | 53.3% | 53.6% | 9.9% |
| g05_balanced_active | 26 | 57.9% | 57.3% | 3 | 50.2% | 50.5% | 6.8% |
| g06_bass_root | 26 | 65.0% | 69.6% | 8 | 57.3% | 57.6% | 13.9% |
| g07_onsets | 26 | 52.7% | 58.9% | 2 | 45.0% | 45.2% | 1.5% |
| g08_onset_delay | 26 | 48.1% | 50.0% | 0 | 40.4% | 40.7% | -3.1% |
| g09_bass_onset_delay | 26 | 50.2% | 52.0% | 0 | 42.4% | 42.7% | -1.0% |
| g10_raw_contrast | 26 | 55.3% | 57.1% | 3 | 47.6% | 47.8% | 4.1% |
| g11_confidence_gate | 26 | 51.6% | 55.5% | 2 | 43.9% | 44.2% | 0.5% |
| g12_broad_prior | 26 | 54.0% | 55.8% | 2 | 46.3% | 46.6% | 2.9% |
| g13_finer_grid | 26 | 55.3% | 57.7% | 3 | 47.6% | 47.9% | 4.2% |

## Additional run: run2_02_rhythm_sequence

These rows are paired with baseline outcomes for the same songs. If the additional run covers a subset, its percentages are not directly comparable to the full-cohort baseline mean; paired deltas use only its evaluated songs.

### async

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r00_rhythm_only | 49 | 13.8% | 6.5% | 0 | -86.1% | -77.3% | -38.5% |
| r01_joint | 49 | 44.8% | 43.3% | 1 | -55.1% | -46.3% | -7.5% |
| r02_joint_weak_harmony | 49 | 42.3% | 37.0% | 3 | -57.5% | -48.8% | -10.0% |
| r03_joint_gate | 49 | 44.6% | 42.6% | 3 | -55.2% | -46.5% | -7.7% |
| r04_joint_bass | 49 | 44.7% | 42.4% | 3 | -55.2% | -46.5% | -7.6% |
| r05_joint_delayed | 49 | 37.5% | 36.4% | 0 | -62.4% | -53.7% | -14.8% |
| r06_joint_bass_root | 49 | 59.6% | 60.8% | 8 | -40.2% | -31.5% | 7.3% |
| r07_joint_bass_root_broad | 49 | 61.7% | 62.4% | 8 | -38.1% | -29.4% | 9.4% |
| r08_joint_drums | 49 | 59.6% | 60.8% | 8 | -40.2% | -31.5% | 7.3% |
| r09_joint_drums_broad | 49 | 61.7% | 62.4% | 8 | -38.1% | -29.4% | 9.4% |
| r10_joint_window_tempered | 49 | 62.1% | 62.5% | 7 | -37.7% | -29.0% | 9.8% |
| s00_beam | 49 | 77.4% | 85.8% | 29 | -22.5% | -13.8% | 25.1% |
| s01_beam_bass | 49 | 86.2% | 93.2% | 34 | -13.7% | -4.9% | 33.9% |
| s02_beam_onsets | 49 | 74.2% | 75.4% | 20 | -25.7% | -17.0% | 21.8% |
| s03_beam_bass_root | 49 | 87.9% | 93.7% | 37 | -12.0% | -3.2% | 35.6% |
| s04_beam_bass_root_broad | 49 | 84.7% | 88.2% | 35 | -15.2% | -6.5% | 32.4% |

### sync

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r00_rhythm_only | 26 | 7.7% | 2.6% | 0 | -0.0% | 0.2% | -43.5% |
| r01_joint | 26 | 54.4% | 60.3% | 1 | 46.6% | 46.9% | 3.2% |
| r02_joint_weak_harmony | 26 | 53.0% | 55.3% | 1 | 45.3% | 45.6% | 1.9% |
| r03_joint_gate | 26 | 53.3% | 58.4% | 1 | 45.5% | 45.8% | 2.1% |
| r04_joint_bass | 26 | 62.1% | 67.4% | 2 | 54.4% | 54.7% | 11.0% |
| r05_joint_delayed | 26 | 45.9% | 48.4% | 0 | 38.2% | 38.5% | -5.2% |
| r06_joint_bass_root | 26 | 64.9% | 70.4% | 8 | 57.2% | 57.5% | 13.7% |
| r07_joint_bass_root_broad | 26 | 66.7% | 69.7% | 5 | 59.0% | 59.3% | 15.6% |
| r08_joint_drums | 26 | 65.4% | 70.9% | 8 | 57.6% | 57.9% | 14.2% |
| r09_joint_drums_broad | 26 | 66.8% | 70.2% | 6 | 59.1% | 59.4% | 15.7% |
| r10_joint_window_tempered | 26 | 65.1% | 69.6% | 8 | 57.4% | 57.6% | 13.9% |
| s00_beam | 26 | 49.4% | 52.0% | 0 | 41.7% | 42.0% | -1.7% |
| s01_beam_bass | 26 | 51.5% | 58.1% | 0 | 43.7% | 44.0% | 0.3% |
| s02_beam_onsets | 26 | 49.3% | 53.8% | 0 | 41.6% | 41.8% | -1.9% |
| s03_beam_bass_root | 26 | 54.1% | 61.2% | 3 | 46.3% | 46.6% | 2.9% |
| s04_beam_bass_root_broad | 26 | 56.8% | 63.9% | 2 | 49.1% | 49.4% | 5.7% |

## Additional run: run3_05_continuous_tempo

These rows are paired with baseline outcomes for the same songs. If the additional run covers a subset, its percentages are not directly comparable to the full-cohort baseline mean; paired deltas use only its evaluated songs.

### async

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g15_root_broad | 49 | 63.5% | 64.1% | 7 | -36.4% | -27.7% | 11.2% |
| g16_root_fine_position | 49 | 63.4% | 62.2% | 8 | -36.4% | -27.7% | 11.1% |
| g17_root_broad_fine | 49 | 65.6% | 66.4% | 11 | -34.3% | -25.6% | 13.3% |
| p00_particles | 49 | 59.7% | 60.1% | 6 | -40.2% | -31.5% | 7.3% |
| p01_particles_broad | 49 | 55.4% | 55.2% | 3 | -44.5% | -35.7% | 3.1% |
| p02_particles_slow | 49 | 66.6% | 70.8% | 11 | -33.3% | -24.6% | 14.3% |
| p03_particles_dense | 49 | 58.2% | 64.4% | 6 | -41.7% | -33.0% | 5.8% |
| s05_beam_width | 49 | 87.3% | 93.3% | 36 | -12.6% | -3.9% | 35.0% |
| s06_beam_fine_tempo | 49 | 91.4% | 95.2% | 43 | -8.5% | 0.3% | 39.1% |
| s07_beam_fine_mobile | 49 | 82.9% | 84.0% | 29 | -17.0% | -8.2% | 30.6% |
| s08_beam_fine_broad | 49 | 81.4% | 82.3% | 27 | -18.4% | -9.7% | 29.1% |

### sync

| Method | Songs | Mean bar | Median | ≥80% | Δ vs clock | Δ vs uniform | Δ vs empirical baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g15_root_broad | 26 | 66.9% | 70.2% | 6 | 59.2% | 59.4% | 15.7% |
| g16_root_fine_position | 26 | 63.9% | 71.6% | 8 | 56.2% | 56.5% | 12.8% |
| g17_root_broad_fine | 26 | 67.7% | 70.6% | 7 | 60.0% | 60.2% | 16.5% |
| p00_particles | 26 | 70.1% | 75.3% | 12 | 62.4% | 62.7% | 19.0% |
| p01_particles_broad | 26 | 70.7% | 78.4% | 13 | 62.9% | 63.2% | 19.5% |
| p02_particles_slow | 26 | 73.0% | 79.7% | 13 | 65.2% | 65.5% | 21.8% |
| p03_particles_dense | 26 | 70.8% | 78.2% | 13 | 63.1% | 63.3% | 19.6% |
| s05_beam_width | 26 | 55.1% | 61.2% | 3 | 47.4% | 47.7% | 4.0% |
| s06_beam_fine_tempo | 26 | 63.2% | 72.3% | 10 | 55.4% | 55.7% | 12.0% |
| s07_beam_fine_mobile | 26 | 64.6% | 77.0% | 10 | 56.9% | 57.2% | 13.5% |
| s08_beam_fine_broad | 26 | 70.6% | 78.0% | 11 | 62.9% | 63.1% | 19.4% |

## Paired changes for p02_particles_slow

Every contrast uses the same song, protocol, target scheme, preferred take and nominal initial tempo. Reference sources are recorded explicitly; cross-run contrasts compare separately executed fixed configurations. Bar/within-two-beat differences are percentage points; negative beat-error differences are improvements. These methods were explored on this same cohort, so this is not a held-out estimate of a selected winner.

| Source | Protocol | Reference | Reference source | Songs | Bar change (pp) | Median change (pp) | Improved/worse/same | Mean absolute error change (beats) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run3_05_continuous_tempo | async | g00_reproduce | run1_01_diagnostics | 49 | +14.72 | +14.17 | 41/8/0 | -0.29 |
| run3_05_continuous_tempo | async | g06_bass_root | run1_01_diagnostics | 49 | +4.65 | +4.24 | 34/15/0 | +4.01 |
| run3_05_continuous_tempo | sync | g00_reproduce | run1_01_diagnostics | 26 | +18.85 | +16.56 | 24/2/0 | -12.02 |
| run3_05_continuous_tempo | sync | g06_bass_root | run1_01_diagnostics | 26 | +7.91 | +6.99 | 23/3/0 | -0.99 |

### run3_05_continuous_tempo · async · changes versus g00_reproduce

| Group | Tune | Reference bar | Current bar | Change (pp) |
| --- | --- | --- | --- | --- |
| largest_5_gains | I Could Write a Book | 16.3% | 78.9% | +62.69 |
| largest_5_gains | Peri's Scope | 13.0% | 67.4% | +54.36 |
| largest_5_gains | Desafinado | 31.4% | 75.0% | +43.53 |
| largest_5_gains | Ornithology | 51.9% | 88.9% | +37.01 |
| largest_5_gains | Mas Que Nada | 19.4% | 54.4% | +34.98 |
| smallest_5_changes | Take the "A" Train | 43.4% | 19.9% | -23.46 |
| smallest_5_changes | My Favorite Things | 51.3% | 41.2% | -10.13 |
| smallest_5_changes | Impressions | 27.1% | 17.3% | -9.77 |
| smallest_5_changes | Au Privave | 46.6% | 41.8% | -4.83 |
| smallest_5_changes | Song for My Father | 45.4% | 40.8% | -4.66 |

The following exploratory associations concern **change** in accuracy, which is distinct from associations with final accuracy. They are post hoc descriptions, not causes.

| Feature | Group | Songs | Spearman ρ with bar change |
| --- | --- | --- | --- |
| pooled_note_fraction_over_10s | MIDI content summary | 49 | 0.462 |
| horns_note_fraction_over_10s | MIDI content summary | 49 | 0.451 |
| triad_vocabulary | chart harmony | 49 | 0.423 |
| baseline_seventh_cosine_onsets_chord_accuracy | evaluation outcome: not inference input | 49 | -0.385 |
| triad_chord_entropy_bits | chart harmony | 49 | 0.383 |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | -0.353 |
| horns_maximum_polyphony | MIDI content summary | 49 | 0.352 |
| baseline_triad_cosine_onsets_chord_accuracy | evaluation outcome: not inference input | 49 | -0.347 |

### run3_05_continuous_tempo · sync · changes versus g00_reproduce

| Group | Tune | Reference bar | Current bar | Change (pp) |
| --- | --- | --- | --- | --- |
| largest_5_gains | Amazing Grace | 23.0% | 86.9% | +63.86 |
| largest_5_gains | Chega de Saudade (No More Blues) | 15.8% | 73.3% | +57.51 |
| largest_5_gains | Waltz for Debby | 39.8% | 84.5% | +44.71 |
| largest_5_gains | Nardis | 49.7% | 89.7% | +40.00 |
| largest_5_gains | Black Orpheus (Manhã de Carnaval) | 62.2% | 91.6% | +29.40 |
| smallest_5_changes | Bye Bye Blackbird | 55.3% | 20.2% | -35.14 |
| smallest_5_changes | Samba de Orfeu | 12.9% | 10.9% | -1.95 |
| smallest_5_changes | When the Saints Go Marching In | 77.3% | 78.6% | +1.25 |
| smallest_5_changes | On the Sunny Side of the Street | 69.1% | 70.4% | +1.30 |
| smallest_5_changes | Invitation | 51.9% | 55.7% | +3.79 |

The following exploratory associations concern **change** in accuracy, which is distinct from associations with final accuracy. They are post hoc descriptions, not causes.

| Feature | Group | Songs | Spearman ρ with bar change |
| --- | --- | --- | --- |
| pooled_p95_note_seconds | MIDI content summary | 26 | 0.538 |
| horns_note_fraction_over_10s | MIDI content summary | 26 | 0.527 |
| pooled_note_fraction_over_10s | MIDI content summary | 26 | 0.515 |
| horns_reference_root_active_fraction | reference harmonic clarity: evaluation-only | 26 | 0.507 |
| horns_maximum_polyphony | MIDI content summary | 26 | 0.471 |
| pooled_maximum_polyphony | MIDI content summary | 26 | 0.462 |
| quality_flagged_pitched_files | MIDI quality audit | 26 | 0.456 |
| pooled_active_coverage | MIDI content summary | 26 | 0.434 |

## Highest and lowest tunes for p02_particles_slow

These are the five highest and five lowest observed outcomes for one named method, not claims that a tune is intrinsically easy or difficult. Ties are broken by song ID. If many tunes reach 100%, the five shown are an arbitrary deterministic subset of that tied group; their feature medians do not represent every perfect tune. Full tables for every method and both target schemes are in `high_low_tunes.csv`.

### run3_05_continuous_tempo · async · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A Sunday Kind of Love | 98.4% | 100.0% | 224 | 0.000 | 0.0% | 99.1% | 0 |
| (I Love You) For Sentimental Reasons | 95.3% | 98.0% | 224 | 0.000 | 0.0% | 99.1% | 1 |
| Lullaby of Birdland | 92.2% | 100.0% | 227 | 0.000 | 0.0% | 99.1% | 0 |
| Take Five | 91.4% | 99.4% | 180 | 0.000 | 0.0% | 99.4% | 0 |
| Ornithology | 88.9% | 95.6% | 227 | 0.000 | 0.0% | 100.0% | 0 |

### run3_05_continuous_tempo · async · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Impressions | 17.3% | 15.0% | 224 | 0.000 | 0.0% | 100.0% | 0 |
| Take the "A" Train | 19.9% | 21.2% | 236 | 0.000 | 0.0% | 97.4% | 1 |
| Song for My Father | 40.8% | 44.9% | 172 | 0.000 | 0.0% | 100.0% | 0 |
| My Favorite Things | 41.2% | 51.1% | 521 | 0.000 | 0.0% | 99.6% | 0 |
| Au Privave | 41.8% | 50.2% | 169 | 0.000 | 0.0% | 100.0% | 0 |

### run3_05_continuous_tempo · async · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 330.181 | 314.667 |
| bars | 224.000 | 224.000 |
| catalog_bpm | 165.000 | 175.000 |
| reference_bar_bpm_cv | 0.000 | 0.000 |
| reference_catalog_bpm_abs_fraction | 0.000 | 0.000 |
| triad_vocabulary | 14.000 | 6.000 |
| triad_changes_per_minute | 58.367 | 34.131 |
| triad_unique_bar_fraction | 0.062 | 0.035 |
| triad_repeated_4bar_fraction | 0.991 | 1.000 |
| bass_onsets_per_second | 1.947 | 1.516 |
| bass_active_coverage | 0.781 | 0.119 |
| bass_median_note_seconds | 0.339 | 0.223 |
| horns_note_fraction_over_10s | 0.000 | 0.000 |
| quality_flagged_pitched_files | 0.000 | 0.000 |
| bass_reference_root_onset_fraction | 0.314 | 0.230 |
| bass_reference_base_tone_onset_fraction | 0.515 | 0.423 |
| bass_reference_root_active_fraction | 0.482 | 0.445 |
| horns_reference_root_onset_fraction | 0.186 | 0.184 |
| horns_reference_base_tone_onset_fraction | 0.486 | 0.473 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | ballad: 1; bebop: 1; swing: 3 | 4/4: 4; 5/4: 1 |
| lowest_5 | bebop: 2; latin: 1; swing: 1; waltz: 1 | 3/4: 1; 4/4: 4 |

### run3_05_continuous_tempo · sync · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Groovin' High | 94.0% | 100.0% | 223 | 0.029 | 13.3% | 100.0% | 0 |
| Ceora | 92.1% | 99.0% | 229 | 0.023 | 5.1% | 100.0% | 2 |
| Black Orpheus (Manhã de Carnaval) | 91.6% | 98.9% | 233 | 0.030 | 3.5% | 97.4% | 2 |
| Triste | 91.6% | 98.5% | 202 | 0.015 | 16.2% | 98.0% | 0 |
| Sway (Quien Sera) | 91.0% | 99.8% | 184 | 0.014 | 10.0% | 98.3% | 0 |

### run3_05_continuous_tempo · sync · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Samba de Orfeu | 10.9% | 13.6% | 356 | 0.032 | 4.7% | 99.7% | 0 |
| Bye Bye Blackbird | 20.2% | 20.4% | 199 | 0.029 | 8.9% | 95.9% | 0 |
| Don't Fence Me In | 41.3% | 44.5% | 232 | 0.131 | 37.0% | 95.6% | 0 |
| Bernie's Tune | 46.2% | 49.4% | 195 | 0.037 | 8.6% | 99.0% | 1 |
| La Vie En Rose (Take Me To Your Heart Again) | 52.7% | 55.8% | 124 | 0.022 | 4.4% | 98.3% | 0 |

### run3_05_continuous_tempo · sync · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 322.262 | 332.784 |
| bars | 223.000 | 199.000 |
| catalog_bpm | 160.000 | 130.000 |
| reference_bar_bpm_cv | 0.023 | 0.032 |
| reference_catalog_bpm_abs_fraction | 0.100 | 0.086 |
| triad_vocabulary | 11.000 | 10.000 |
| triad_changes_per_minute | 44.684 | 45.519 |
| triad_unique_bar_fraction | 0.060 | 0.055 |
| triad_repeated_4bar_fraction | 0.983 | 0.983 |
| bass_onsets_per_second | 2.172 | 1.856 |
| bass_active_coverage | 0.811 | 0.779 |
| bass_median_note_seconds | 0.340 | 0.323 |
| horns_note_fraction_over_10s | 0.000 | 0.000 |
| quality_flagged_pitched_files | 0.000 | 0.000 |
| bass_reference_root_onset_fraction | 0.411 | 0.283 |
| bass_reference_base_tone_onset_fraction | 0.674 | 0.568 |
| bass_reference_root_active_fraction | 0.504 | 0.397 |
| horns_reference_root_onset_fraction | 0.165 | 0.177 |
| horns_reference_base_tone_onset_fraction | 0.508 | 0.447 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | bebop: 1; bossa nova: 3; latin: 1 | 4/4: 5 |
| lowest_5 | ballad: 1; samba: 1; swing: 3 | 4/4: 5 |

## Reference harmonic clarity

These within-protocol Spearman correlations compare each clarity measure with final bar accuracy. Onset fractions count each note event equally; active-root fractions weight non-silent 200-ms frames and ask whether any active pitch matches the root. They answer different questions, and extra polyphony or persistent transcribed notes can inflate active-root presence. All use timed reference harmony and are evaluation-only. Their correlations describe this inspected sample, not effects established by intervention.

| Source | Protocol | Measure | Songs | Spearman ρ with bar accuracy |
| --- | --- | --- | --- | --- |
| run3_05_continuous_tempo | async | bass_reference_base_tone_onset_fraction | 49 | 0.072 |
| run3_05_continuous_tempo | async | bass_reference_root_active_fraction | 49 | 0.286 |
| run3_05_continuous_tempo | async | bass_reference_root_onset_fraction | 49 | 0.131 |
| run3_05_continuous_tempo | async | horns_reference_base_tone_onset_fraction | 49 | -0.008 |
| run3_05_continuous_tempo | async | horns_reference_root_active_fraction | 49 | -0.286 |
| run3_05_continuous_tempo | async | horns_reference_root_onset_fraction | 49 | -0.023 |
| run3_05_continuous_tempo | sync | bass_reference_base_tone_onset_fraction | 26 | 0.482 |
| run3_05_continuous_tempo | sync | bass_reference_root_active_fraction | 26 | 0.593 |
| run3_05_continuous_tempo | sync | bass_reference_root_onset_fraction | 26 | 0.412 |
| run3_05_continuous_tempo | sync | horns_reference_base_tone_onset_fraction | 26 | -0.055 |
| run3_05_continuous_tempo | sync | horns_reference_root_active_fraction | 26 | -0.193 |
| run3_05_continuous_tempo | sync | horns_reference_root_onset_fraction | 26 | -0.121 |

## Exploratory within-protocol rank associations

Spearman correlations use average ranks for ties. They are descriptive associations across songs, not causal effects or significance tests. The strongest absolute correlations below are selected from many features after inspecting this same sample; correlated features and outliers can dominate. Constant or insufficient-data features have no correlation value. In particular, reference/catalog mismatch and final clock mismatch are mathematically related to timing error.

### run3_05_continuous_tempo · async · p02_particles_slow

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| bass_active_coverage | MIDI content summary | 49 | 0.595 |
| bass_onset_coverage_1s | MIDI content summary | 49 | 0.573 |
| bass_reference_harmony_active_frames | reference harmonic clarity: evaluation-only | 49 | 0.569 |
| pooled_active_coverage | MIDI content summary | 49 | 0.526 |
| bass_note_count | MIDI content summary | 49 | 0.496 |
| bass_reference_harmony_onset_count | reference harmonic clarity: evaluation-only | 49 | 0.491 |
| bass_median_note_seconds | MIDI content summary | 49 | 0.483 |
| pooled_note_count | MIDI content summary | 49 | 0.478 |
| pooled_onset_coverage_1s | MIDI content summary | 49 | 0.449 |
| bass_onsets_per_second | MIDI content summary | 49 | 0.403 |

### run3_05_continuous_tempo · sync · p02_particles_slow

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| bass_reference_root_active_fraction | reference harmonic clarity: evaluation-only | 26 | 0.593 |
| bass_reference_base_tone_onset_fraction | reference harmonic clarity: evaluation-only | 26 | 0.482 |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 26 | 0.469 |
| baseline_seventh_pardo_chord_accuracy | evaluation outcome: not inference input | 26 | 0.458 |
| reference_bar_bpm_cv | reference timing: evaluation-only | 26 | -0.457 |
| bass_reference_root_onset_fraction | reference harmonic clarity: evaluation-only | 26 | 0.412 |
| bass_active_coverage | MIDI content summary | 26 | 0.377 |
| baseline_seventh_cosine_duration_chord_accuracy | evaluation outcome: not inference input | 26 | 0.298 |
| bass_p95_note_seconds | MIDI content summary | 26 | 0.270 |
| baseline_seventh_yamaha_active_chord_accuracy | evaluation outcome: not inference input | 26 | 0.267 |

## Structural sections and solo roles

Section metrics come from saved trajectories whose whole-song metrics were checked against their CSV rows. Repeated sections of the same role are pooled within each song before songs are averaged. Section errors inherit previous tracking drift; a late solo's accuracy is not a local causal test of that soloist's evidence. Piano and drums can be labeled as soloists even though their notes are absent from harmonic input. These are instrument-role annotations, not tests of individual performer effects. Clock/uniform rows without saved traces are omitted here, not treated as missing whole-song outcomes.

### run3_05_continuous_tempo · async · p02_particles_slow

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 49 | 61.6% | 66.9% | 8.9% | 96.1% | 41.1% |
| head-out | NA | 49 | 61.5% | 67.3% | 92.7% | 96.7% | 47.9% |
| intro | NA | 34 | 75.0% | 80.0% | 1.3% | 27.8% | 23.8% |
| outro | NA | 9 | 52.9% | 55.6% | 98.8% | 89.3% | 25.7% |
| solo | bass | 48 | 66.7% | 74.0% | 65.3% | 61.9% | 60.5% |
| solo | drums | 47 | 50.1% | 54.0% | 77.2% | 25.1% | 21.9% |
| solo | piano | 49 | 75.4% | 82.0% | 52.3% | 49.7% | 47.0% |
| solo | saxophone_alto | 25 | 78.9% | 85.4% | 38.6% | 92.0% | 41.3% |
| solo | saxophone_tenor | 24 | 78.4% | 85.4% | 37.1% | 87.0% | 45.3% |
| solo | trumpet | 49 | 72.7% | 77.7% | 23.0% | 93.5% | 42.1% |

### run3_05_continuous_tempo · sync · p02_particles_slow

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 26 | 84.5% | 91.9% | 9.5% | 98.4% | 88.7% |
| head-out | NA | 26 | 67.9% | 75.6% | 91.9% | 99.0% | 90.9% |
| intro | NA | 16 | 69.1% | 73.7% | 1.3% | 57.5% | 50.4% |
| outro | NA | 8 | 65.6% | 73.5% | 99.0% | 99.3% | 84.7% |
| solo | bass | 26 | 63.4% | 69.5% | 63.0% | 92.6% | 90.8% |
| solo | drums | 25 | 52.3% | 57.6% | 77.3% | 40.2% | 15.6% |
| solo | piano | 26 | 79.8% | 85.6% | 47.5% | 92.8% | 90.5% |
| solo | saxophone_alto | 16 | 87.2% | 94.6% | 28.8% | 98.6% | 86.2% |
| solo | trumpet | 23 | 83.3% | 88.9% | 29.0% | 99.4% | 91.3% |

## Confounders and limits

- Async and sync use different tunes, protocols, available instruments, and performance conditions. Their difference is not an isolated synchrony effect.
- Only 49 async and 26 sync preferred performances are available. All feature/outcome relationships here are exploratory. Genre and meter subgroups can be very small.
- Tempo summaries use annotation bar durations and catalog meter, not independent beat onsets. Pickups, endings, and bar-level interpolation affect them.
- The unfolded chart already includes the performed bar order. Repetition statistics describe ambiguity within that supplied plan, not unknown-form tracking.
- Transcription-derived bass/horn notes contain artifacts. Quality flags are diagnostic, not verified labels of erroneous notes. Flagged/unflagged tunes also differ musically.
- Chord-accuracy features are evaluation outcomes. Their association with position accuracy is descriptive and not proof that a classifier change causes tracking improvement.
- Harmonic-clarity features compare exported pitches with timed reference chords. They depend on transcription quality, annotation alignment and canonical chord reduction; non-chord tones and anticipations are not automatically errors. These quantities are evaluation-only and never online inputs. Onset fractions weight note events; active-root fractions condition on non-silent frames, with denominator counts retained.
- Independent MIDI/audio/annotation time-origin verification remains outstanding. Timing or transcription mismatch can affect both clarity features and evaluation outcomes.
- Baseline profiles exclude the evaluated composition, but choices among windows, methods and improvements on these same songs do not constitute a held-out generalization test.
- Whole-song densities and durations are unavailable at the beginning of a live performance. They are not an online gating feature specification.

## Artifacts and reproduction

`song_features.csv` has one row per preferred song, with baseline/additional outcomes attached. `feature_definitions.json` gives exact feature meanings and separates reference-derived quantities. `method_distributions.csv`, `high_low_tunes.csv`, `rank_associations.csv`, `high_low_feature_medians.csv`, `method_comparisons.csv`, `section_metrics.csv`, and `section_summary.csv` retain the inspectable numbers. `paired_method_changes.csv`, `paired_change_summary.csv`, `paired_change_associations.csv`, and `paired_change_extremes.csv` compare matched methods and preserve both source runs; `feature_ranges.csv` records physical feature ranges. `analysis_provenance.json` records paths, hashes, settings, and QA warnings.

Run `python analyze_tunes.py`; add `--metrics /absolute/path/to/run/metrics.csv` for each additional run. Use `--focus-method METHOD` to choose which fixed method's high/low tunes and associations appear in this narrative. Use repeated `--reference-method METHOD` to add paired comparisons; the default is g00_reproduce. Add `--cross-run-references` to use an unambiguous reference from another supplied run when absent from the current run. Additional run traces are discovered at `run_dir/traces/{protocol}/{song}/preferred/{method}.npz`.
