# Tune characteristics and bar-tracking outcomes

This is a descriptive analysis of the 75 preferred performances, with async and sync kept separate. Chord recognition and position following are different tasks: a stronger chord classifier need not give a better tracking trajectory. The fixed primary view uses triad targets, nominal initial tempo, and the saved baseline settings. No per-song oracle chooses a winning method. Additional runs remain separately identified.

Reference-derived tempo, note-duration summaries, complete-song structure, and accuracy fields are explanatory measurements. They are **not causal inference inputs** and must not be supplied to an online selector as if available in advance.

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

## Paired changes for g06_bass_root

Every contrast uses the same song, protocol, target scheme, and run. Bar/within-two-beat differences are percentage points; negative beat-error differences are improvements. These methods were explored on this same cohort, so this is not a held-out estimate of a selected winner.

| Source | Protocol | Reference | Songs | Bar change (pp) | Median change (pp) | Improved/worse/same | Mean absolute error change (beats) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run1_01_diagnostics | async | g00_reproduce | 49 | +10.07 | +7.51 | 38/11/0 | -4.30 |
| run1_01_diagnostics | async | g01_bar_decision | 49 | +9.83 | +7.11 | 38/10/1 | -4.66 |
| run1_01_diagnostics | async | g04_bass_active | 49 | +5.09 | +4.10 | 45/3/1 | -0.38 |
| run1_01_diagnostics | sync | g00_reproduce | 26 | +10.94 | +7.17 | 21/5/0 | -11.02 |
| run1_01_diagnostics | sync | g01_bar_decision | 26 | +10.93 | +7.74 | 21/5/0 | -11.12 |
| run1_01_diagnostics | sync | g04_bass_active | 26 | +4.01 | +1.85 | 20/6/0 | -3.39 |

This method receives the supplied **bass instrument label** and uses only that role's active pitches for harmony, with extra root weight and a bar-posterior decision rule. It is not role-free inference. Compared with g00_reproduce, three choices change together. The g01_bar_decision contrast holds the decision rule fixed; the g04_bass_active contrast additionally holds bass-only input fixed and isolates extra root weighting. No reference tempo or outcome features in this analysis enter the tracker.

### run1_01_diagnostics · async · changes versus g00_reproduce

| Group | Tune | Reference bar | Current bar | Change (pp) |
| --- | --- | --- | --- | --- |
| largest_5_gains | I Could Write a Book | 16.3% | 67.4% | +51.11 |
| largest_5_gains | Peri's Scope | 13.0% | 50.0% | +37.02 |
| largest_5_gains | Autumn Leaves | 26.6% | 58.2% | +31.59 |
| largest_5_gains | All the Things You Are | 44.8% | 75.8% | +30.96 |
| largest_5_gains | Lazy Bird | 37.2% | 64.2% | +27.05 |
| smallest_5_changes | The Girl From Ipanema (Garota de Ipanema) | 57.1% | 49.4% | -7.67 |
| smallest_5_changes | Like Someone in Love | 48.1% | 40.4% | -7.64 |
| smallest_5_changes | Blue Bossa | 79.4% | 74.0% | -5.47 |
| smallest_5_changes | A Night in Tunisia | 54.6% | 50.8% | -3.76 |
| smallest_5_changes | Wait Till You See Her | 67.2% | 64.6% | -2.63 |

The following exploratory associations concern **change** in accuracy, which is distinct from associations with final accuracy. They are post hoc descriptions, not causes.

| Feature | Group | Songs | Spearman ρ with bar change |
| --- | --- | --- | --- |
| bass_note_count | MIDI content summary | 49 | -0.588 |
| bass_onsets_per_second | MIDI content summary | 49 | -0.502 |
| bass_active_coverage | MIDI content summary | 49 | -0.466 |
| bass_onset_coverage_1s | MIDI content summary | 49 | -0.460 |
| pooled_onset_coverage_1s | MIDI content summary | 49 | -0.367 |
| baseline_triad_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | -0.357 |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | -0.352 |
| bass_p95_note_seconds | MIDI content summary | 49 | -0.341 |

### run1_01_diagnostics · sync · changes versus g00_reproduce

| Group | Tune | Reference bar | Current bar | Change (pp) |
| --- | --- | --- | --- | --- |
| largest_5_gains | Chega de Saudade (No More Blues) | 15.8% | 63.4% | +47.57 |
| largest_5_gains | Waltz for Debby | 39.8% | 80.9% | +41.04 |
| largest_5_gains | Nardis | 49.7% | 86.7% | +37.00 |
| largest_5_gains | Don't Fence Me In | 13.0% | 35.9% | +22.90 |
| largest_5_gains | In a Sentimental Mood | 50.2% | 69.4% | +19.21 |
| smallest_5_changes | Bye Bye Blackbird | 55.3% | 43.0% | -12.37 |
| smallest_5_changes | Samba de Orfeu | 12.9% | 10.6% | -2.24 |
| smallest_5_changes | It Don't Mean a Thing (If It Ain't Got That Swing) | 57.0% | 55.3% | -1.67 |
| smallest_5_changes | One Note Samba (Samba de Uma Nota Só) | 53.9% | 52.4% | -1.53 |
| smallest_5_changes | When the Saints Go Marching In | 77.3% | 76.1% | -1.25 |

The following exploratory associations concern **change** in accuracy, which is distinct from associations with final accuracy. They are post hoc descriptions, not causes.

| Feature | Group | Songs | Spearman ρ with bar change |
| --- | --- | --- | --- |
| pooled_maximum_polyphony | MIDI content summary | 26 | 0.511 |
| horns_maximum_polyphony | MIDI content summary | 26 | 0.507 |
| baseline_triad_yamaha_active_chord_accuracy | evaluation outcome: not inference input | 26 | -0.503 |
| quality_flagged_pitched_files | MIDI quality audit | 26 | 0.495 |
| pooled_p95_note_seconds | MIDI content summary | 26 | 0.458 |
| horns_note_fraction_over_10s | MIDI content summary | 26 | 0.449 |
| pooled_note_fraction_over_10s | MIDI content summary | 26 | 0.447 |
| horns_note_count | MIDI content summary | 26 | 0.423 |

## Highest and lowest tunes for g06_bass_root

These are the five highest and five lowest observed outcomes for one named method, not claims that a tune is intrinsically easy or difficult. Tied ranks use song-name order. Full tables for every method and both target schemes are in `high_low_tunes.csv`.

### run1_01_diagnostics · async · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Blue Skies | 90.3% | 93.8% | 232 | 0.000 | 0.0% | 100.0% | 0 |
| A Sunday Kind of Love | 89.3% | 94.4% | 224 | 0.000 | 0.0% | 99.1% | 0 |
| Lullaby of Birdland | 88.0% | 96.4% | 227 | 0.000 | 0.0% | 99.1% | 0 |
| Take Five | 86.0% | 90.2% | 180 | 0.000 | 0.0% | 99.4% | 0 |
| Tour De Force | 85.8% | 89.5% | 227 | 0.000 | 0.0% | 100.0% | 0 |

### run1_01_diagnostics · async · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Desafinado | 32.6% | 36.2% | 478 | 0.000 | 0.0% | 99.8% | 1 |
| All of Me | 33.4% | 35.1% | 236 | 0.000 | 0.0% | 98.7% | 1 |
| Mas Que Nada | 34.9% | 38.5% | 283 | 0.000 | 0.0% | 100.0% | 1 |
| Impressions | 36.3% | 37.3% | 224 | 0.000 | 0.0% | 100.0% | 0 |
| Like Someone in Love | 40.4% | 45.2% | 232 | 0.000 | 0.0% | 99.6% | 0 |

### run1_01_diagnostics · async · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 363.200 | 377.334 |
| bars | 227.000 | 236.000 |
| catalog_bpm | 150.000 | 170.000 |
| reference_bar_bpm_cv | 0.000 | 0.000 |
| reference_catalog_bpm_abs_fraction | 0.000 | 0.000 |
| triad_vocabulary | 9.000 | 11.000 |
| triad_changes_per_minute | 58.367 | 32.415 |
| triad_unique_bar_fraction | 0.048 | 0.040 |
| triad_repeated_4bar_fraction | 0.994 | 0.998 |
| bass_onsets_per_second | 1.444 | 1.600 |
| bass_active_coverage | 0.722 | 0.450 |
| bass_median_note_seconds | 0.332 | 0.223 |
| horns_note_fraction_over_10s | 0.000 | 0.020 |
| quality_flagged_pitched_files | 0.000 | 1.000 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | swing: 5 | 4/4: 4; 5/4: 1 |
| lowest_5 | bebop: 1; bossa nova: 1; samba: 1; swing: 2 | 4/4: 5 |

### run1_01_diagnostics · sync · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Groovin' High | 89.4% | 94.6% | 223 | 0.029 | 13.3% | 100.0% | 0 |
| Ceora | 89.3% | 97.8% | 229 | 0.023 | 5.1% | 100.0% | 2 |
| Nardis | 86.7% | 94.4% | 200 | 0.027 | 2.1% | 97.5% | 1 |
| Boplicity (Be Bop Lives) | 86.1% | 94.2% | 224 | 0.022 | 0.5% | 100.0% | 0 |
| Sway (Quien Sera) | 85.9% | 92.2% | 184 | 0.014 | 10.0% | 98.3% | 0 |

### run1_01_diagnostics · sync · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Samba de Orfeu | 10.6% | 9.0% | 356 | 0.032 | 4.7% | 99.7% | 0 |
| Bernie's Tune | 33.3% | 33.9% | 195 | 0.037 | 8.6% | 99.0% | 1 |
| Amazing Grace | 35.8% | 39.7% | 100 | 0.021 | 45.5% | 99.0% | 1 |
| Don't Fence Me In | 35.9% | 36.2% | 232 | 0.131 | 37.0% | 95.6% | 0 |
| Bye Bye Blackbird | 43.0% | 45.5% | 199 | 0.029 | 8.9% | 95.9% | 0 |

### run1_01_diagnostics · sync · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 326.857 | 332.784 |
| bars | 223.000 | 199.000 |
| catalog_bpm | 135.000 | 155.000 |
| reference_bar_bpm_cv | 0.023 | 0.032 |
| reference_catalog_bpm_abs_fraction | 0.051 | 0.089 |
| triad_vocabulary | 10.000 | 10.000 |
| triad_changes_per_minute | 44.684 | 45.519 |
| triad_unique_bar_fraction | 0.054 | 0.050 |
| triad_repeated_4bar_fraction | 1.000 | 0.990 |
| bass_onsets_per_second | 2.005 | 2.332 |
| bass_active_coverage | 0.799 | 0.783 |
| bass_median_note_seconds | 0.398 | 0.308 |
| horns_note_fraction_over_10s | 0.000 | 0.000 |
| quality_flagged_pitched_files | 0.000 | 0.000 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | bebop: 3; bossa nova: 1; latin: 1 | 4/4: 5 |
| lowest_5 | samba: 1; swing: 4 | 4/4: 5 |

## Exploratory within-protocol rank associations

Spearman correlations use average ranks for ties. They are descriptive associations across songs, not causal effects or significance tests. The strongest absolute correlations below are selected from many features after inspecting this same sample; correlated features and outliers can dominate. Constant or insufficient-data features have no correlation value. In particular, reference/catalog mismatch and final clock mismatch are mathematically related to timing error.

### run1_01_diagnostics · async · g06_bass_root

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| seventh_changes_per_minute | chart harmony plus reference duration | 49 | 0.416 |
| triad_changes_per_minute | chart harmony plus reference duration | 49 | 0.406 |
| horns_p95_note_seconds | MIDI content summary | 49 | -0.371 |
| bass_median_note_seconds | MIDI content summary | 49 | 0.356 |
| bass_active_coverage | MIDI content summary | 49 | 0.341 |
| bass_onset_coverage_1s | MIDI content summary | 49 | 0.329 |
| pooled_p95_note_seconds | MIDI content summary | 49 | -0.322 |
| baseline_triad_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | 0.294 |
| pooled_active_coverage | MIDI content summary | 49 | 0.283 |
| baseline_triad_pardo_chord_accuracy | evaluation outcome: not inference input | 49 | 0.281 |

### run1_01_diagnostics · sync · g06_bass_root

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 26 | 0.422 |
| baseline_seventh_pardo_chord_accuracy | evaluation outcome: not inference input | 26 | 0.383 |
| triad_largest_chord_beat_share | chart harmony | 26 | -0.338 |
| horns_median_note_seconds | MIDI content summary | 26 | -0.306 |
| bass_active_coverage | MIDI content summary | 26 | 0.294 |
| baseline_triad_yamaha_active_chord_accuracy | evaluation outcome: not inference input | 26 | -0.290 |
| reference_bar_bpm_cv | reference timing: evaluation-only | 26 | -0.290 |
| reference_catalog_bpm_abs_fraction | reference timing: evaluation-only | 26 | -0.285 |
| bass_median_note_seconds | MIDI content summary | 26 | 0.279 |
| bass_maximum_polyphony | MIDI content summary | 26 | 0.277 |

## Structural sections and solo roles

Section metrics come from saved trajectories whose whole-song metrics were checked against their CSV rows. Repeated sections of the same role are pooled within each song before songs are averaged. Section errors inherit previous tracking drift; a late solo's accuracy is not a local causal test of that soloist's evidence. Piano and drums can be labeled as soloists even though their notes are absent from harmonic input. Clock/uniform rows without saved traces are omitted here, not treated as missing whole-song outcomes.

### run1_01_diagnostics · async · g06_bass_root

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 49 | 76.9% | 82.7% | 8.9% | 96.1% | 41.1% |
| head-out | NA | 49 | 55.3% | 59.7% | 92.7% | 96.7% | 47.9% |
| intro | NA | 34 | 90.2% | 96.6% | 1.3% | 27.8% | 23.8% |
| outro | NA | 9 | 49.3% | 54.0% | 98.8% | 89.3% | 25.7% |
| solo | bass | 48 | 45.2% | 48.5% | 65.3% | 61.9% | 60.5% |
| solo | drums | 47 | 43.6% | 46.6% | 77.2% | 25.1% | 21.9% |
| solo | piano | 49 | 66.1% | 72.9% | 52.3% | 49.7% | 47.0% |
| solo | saxophone_alto | 25 | 73.5% | 80.6% | 38.6% | 92.0% | 41.3% |
| solo | saxophone_tenor | 24 | 63.6% | 70.0% | 37.1% | 87.0% | 45.3% |
| solo | trumpet | 49 | 74.9% | 80.9% | 23.0% | 93.5% | 42.1% |

### run1_01_diagnostics · sync · g06_bass_root

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 26 | 77.9% | 84.4% | 9.5% | 98.4% | 88.7% |
| head-out | NA | 26 | 72.1% | 78.0% | 91.9% | 99.0% | 90.9% |
| intro | NA | 16 | 71.2% | 76.5% | 1.3% | 57.5% | 50.4% |
| outro | NA | 8 | 69.5% | 81.7% | 99.0% | 99.3% | 84.7% |
| solo | bass | 26 | 45.1% | 48.5% | 63.0% | 92.6% | 90.8% |
| solo | drums | 25 | 38.6% | 42.0% | 77.3% | 40.2% | 15.6% |
| solo | piano | 26 | 71.8% | 77.3% | 47.5% | 92.8% | 90.5% |
| solo | saxophone_alto | 16 | 79.8% | 86.3% | 28.8% | 98.6% | 86.2% |
| solo | trumpet | 23 | 75.4% | 81.9% | 29.0% | 99.4% | 91.3% |

## Confounders and limits

- Async and sync use different tunes, protocols, available instruments, and performance conditions. Their difference is not an isolated synchrony effect.
- Only 49 async and 26 sync preferred performances are available. All feature/outcome relationships here are exploratory. Genre and meter subgroups can be very small.
- Tempo summaries use annotation bar durations and catalog meter, not independent beat onsets. Pickups, endings, and bar-level interpolation affect them.
- The unfolded chart already includes the performed bar order. Repetition statistics describe ambiguity within that supplied plan, not unknown-form tracking.
- Transcription-derived bass/horn notes contain artifacts. Quality flags are diagnostic, not verified labels of erroneous notes. Flagged/unflagged tunes also differ musically.
- Chord-accuracy features are evaluation outcomes. Their association with position accuracy is descriptive and not proof that a classifier change causes tracking improvement.
- Baseline profiles exclude the evaluated composition, but choices among windows, methods and improvements on these same songs do not constitute a held-out generalization test.
- Whole-song densities and durations are unavailable at the beginning of a live performance. They are not an online gating feature specification.

## Artifacts and reproduction

`song_features.csv` has one row per preferred song, with baseline/additional outcomes attached. `feature_definitions.json` gives exact feature meanings and separates reference-derived quantities. `method_distributions.csv`, `high_low_tunes.csv`, `rank_associations.csv`, `high_low_feature_medians.csv`, `method_comparisons.csv`, `section_metrics.csv`, and `section_summary.csv` retain the inspectable numbers. `paired_method_changes.csv`, `paired_change_summary.csv`, `paired_change_associations.csv`, and `paired_change_extremes.csv` compare same-run methods; `feature_ranges.csv` records physical feature ranges. `analysis_provenance.json` records paths, hashes, settings, and QA warnings.

Run `python analyze_tunes.py`; add `--metrics /absolute/path/to/run/metrics.csv` for each additional run. Use `--focus-method METHOD` to choose which fixed method's high/low tunes and associations appear in this narrative. Use repeated `--reference-method METHOD` to add same-run paired comparisons; the default is g00_reproduce. Additional run traces are discovered at `run_dir/traces/{protocol}/{song}/preferred/{method}.npz`.
