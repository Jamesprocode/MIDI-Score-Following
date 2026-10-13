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

## Highest and lowest tunes for empirical_profile

These are the five highest and five lowest observed outcomes for one named method, not claims that a tune is intrinsically easy or difficult. Tied ranks use song-name order. Full tables for every method and both target schemes are in `high_low_tunes.csv`.

### baseline · async · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tour De Force | 85.7% | 93.7% | 227 | 0.000 | 0.0% | 100.0% | 0 |
| A Sunday Kind of Love | 79.6% | 86.8% | 224 | 0.000 | 0.0% | 99.1% | 0 |
| Road Song | 77.9% | 81.0% | 232 | 0.000 | 0.0% | 98.7% | 1 |
| (I Love You) For Sentimental Reasons | 75.7% | 82.2% | 224 | 0.000 | 0.0% | 99.1% | 1 |
| St. Thomas | 71.8% | 77.7% | 240 | 0.000 | 0.0% | 100.0% | 0 |

### baseline · async · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Autumn Leaves | 19.5% | 22.3% | 224 | 0.000 | 0.0% | 100.0% | 0 |
| Spain | 19.6% | 18.6% | 360 | 0.000 | 0.0% | 94.7% | 1 |
| Peri's Scope | 20.0% | 19.5% | 167 | 0.000 | 0.0% | 99.4% | 1 |
| Caravan | 21.9% | 23.1% | 464 | 0.000 | 0.0% | 100.0% | 0 |
| My Favorite Things | 30.4% | 38.0% | 521 | 0.000 | 0.0% | 99.6% | 0 |

### baseline · async · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 384.000 | 320.000 |
| bars | 227.000 | 360.000 |
| catalog_bpm | 145.000 | 190.000 |
| reference_bar_bpm_cv | 0.000 | 0.000 |
| reference_catalog_bpm_abs_fraction | 0.000 | 0.000 |
| triad_vocabulary | 13.000 | 10.000 |
| triad_changes_per_minute | 49.219 | 42.226 |
| triad_unique_bar_fraction | 0.060 | 0.040 |
| triad_repeated_4bar_fraction | 0.991 | 0.996 |
| bass_onsets_per_second | 1.444 | 0.371 |
| bass_active_coverage | 0.775 | 0.114 |
| bass_median_note_seconds | 0.291 | 0.220 |
| horns_note_fraction_over_10s | 0.000 | 0.000 |
| quality_flagged_pitched_files | 0.000 | 0.000 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | ballad: 1; calypso: 1; latin: 1; swing: 2 | 4/4: 5 |
| lowest_5 | latin: 1; samba: 1; swing: 2; waltz: 1 | 3/4: 1; 4/4: 4 |

### baseline · sync · highest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ceora | 84.5% | 94.3% | 229 | 0.023 | 5.1% | 100.0% | 2 |
| Triste | 78.9% | 88.5% | 202 | 0.015 | 16.2% | 98.0% | 0 |
| Nardis | 76.1% | 83.9% | 200 | 0.027 | 2.1% | 97.5% | 1 |
| Boplicity (Be Bop Lives) | 75.7% | 84.8% | 224 | 0.022 | 0.5% | 100.0% | 0 |
| Sway (Quien Sera) | 73.9% | 82.1% | 184 | 0.014 | 10.0% | 98.3% | 0 |

### baseline · sync · lowest 5

| Tune | Exact bar | Within 2 beats | Bars | Reference BPM CV | Catalog mismatch | 4-bar repetition | Flagged files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Don't Fence Me In | 7.2% | 5.5% | 232 | 0.131 | 37.0% | 95.6% | 0 |
| Samba de Orfeu | 10.0% | 8.7% | 356 | 0.032 | 4.7% | 99.7% | 0 |
| Amazing Grace | 12.2% | 1.9% | 100 | 0.021 | 45.5% | 99.0% | 1 |
| Invitation | 21.9% | 13.6% | 336 | 0.076 | 17.0% | 99.7% | 1 |
| Chega de Saudade (No More Blues) | 29.2% | 26.8% | 467 | 0.027 | 30.0% | 98.3% | 2 |

### baseline · sync · characteristics of these two groups

Each column summarizes five tunes selected by this method's observed bar accuracy. These small, outcome-selected groups describe this sample; they do not define a reliable eligibility rule for future tunes.

| Feature (median) | Highest five | Lowest five |
| --- | --- | --- |
| duration_seconds | 360.938 | 332.784 |
| bars | 202.000 | 336.000 |
| catalog_bpm | 135.000 | 155.000 |
| reference_bar_bpm_cv | 0.022 | 0.032 |
| reference_catalog_bpm_abs_fraction | 0.051 | 0.300 |
| triad_vocabulary | 11.000 | 11.000 |
| triad_changes_per_minute | 44.684 | 44.961 |
| triad_unique_bar_fraction | 0.055 | 0.057 |
| triad_repeated_4bar_fraction | 0.983 | 0.990 |
| bass_onsets_per_second | 1.937 | 2.386 |
| bass_active_coverage | 0.799 | 0.780 |
| bass_median_note_seconds | 0.409 | 0.291 |
| horns_note_fraction_over_10s | 0.000 | 0.000 |
| quality_flagged_pitched_files | 0.000 | 1.000 |

| Group | Catalog genres (counts) | Meters (counts) |
| --- | --- | --- |
| highest_5 | bebop: 2; bossa nova: 2; latin: 1 | 4/4: 5 |
| lowest_5 | bossa nova: 1; samba: 1; swing: 3 | 4/4: 5 |

## Exploratory within-protocol rank associations

Spearman correlations use average ranks for ties. They are descriptive associations across songs, not causal effects or significance tests. The strongest absolute correlations below are selected from many features after inspecting this same sample; correlated features and outliers can dominate. Constant or insufficient-data features have no correlation value. In particular, reference/catalog mismatch and final clock mismatch are mathematically related to timing error.

### baseline · async · empirical_profile

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| bass_active_coverage | MIDI content summary | 49 | 0.603 |
| bass_onset_coverage_1s | MIDI content summary | 49 | 0.558 |
| bass_median_note_seconds | MIDI content summary | 49 | 0.532 |
| baseline_triad_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | 0.522 |
| baseline_triad_pardo_chord_accuracy | evaluation outcome: not inference input | 49 | 0.487 |
| pooled_onset_coverage_1s | MIDI content summary | 49 | 0.480 |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 49 | 0.463 |
| pooled_active_coverage | MIDI content summary | 49 | 0.455 |
| bass_note_count | MIDI content summary | 49 | 0.445 |
| baseline_triad_cosine_duration_chord_accuracy | evaluation outcome: not inference input | 49 | 0.434 |

### baseline · sync · empirical_profile

| Feature | Group | Songs | Spearman ρ |
| --- | --- | --- | --- |
| reference_catalog_bpm_signed_fraction | reference timing: evaluation-only | 26 | -0.507 |
| reference_bar_bpm_median | reference timing: evaluation-only | 26 | -0.492 |
| seventh_changes_per_minute | chart harmony plus reference duration | 26 | -0.458 |
| bass_median_note_seconds | MIDI content summary | 26 | 0.438 |
| baseline_seventh_empirical_profile_chord_accuracy | evaluation outcome: not inference input | 26 | 0.432 |
| bass_p95_note_seconds | MIDI content summary | 26 | 0.413 |
| baseline_seventh_pardo_chord_accuracy | evaluation outcome: not inference input | 26 | 0.409 |
| reference_bar_bpm_cv | reference timing: evaluation-only | 26 | -0.404 |
| reference_catalog_bpm_abs_fraction | reference timing: evaluation-only | 26 | -0.368 |
| triad_changes_per_minute | chart harmony plus reference duration | 26 | -0.352 |

## Structural sections and solo roles

Section metrics come from saved trajectories whose whole-song metrics were checked against their CSV rows. Repeated sections of the same role are pooled within each song before songs are averaged. Section errors inherit previous tracking drift; a late solo's accuracy is not a local causal test of that soloist's evidence. Piano and drums can be labeled as soloists even though their notes are absent from harmonic input. Clock/uniform rows without saved traces are omitted here, not treated as missing whole-song outcomes.

### baseline · async · empirical_profile

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 49 | 66.4% | 73.6% | 8.9% | 96.1% | 41.1% |
| head-out | NA | 49 | 49.2% | 54.7% | 92.7% | 96.7% | 47.9% |
| intro | NA | 34 | 86.0% | 93.1% | 1.3% | 27.8% | 23.8% |
| outro | NA | 9 | 45.9% | 50.6% | 98.8% | 89.3% | 25.7% |
| solo | bass | 48 | 31.9% | 32.9% | 65.3% | 61.9% | 60.5% |
| solo | drums | 47 | 37.1% | 41.6% | 77.2% | 25.1% | 21.9% |
| solo | piano | 49 | 58.6% | 63.3% | 52.3% | 49.7% | 47.0% |
| solo | saxophone_alto | 25 | 57.3% | 64.5% | 38.6% | 92.0% | 41.3% |
| solo | saxophone_tenor | 24 | 51.8% | 58.2% | 37.1% | 87.0% | 45.3% |
| solo | trumpet | 49 | 64.3% | 71.4% | 23.0% | 93.5% | 42.1% |

### baseline · sync · empirical_profile

| Section | Solo role | Songs | Bar accuracy | Within 2 beats | Mean song progress | Active pooled | Active bass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head-in | NA | 26 | 64.3% | 70.1% | 9.5% | 98.4% | 88.7% |
| head-out | NA | 26 | 55.3% | 61.7% | 91.9% | 99.0% | 90.9% |
| intro | NA | 16 | 70.6% | 76.3% | 1.3% | 57.5% | 50.4% |
| outro | NA | 8 | 71.3% | 78.9% | 99.0% | 99.3% | 84.7% |
| solo | bass | 26 | 36.8% | 39.6% | 63.0% | 92.6% | 90.8% |
| solo | drums | 25 | 30.9% | 33.5% | 77.3% | 40.2% | 15.6% |
| solo | piano | 26 | 55.0% | 57.6% | 47.5% | 92.8% | 90.5% |
| solo | saxophone_alto | 16 | 54.0% | 58.7% | 28.8% | 98.6% | 86.2% |
| solo | trumpet | 23 | 58.3% | 63.4% | 29.0% | 99.4% | 91.3% |

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
