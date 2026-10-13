# Finalist secondary validation

Verified 5 completed runs and recomputed all 698 finalist trace metrics to within 1e-12. Source snapshots match their recorded hashes; all five runs share identical source and manifest hashes. No inference sources or parameters were changed.

All values below are equal-weight means across performances within each protocol. Accuracy is percent; mean error is quarter-note beats. Preferred cohorts contain 49 async and 26 sync performances. Alternate contains 49 async performances only. The primary particle rows use seed 0; the beam is deterministic. Full-precision values, failure counts, source hashes, and paired comparisons are in `secondary_validation.json`.

## Preferred and alternate takes

| Take | Protocol | Method | N | Correct bar (%) | Within 2 beats (%) | Mean error |
|---|---|---|---:|---:|---:|---:|
| Preferred | async | p02_particles_slow | 49 | 66.604630 | 72.368241 | 8.325157 |
| Preferred | sync | p02_particles_slow | 26 | 72.959788 | 79.380266 | 3.578365 |
| Preferred | async | s06_beam_fine_tempo | 49 | 91.407505 | 92.577576 | 0.529093 |
| Preferred | sync | s06_beam_fine_tempo | 26 | 63.161734 | 72.395766 | 13.678166 |
| Alternate | async | p02_particles_slow | 49 | 53.116158 | 57.423030 | 48.164232 |
| Alternate | async | s06_beam_fine_tempo | 49 | 84.333144 | 85.619044 | 5.284061 |

## Supplied-tempo stress on preferred takes

| Protocol | Method | BPM factor | Correct bar (%) | Within 2 beats (%) | Mean error |
|---|---|---:|---:|---:|---:|
| async | p02_particles_slow | 0.9 | 67.587087 | 73.208803 | 5.075331 |
| sync | p02_particles_slow | 0.9 | 72.015311 | 77.816115 | 6.281740 |
| async | p02_particles_slow | 1.1 | 68.003207 | 73.601678 | 16.373024 |
| sync | p02_particles_slow | 1.1 | 73.190576 | 79.242652 | 5.246991 |
| async | s06_beam_fine_tempo | 0.9 | 52.214447 | 58.975848 | 5.314590 |
| sync | s06_beam_fine_tempo | 0.9 | 52.410071 | 58.779531 | 36.136739 |
| async | s06_beam_fine_tempo | 1.1 | 51.220625 | 57.257357 | 6.895146 |
| sync | s06_beam_fine_tempo | 1.1 | 66.305684 | 75.346265 | 4.444236 |

The beam’s async bar accuracy falls from 91.407505% with nominal initialization to 52.214447% / 51.220625% at factors 0.9 / 1.1. Its nominal async success is not evidence of robust tempo recovery. Particle sync accuracy remains 72.015311% / 73.190576%, but mean errors rise to 6.281740 / 5.246991 beats versus 3.578365 at nominal seed 0.

## Particle seed sensitivity on preferred takes

| Protocol | Seed | Correct bar (%) | Within 2 beats (%) | Mean error |
|---|---:|---:|---:|---:|
| async | 0 | 66.604630 | 72.368241 | 8.325157 |
| sync | 0 | 72.959788 | 79.380266 | 3.578365 |
| async | 1 | 68.308003 | 73.760189 | 7.669427 |
| sync | 1 | 72.473091 | 78.637652 | 5.858584 |
| async | 2 | 67.338060 | 72.930700 | 7.792459 |
| sync | 2 | 72.902805 | 79.018144 | 5.078832 |

| Protocol | Mean correct bar (%) | Seed range (%) | Mean within 2 beats (%) | Seed range (%) | Mean error | Seed range (beats) |
|---|---:|---|---:|---|---:|---|
| async | 67.416898 | 66.604630–68.308003 | 73.019710 | 72.368241–73.760189 | 7.929014 | 7.669427–8.325157 |
| sync | 72.778562 | 72.473091–72.959788 | 79.012020 | 78.637652–79.380266 | 4.838594 | 3.578365–5.858584 |

These are means/ranges of the three complete per-protocol macro results, not confidence intervals. Each run uses one fixed seed across all songs; no favorable seed was selected per song.

## Exact matched alternate-take comparison

| Method | Correct bar (%) | Within 2 beats (%) | Mean error |
|---|---:|---:|---:|
| yamaha_active | 47.188428 | 52.345755 | 8.394794 |
| clock | 99.871616 | 100.000000 | 0.000585 |
| uniform_filter | 91.155874 | 99.538114 | 0.350429 |

s06_beam_fine_tempo: +37.144716 percentage points versus original `yamaha_active`, with 45 improved / 4 worse / 0 tied songs; mean absolute error changes by -3.110733 beats.

p02_particles_slow: +5.927730 percentage points versus original `yamaha_active`, with 30 improved / 19 worse / 0 tied songs; mean absolute error changes by +39.769438 beats.

`yamaha_active` is the preserved experimental harmonic-grid baseline, not the full published Yamaha system. The matching used exact performance IDs, take, triad scheme, nominal tempo factor, and equal frame counts. The almost-perfect async clock remains the relevant known-tempo control.

## Major failure cases

- Preferred beam: Don’t Fence Me In 0.78% correct bar / 188.53 mean error beats; Amazing Grace 1.87% / 73.17; Samba de Orfeu 6.10% / 49.41. Its 0.7–1.3 tempo support cannot represent some performed/catalog tempo ratios; support mismatch does not explain every failure.
- Alternate beam: Nica’s Dream 2.07% / 108.72 beats and All of Me 18.85% / 69.61. Particle alternate Nica’s Dream 1.45% / 346.30 and All of Me 2.46% / 237.59 expose severe failures despite higher aggregate bar accuracy than the harmonic-grid baseline.
- Particle preferred failures persist across seeds: Samba de Orfeu 4.05–13.27% correct bar / 31.08–58.78 mean-error beats; Bye Bye Blackbird 12.42–20.20% / 9.36–40.61. Async Take the A Train has large seed-dependent errors. Exact per-song seed variation and each condition’s largest errors are retained in JSON.

## Limits

- Exploratory finalist selection after inspecting this corpus; no untouched external test or significance claim.
- Alternate takes cover the same 49 async compositions only, not independent compositions; no sync alternate evidence.
- All methods use a known unfolded chord plan and synchronized starting position/time; this does not test unknown starts, jumps, or unplanned repeats.
- Stress factors multiply supplied catalog BPM; recordings are not time-stretched. Neither finalist uses rhythm evidence. Beam support therefore changes to 0.63–1.17 or 0.77–1.43 of catalog BPM.
- Three particle seeds quantify limited Monte Carlo sensitivity; macro stability does not imply stable per-song failure severity.
- AMT MIDI timestamps versus original audio/annotation origin remain independently unverified; reference beats are interpolated within annotated bars.
- Results are causal MIDI simulations, not measured live audio/MIDI processing latency or calibrated reliability.
- Exact bar accuracy, within-two-beat accuracy, and absolute beat error measure different losses; large errors must not be hidden by one aggregate percentage.
