# Melody comparison: seed sensitivity

All fixed methods are repeated with seeds 0, 1, and 2. Each gain uses the harmonic-only result for the same seed, performance, and evaluation scope. Means/ranges describe these three complete runs; they are not confidence intervals or independent test samples.

## Highest-ranked fixed fusion by mean bar accuracy

The setting is ranked across the full cohort and all three seeds, never chosen separately for an individual song. Both strengths and all five matchers remain in the full CSVs.

| Reference | Protocol | Scope | N | Fixed fusion | Bar mean [range] % | Gain mean [range] pp | Seeds with positive gain |
|---|---|---|---|---|---|---|---|
| past | async | all | 49 | pc_edit__fused1 | 68.264 [66.942, 69.362] | 0.847 [-0.396, 1.882] | 2/3 |
| past | async | head_in | 49 | interval_ngrams__fused1 | 63.331 [61.605, 66.374] | -0.027 [-0.049, -0.009] | 0/3 |
| past | async | head_out | 49 | interval_edit__fused3 | 69.004 [67.691, 71.243] | 5.707 [3.204, 9.712] | 3/3 |
| past | async | solo | 49 | pc_edit__fused1 | 69.403 [68.277, 70.142] | 0.428 [-0.705, 1.482] | 2/3 |
| past | sync | all | 26 | interval_edit__fused3 | 73.228 [72.812, 73.571] | 0.449 [0.339, 0.611] | 3/3 |
| past | sync | head_in | 26 | interval_edit__fused1 | 84.282 [83.584, 84.788] | 0.028 [-0.021, 0.067] | 2/3 |
| past | sync | head_out | 26 | interval_dtw__fused3 | 69.015 [68.372, 69.349] | 0.811 [-0.925, 1.941] | 2/3 |
| past | sync | solo | 26 | interval_edit__fused1 | 71.489 [70.875, 72.033] | 0.438 [0.365, 0.525] | 3/3 |
| written | async | all | 4 | pc_edit__fused3 | 76.464 [76.135, 76.635] | 14.616 [12.980, 16.531] | 3/3 |
| written | async | head_in | 4 | pc_edit__fused3 | 92.787 [92.086, 93.409] | 15.798 [12.777, 18.656] | 3/3 |
| written | async | head_out | 4 | pc_edit__fused3 | 88.089 [87.179, 88.626] | 39.719 [33.339, 43.385] | 3/3 |
| written | async | solo | 4 | pc_edit__fused3 | 71.644 [71.416, 71.834] | 9.409 [7.190, 11.308] | 3/3 |
| written | sync | all | 2 | pc_cosine__fused3 | 85.741 [84.259, 88.052] | 1.926 [0.970, 3.311] | 3/3 |
| written | sync | head_in | 2 | pc_cosine__fused3 | 88.544 [87.739, 89.048] | 4.176 [3.702, 4.583] | 3/3 |
| written | sync | head_out | 2 | interval_edit__fused3 | 93.419 [92.948, 93.693] | 0.358 [0.294, 0.396] | 3/3 |
| written | sync | solo | 2 | pc_cosine__fused3 | 83.440 [81.413, 86.870] | 1.850 [0.642, 3.846] | 3/3 |

## All-frame fusion results

| Reference | Protocol | Fixed fusion | Bar mean [range] % | Gain mean [range] pp | Mean error [range] beats |
|---|---|---|---|---|---|
| past | async | interval_dtw__fused1 | 66.861 [66.484, 67.426] | -0.556 [-0.882, -0.120] | 10.549 [8.821, 13.785] |
| past | async | interval_dtw__fused3 | 65.013 [64.584, 65.825] | -2.404 [-2.708, -2.020] | 12.570 [11.332, 13.768] |
| past | async | interval_edit__fused1 | 67.757 [67.030, 69.076] | 0.340 [-0.308, 0.768] | 7.258 [4.895, 11.879] |
| past | async | interval_edit__fused3 | 66.871 [66.405, 67.295] | -0.546 [-1.013, 0.309] | 6.562 [6.054, 7.466] |
| past | async | interval_ngrams__fused1 | 67.731 [66.723, 68.793] | 0.314 [0.118, 0.485] | 7.766 [7.249, 8.327] |
| past | async | interval_ngrams__fused3 | 67.573 [66.465, 68.686] | 0.156 [-0.140, 0.378] | 7.671 [7.118, 7.953] |
| past | async | pc_cosine__fused1 | 66.442 [65.824, 67.414] | -0.975 [-1.514, -0.515] | 12.158 [11.786, 12.685] |
| past | async | pc_cosine__fused3 | 59.967 [57.866, 61.932] | -7.450 [-8.738, -6.376] | 28.959 [24.253, 37.659] |
| past | async | pc_edit__fused1 | 68.264 [66.942, 69.362] | 0.847 [-0.396, 1.882] | 6.867 [5.085, 8.099] |
| past | async | pc_edit__fused3 | 66.479 [65.633, 66.970] | -0.938 [-1.474, -0.368] | 7.396 [6.189, 8.720] |
| past | sync | interval_dtw__fused1 | 71.804 [70.279, 72.632] | -0.975 [-2.194, -0.327] | 5.131 [4.390, 6.021] |
| past | sync | interval_dtw__fused3 | 72.748 [72.661, 72.884] | -0.031 [-0.299, 0.411] | 4.673 [3.942, 5.055] |
| past | sync | interval_edit__fused1 | 73.063 [72.525, 73.395] | 0.284 [0.051, 0.436] | 4.968 [3.580, 6.400] |
| past | sync | interval_edit__fused3 | 73.228 [72.812, 73.571] | 0.449 [0.339, 0.611] | 4.975 [3.594, 6.419] |
| past | sync | interval_ngrams__fused1 | 72.871 [72.607, 73.103] | 0.092 [-0.000, 0.143] | 4.835 [3.586, 5.843] |
| past | sync | interval_ngrams__fused3 | 72.783 [72.504, 72.931] | 0.005 [-0.044, 0.030] | 4.853 [3.604, 5.873] |
| past | sync | pc_cosine__fused1 | 72.428 [71.964, 72.856] | -0.351 [-0.939, -0.010] | 4.690 [3.818, 5.198] |
| past | sync | pc_cosine__fused3 | 72.743 [72.601, 72.893] | -0.036 [-0.170, 0.128] | 5.178 [3.747, 6.578] |
| past | sync | pc_edit__fused1 | 72.731 [72.235, 72.985] | -0.047 [-0.238, 0.071] | 5.555 [5.035, 6.420] |
| past | sync | pc_edit__fused3 | 72.702 [72.357, 72.875] | -0.077 [-0.603, 0.402] | 5.554 [3.751, 6.574] |
| written | async | interval_dtw__fused1 | 74.597 [72.717, 76.174] | 12.748 [10.919, 14.807] | 1.359 [1.263, 1.444] |
| written | async | interval_dtw__fused3 | 75.400 [74.691, 76.531] | 13.551 [12.875, 14.885] | 1.337 [1.272, 1.402] |
| written | async | interval_edit__fused1 | 72.629 [71.498, 73.748] | 10.781 [10.093, 11.406] | 1.571 [1.558, 1.590] |
| written | async | interval_edit__fused3 | 75.073 [74.263, 75.728] | 13.224 [12.073, 15.134] | 1.395 [1.344, 1.427] |
| written | async | interval_ngrams__fused1 | 61.931 [60.074, 63.495] | 0.083 [-0.160, 0.427] | 12.292 [2.181, 20.592] |
| written | async | interval_ngrams__fused3 | 62.316 [60.501, 64.083] | 0.467 [0.409, 0.565] | 12.271 [2.167, 20.556] |
| written | async | pc_cosine__fused1 | 74.509 [73.308, 76.053] | 12.660 [12.367, 13.216] | 1.373 [1.294, 1.473] |
| written | async | pc_cosine__fused3 | 75.535 [75.119, 76.095] | 13.687 [12.439, 15.028] | 1.326 [1.295, 1.352] |
| written | async | pc_edit__fused1 | 73.663 [73.037, 74.577] | 11.815 [10.922, 12.945] | 1.452 [1.347, 1.515] |
| written | async | pc_edit__fused3 | 76.464 [76.135, 76.635] | 14.616 [12.980, 16.531] | 1.281 [1.279, 1.287] |
| written | sync | interval_dtw__fused1 | 84.745 [84.033, 85.908] | 0.929 [-0.448, 3.146] | 0.658 [0.608, 0.691] |
| written | sync | interval_dtw__fused3 | 84.426 [82.924, 86.080] | 0.611 [0.163, 1.338] | 0.675 [0.593, 0.743] |
| written | sync | interval_edit__fused1 | 83.505 [82.949, 83.998] | -0.311 [-1.175, 0.188] | 0.724 [0.705, 0.752] |
| written | sync | interval_edit__fused3 | 83.868 [83.054, 84.288] | 0.052 [-0.481, 0.345] | 0.713 [0.690, 0.757] |
| written | sync | interval_ngrams__fused1 | 83.815 [82.762, 84.741] | 0.000 [0.000, 0.000] | 0.715 [0.676, 0.753] |
| written | sync | interval_ngrams__fused3 | 83.815 [82.762, 84.741] | 0.000 [0.000, 0.000] | 0.715 [0.676, 0.753] |
| written | sync | pc_cosine__fused1 | 84.809 [84.702, 84.892] | 0.994 [0.092, 1.940] | 0.661 [0.655, 0.664] |
| written | sync | pc_cosine__fused3 | 85.741 [84.259, 88.052] | 1.926 [0.970, 3.311] | 0.626 [0.542, 0.682] |
| written | sync | pc_edit__fused1 | 83.962 [83.426, 84.356] | 0.147 [-0.637, 0.664] | 0.716 [0.681, 0.745] |
| written | sync | pc_edit__fused3 | 83.839 [83.637, 84.041] | 0.023 [-0.700, 0.876] | 0.706 [0.682, 0.726] |

## Interpretation

- A positive mean gain whose range crosses zero did not improve the macro bar metric in every tested seed. Stable macro signs still do not guarantee consistent gains for individual songs.
- Mean absolute beat error and within-two-beat accuracy are retained alongside bar accuracy in `seed_summary.csv`; a higher bar percentage can coexist with more severe failures.
- The written-minus-past seed comparison restricts both references to the identical written cohort and pairs within seed. It does not compare six written performances against all 75 past-memory performances.
- Matcher availability itself can vary by seed because the independent harmonic anchor controls memory labels and planned-role masks. The uncertainty therefore includes both the anchor and evaluated particle filter.
- No seed was chosen per song. Neither repeated seeds nor retrospective fixed-setting ranking establish held-out generalization, calibrated lick detection, or statistical significance.
- The original analysis limits remain: no certified lick labels, uncalibrated matching thresholds/rewards, variable-duration nine-note contexts, known unfolded plan/start, and unresolved independent AMT/audio/annotation alignment.

`seed_analysis.json` contains per-seed macros and matched-control gains, complete mean/range summaries, paired reference comparisons, per-song consistency, and provenance. CSV files preserve all methods, including harmonic controls and melody-only observation ablations. Pure melody ablations are excluded from fusion rankings.
