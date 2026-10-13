# Live-band tempo-prior comparison

This is an exploratory step toward a lead-sheet-following baseline. It uses
Maezawa-inspired harmonic template matching inside the existing simple clock
and position filter. It does not reproduce Maezawa's neural downbeat tracker
or full HHSMM follower.

The study evaluates the same 26 simultaneous (`sync`) preferred takes as the
original experiment. Overdubbed (`async`) recordings are excluded. All tests
use the original bass-and-horn transcriptions, without piano or drums, binary
triad templates, a known starting score position/time, the catalog BPM, an
already unfolded chord plan, nine tempo candidates, half-beat position bins,
and 200-ms updates. This is causal offline replay; live execution latency is
not measured.

## Controlled comparison

The original Gaussian tempo vector is used in two places: initialization and
the ongoing tempo transition rule. The study separates them in a copy of the
follower. Original source, manifests, annotations, and results are unchanged.

| Variant | Initial weights | Ongoing stationary weights |
|---|---|---|
| A | Gaussian around catalog BPM | Gaussian |
| B | Equal (1/9 each) | Same Gaussian as A |
| C | Gaussian | Equal |
| D | Equal | Equal |

A versus B is the requested test of initial tempo probabilities. Equal weights
apply only at initialization: prediction and harmonic observations subsequently
change their probabilities. Uniform dynamics change the tempo-transition
acceptance rule, including both directional preference and effective mobility.
They do not force probabilities to stay equal during the performance.

The candidate BPM values remain 0.7 to 1.3 times the catalog BPM in every
variant. Equal probabilities therefore do not remove tempo-range assumptions.
A separate A-without-harmony ablation and a constant catalog-tempo clock measure
how much the harmonic observations add on the live-band subset.

## Findings

Across the 26 pieces, average correct-bar accuracy is 54.11% for A and 53.98%
for B. Ten pieces improve, fourteen worsen, and two are unchanged. D reaches
52.32%. Initialization alone therefore has little aggregate effect in this
specific experiment; these descriptive differences are not a significance test.

Groovin' High and Samba de Orfeu were selected as the best and worst cases in
the original A results, before inspecting the new variants. A/B accuracy is
86.81%/86.11% for Groovin' High and 12.88%/10.63% for Samba. A/B reported position
estimates are identical after 57.4/76.0 seconds, respectively, although their
full probability distributions need not be identical. Samba's sustained error
after 106 seconds is therefore not repaired by equal initialization.

The poor case has sparse and ambiguous pitched observations, particularly
during piano and drum solos. In the original input, piano is omitted even when
the pianist is soloing. This is an input limitation as well as a follower
failure. Native piano MIDI and independently labeled alternate sync takes are
now available locally, but they were deliberately excluded from this controlled
comparison. Testing richer input is a separate next experiment.

These results support closer study of harmonic ambiguity, evidence availability,
and correction behavior. They do not establish that learned weighting is needed
or that any proposed adaptive weighting method will solve the observed failures.

## Metrics and references

Correct-bar accuracy uses the exact chronological bar, including repeat
occurrences. The two-beat rate and mean absolute position error use reference
positions interpolated inside annotated bar boundaries. They are position
metrics, not beat-detection scores. Reference-derived BPM is explanatory only
and is never supplied to inference. Aggregate metrics average pieces equally;
the aggregate median-error column is an average of per-piece medians.

The reported position and tempo are separate marginal modes. Tempo plots also
show the distribution mean so equal-probability tie-breaking is not mistaken
for evidence favoring the slowest tempo. Position error is shown directly
because full-song progress curves can conceal substantial errors.

## Reproduction and artifacts

Run `run_study.py --workers 3` with the bundled Python runtime. It remaps the old
manifest paths in memory, verifies MIDI hashes, and does not rewrite the manifest.
`--limit 1 --output smoke_results` produces an explicitly incomplete smoke test.

- `results/config.json`: settings, input fingerprints, and comparison definitions.
- `results/metrics.csv`: per-piece results for all variants and controls.
- `results/aggregate.csv`: equal-piece aggregate comparison.
- `results/per_song_comparison.csv`: paired A/B changes.
- `results/traces/`: position, tempo, tempo probabilities, and harmonic evidence.
- `results/case_sections.csv`: region-based descriptive diagnostics.
- `results/case_detail.json`: selected-case failure periods and one mismatch example.
- `results/summary.json`: completion and validation records.

Validation checks that the default study copy matches the original follower,
the equal initial weights are exactly one ninth, and changing dynamics alone
does not alter initialization. All 26 original position and tempo traces were
reproduced exactly. All saved tempo probabilities are finite and normalized.
