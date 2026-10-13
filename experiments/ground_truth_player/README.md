# MIDI following comparison player

A local, dependency-free webpage with MIDI playback, an annotation reference, three selectable harmonic input conditions, and a fixed-tempo clock:

- **Blue:** annotation reference, interpolated within each annotated bar.
- **Orange, dashed:** the selected bass-only, horn-only, or combined bass + horns harmonic particle follower (`p02_particles_slow`), with a new estimate every 200 ms.
- **Green, dotted, with a ring marker:** a clock using the supplied starting tempo for the entire performance: 160 BPM for Blue Bossa, 165 for Green Dolphin Street, and 200 for Samba de Orfeu. It starts at the same known musical start, uses only elapsed time, and receives no note evidence or later tempo/position corrections. The clock does not change when the tracking input changes.
- **Tracking input selector:** bass only, horns only, or bass + horns. Horns are alto saxophone plus trumpet in Blue Bossa, alto saxophone in Green Dolphin, and trumpet in Samba de Orfeu. It changes the independently computed trace without changing playback time or audio mutes.

Three recordings are included: On Green Dolphin Street (synchronous, bass/drums/alto saxophone, 2,237 notes), Blue Bossa (asynchronous, bass/alto saxophone/trumpet, 1,466 notes), and Samba de Orfeu (synchronous, bass/drums/trumpet, 4,510 notes). Samba de Orfeu was deliberately selected as a poor p02 example; these three demonstrations are not a representative benchmark sample. The browser synthesizes transcribed MIDI notes at their original timing and velocity. Its timbres are simple browser synthesizers, not the original instruments or a General MIDI soundfont. Track checkboxes change what is heard; they do not alter any precomputed follower. Both drum MIDIs use General MIDI percussion channel 10 (zero-based channel 9).

## Open

From this directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8765>. Use Play/Pause, the seek slider, section jumps, clickable bars, or ±200 ms stepping. The default chart follows a close-up window; check “Show the entire performance” for the overview. Space plays/pauses when the page background has focus.

All three displayed positions use the same 200 ms sample times. Green chord/section labels are read from the plan at the clock's predicted position; they do not correct the clock. On Blue Bossa, green overlaps the blue reference because its annotation grid exactly matches the supplied tempo. The ring marker remains visible around the reference dot, and green dots overlay the blue line. Nearby-bar green right edges indicate the clock's current bar.

## Data and meaning

Source MIDI is under `../../SAMBA dataset/async_midi/songs/07-blue_bossa/preferred`, `../../SAMBA dataset/sync_midi/songs/67-on_green_dolphin_street/preferred`, and `../../SAMBA dataset/sync_midi/songs/70-samba_de_orfeu/preferred`. Downloaded annotations are under `../ensemble_benchmark/data/annotations/{async,sync}`. Exact absolute paths and SHA-256 hashes are embedded in each JSON payload and audit. Original inputs were not moved or modified. Piano is absent from these selected derived-MIDI tracks; a separate piano MIDI in the audio directory is not included or used by the follower.

Blue Bossa has leading padding from 0 to 1.5 seconds. Performance bar 1 corresponds to chart measure 9; the first head starts at 13.5 seconds. Its annotations end at 360 seconds and MIDI tails at 361.36953125 seconds. Green Dolphin's musical start is 1.397 seconds, annotations end at 246.486 seconds and MIDI tails at 255.062623725 seconds. The page shows no ground-truth position beyond either annotation endpoint. Performance bar numbers distinguish repeated chorus occurrences.

Samba de Orfeu starts musically at 1.134292 seconds, with performance bar 1 corresponding to chart measure 28. It contains 356 musical bars; annotations end at 411.070958 seconds and MIDI tails at 411.160625 seconds. Its input condition called “horn” contains trumpet only.

Reference position within each bar is linearly interpolated from its annotated start/end. These are not independent beat-onset annotations. `referenceBpm` is the tempo averaged over the current annotated bar. Blue Bossa is exactly 160 BPM. Green Dolphin ranges from 171.80 to 200.00 BPM, with median 188.98 BPM, despite its supplied catalog tempo of 165 BPM. Samba de Orfeu ranges from 192.98 to 233.44 BPM, with median 209.41 BPM and supplied catalog tempo 200 BPM. Ground truth means position annotation; the MIDI notes are automatic transcriptions.

Each follower receives the same known start, catalog BPM, and untimed chord plan. All three input conditions use instantaneous active-note piano-roll compatibility with identical root-weighted triad templates and the same continuous position/tempo particle filter, configuration, and random seed. The template gives chord roots weight 2 and other triad tones weight 1. Selected streams are pooled by binary union across MIDI keys: simultaneous unisons count once, while different octaves remain separate. The combined condition pools bass and horn notes **before harmonic scoring**, without averaging their independent position estimates or choosing adaptive weights. Bass cannot contribute to the horn condition, and horns cannot contribute to the bass condition; drums and piano are excluded from all three. There is no melodic-order matching, neural beat detector, adaptive fusion layer, or subsequent ground-truth timing input. These are exploratory baselines, not reproductions of the full Yamaha or Pardo systems. Seeking replays saved states at that time; it does not restart inference there.

The comparison changes only the selected input streams; it does not tune a new matcher for horns. Root-weighted templates may favor bass material. A poor horn result therefore does not establish that melodic information is unhelpful. With no active selected notes, observation updates are skipped and the filter continues predicting from its previous position/tempo distribution.

## Rebuild and verify

Run `prepare_data.py` with Python and NumPy available:

```sh
python3 prepare_data.py
```

In this workspace, the bundled Python can also be used:

```sh
/Users/james/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 prepare_data.py
```

This rebuilds `data.json` / `data_audit.json` (Blue Bossa), `green_dolphin.json` / `green_dolphin_audit.json` (Green Dolphin), and `samba_de_orfeu.json` / `samba_de_orfeu_audit.json` (Samba de Orfeu). It reconstructs the existing p02 follower and checks MIDI note/velocity preservation, annotation timing, 200 ms spacing, source hashes, prefix causality, and exact agreement of the bass runs with earlier saved traces. All input conditions are tested by deleting or heavily modifying excluded tracks and removing future note events. Combined features are checked against the binary union of component piano rolls and against a piano roll of the concatenated selected MIDI notes. Rebuilding also verifies that all previously generated sample objects and metrics remain exactly unchanged. No training or source-dataset edits occur.

The original top-level `samples` remain the unchanged bass trace. `followers.bass`, `followers.horn`, and `followers.both` contain their own samples, metrics, instrument lists, and evidence coverage. `activeEvidenceNotes` and `evidenceAvailable` describe the selected input at each sample; legacy bass samples retain `activeBassNotes`. `evidenceCoverage` is the fraction of evaluated frames with at least one active selected MIDI key. It is not a measure of harmonic correctness.

| Recording / input | Correct chronological bar | Within two beats | Mean absolute beat error | Median absolute beat error | Evidence coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blue Bossa / bass | 88.34% | 94.48% | 0.797 | 0.370 | 56.00% |
| Blue Bossa / horns | 52.70% | 59.96% | 2.260 | 1.486 | 53.54% |
| Blue Bossa / bass + horns | 88.40% | 94.81% | 0.767 | 0.387 | 81.04% |
| On Green Dolphin Street / bass | 90.46% | 99.51% | 0.474 | 0.336 | 81.57% |
| On Green Dolphin Street / horn | 17.05% | 18.03% | 32.414 | 30.866 | 41.92% |
| On Green Dolphin Street / bass + horn | 90.86% | 99.35% | 0.483 | 0.401 | 83.28% |
| Samba de Orfeu / bass | 10.93% | 13.61% | 31.077 | 23.084 | 79.61% |
| Samba de Orfeu / trumpet | 24.83% | 26.49% | 13.768 | 6.437 | 43.27% |
| Samba de Orfeu / bass + trumpet | 17.85% | 20.15% | 16.118 | 6.970 | 80.73% |

These metrics cover **all** evaluated 200 ms frames, including frames with no selected notes (1,793 frames for Blue Bossa; 1,226 for Green Dolphin; 2,050 for Samba de Orfeu). Optional `metricsByEvidence.available` and `.absent` report the same position metrics separately, each with its own `frames` denominator. These are this exploratory baseline's per-recording replay results, with the known start and supplied catalog BPM. Blue Bossa's exact supplied tempo makes a fixed clock a perfect reference for that take; none of the harmonic followers outperforms it.

Pooling is not consistently beneficial: it slightly improves Blue Bossa's bar accuracy and mean error; Green Dolphin's bar accuracy improves slightly while its beat errors worsen slightly; Samba's combined result falls between bass-only and trumpet-only. These examples do not demonstrate learned or adaptive cue weighting.

To inspect a failure, choose Samba de Orfeu and seek to 20 seconds: the bass follower is about two bars behind. It briefly recovers around 50–90 seconds, then remains more than four beats wrong from 120 seconds to the final update. At 300 seconds its estimate is about 52 beats behind. The trumpet-only run also fails but follows a different error trajectory; this is an input comparison, not an adaptive fusion result.

## Why this simple method can work

The tracker maintains competing position/tempo hypotheses over time; it does not identify a bar independently from each note. It knows the first position and complete unfolded chord order, including planned repeats and solo choruses. A weak pitch match can help choose between ongoing hypotheses even when it cannot identify a unique chord. With no current notes, past estimates still advance the clock. This setup tests position maintenance from a known start, not recovery from unknown starts, jumps or unplanned repetitions.

`diagnose_sources.py` reproduces extra controls in `diagnostic_controls.json`:

| Control (same evaluation frames) | Blue Bossa correct bar | Green Dolphin correct bar |
| --- | ---: | ---: |
| Fixed supplied-tempo clock | 100.00% | 4.32% |
| Existing bass/root follower | 88.34% | 90.46% |
| Bass follower without extra root weight | 83.27% | 88.66% |
| Same filter with shuffled chord order | 0.73% | 1.55% |

Blue Bossa demonstrates no improvement over a clock. Green Dolphin shows useful harmonic correction under the stated initialization and plan assumptions. Five particle seeds give bass correct-bar ranges of 87.23–88.62% and 89.72–90.70%, respectively. The additional no-harmony particle control has a deliberately broad tempo prior and becomes diffuse, so it should not replace the much clearer fixed-clock comparison.

The p02 configuration was selected after examining this corpus (`../ensemble_benchmark/improvement_study/secondary_validation.md`). These post-hoc comparisons on two examples are exploratory, not evidence of unseen-song generalization. In particular, poor results from feeding horns into a bass-developed template do not measure the best possible melodic follower. Horn runs include the whole horn tracks, including heads and improvised passages; no annotation-based solo selection is applied.

The page uses the latest trace sample at or before the audio clock. A 50 ms browser poll detects new samples; source positions change at 200 ms intervals. Browser background throttling can skip visible updates; the next update catches up to the audio clock. Playback computation and sample spacing do not establish a measured real-time inference latency.

The player displays instantaneous errors and mean absolute errors in approximate milliseconds. At each check, it converts the signed beat difference with `60000 / referenceBpm`, using that check's annotated bar-average tempo; the summary averages the absolute converted errors, rather than converting a mean beat error using one song-wide BPM. This is a local-tempo time equivalent, not exact timestamp alignment across changing-tempo bars or processing latency. Positions, plots, stored traces and benchmark metrics retain their original musical units. The chord strip is labeled “Score positions.”
