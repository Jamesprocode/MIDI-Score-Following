# Written melody reference audit

These are independent written JAZZMUS MusicXML encodings already present in the local OMR workspace. They are not first-head AMT transcriptions. Only the six (or otherwise explicitly counted) usable files under `aligned/` should enter the written-reference experiment; `events/` also includes rejected title matches.

## Source and preparation

Local source: `/Users/james/Documents/Workspace/OMR/JAZZMUS/data/train-00000-of-00001.original.parquet`. Public dataset record: https://huggingface.co/datasets/PRAIG/JAZZMUS/blob/main/data/train-00000-of-00001.parquet. The Parquet `annotation` string contains `encodings.musicxml`; original XML is preserved under `originals/`. All 14 matching rows are catalogued; the lowest row ID is chosen for the three Blue Skies variants without inspecting performance accuracy.

Music21 parses backup/forward voices, note lengths and notated repeats. Ties are merged. Grace notes are omitted. Only explicit XML transpose declarations are applied; instrument-name inferred transpositions are ignored. Missing declarations are provisionally interpreted as concert notation, then the edition is transposed using source/target chord agreement only.

Acceptance was fixed at exact root-plus-base-quality agreement ≥85% and pitched-chord coverage ≥99%, with a complete contiguous first-head span of the same length and quarter-note beats per bar, verified repeat expansion, and no unequal-note tied-best mapping. These are source-comparison criteria, not performance accuracy. No evaluated performance MIDI, audio, pitches or annotation timestamp fields are read.

| Song | Status | Base-chord agreement | Coverage | Added pitch shift | Notes |
|---|---|---:|---:|---:|---:|
| 00-a_night_in_tunisia | no_complete_matching_length_span | — | — | — | — |
| 08-blue_skies | insufficient_chord_coverage | 56.25% | 98.05% | 5 | — |
| 09-caravan | no_complete_matching_length_span | — | — | — | — |
| 13-four | unverified_repeat_expansion | 48.05% | 100.00% | 0 | — |
| 14-freddie_freeloader | usable | 100.00% | 100.00% | 2 | 30 |
| 20-impressions | usable | 100.00% | 100.00% | 0 | 67 |
| 39-st_thomas | usable | 94.53% | 100.00% | 0 | 76 |
| 41-take_five | usable | 86.67% | 100.00% | 0 | 150 |
| 54-boplicity | poor_score_chord_agreement | 76.95% | 100.00% | 5 | — |
| 60-in_a_sentimental_mood | usable | 92.58% | 100.00% | -2 | 117 |
| 62-it_dont_mean_a_thing | poor_score_chord_agreement | 82.81% | 100.00% | 0 | — |
| 66-nardis | usable | 90.62% | 100.00% | 0 | 70 |

## Coordinates and interpretation

`aligned/{protocol}__{song}.json` contains `notes` with concert-oriented MIDI pitch, `onset_beats` and `duration_beats`. Aligned onset is q in the supplied untimed JazzSAMBA plan, beginning at the first musical bar, and occupies FIRST head-in only. `score_onset_beats` and `score_duration_beats` preserve the expanded written score coordinates. Alignment applies one constant beat offset and pitch-class shift; it does not stretch rhythm or perform nonlinear warping. Pickups before the selected full head are retained only in raw events; durations crossing the head end are clipped. Written repeats generate distinct legitimate beat positions; no new notes are copied by measure-number matching here.

Title overlap covers 12 songs, not 12 validated charts. Chart agreement does not establish note-for-note agreement with the JazzSAMBA internal chart. Octave placement and swing/ornament differences remain possible. Compare written and past-phrase methods on the same usable subset; the 75-song past-phrase result is a separate cohort. Scores and source metadata are preserved unchanged outside this output directory.

## Registration QA and frozen inputs

All six accepted JSONs have strictly increasing unique onsets and no duplicate note tuples. Their source-to-q onset offset is constant, so all inter-onset intervals are preserved. Synthetic tests covered explicit transposition, two-voice backup timing, ties, repeats and chord-agreement arithmetic. Original/aligned file hashes were verified. The note JSONs are frozen while experiments run.

St. Thomas is written in2/2 while the target plan says4/4; both use four quarter-note beats per bar. Acceptance checks this common bar length, not identical metrical grouping. Freddie Freeloader and Take Five source-page images were visually checked for title, key/harmony, meter and broad notation structure; this is not an exhaustive note-level proof. Freddie explicit XML transposition−2 is followed by score-chord alignment+2; Take Five has no explicit XML transpose and keeps its source pitches.
