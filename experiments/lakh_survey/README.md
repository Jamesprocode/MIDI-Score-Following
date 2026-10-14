# Lakh MIDI survey: can its tempo maps label beats and downbeats?

The Maezawa (2023) baseline trains on popular-song scores with tempo drift.
We use the Lakh MIDI Dataset as the public stand-in, as Jiang, Maezawa & Xia
(2025) did. Beat and downbeat labels come from each file's own tempo map and
time signature, so they are only correct if the file was sequenced on its beat
grid. This survey measured how often that holds, and it set the cleaning rules
used by `../yamaha_timing/prepare_lakh.py`.

Run on 2026-10-13 on a random sample of 2,000 of the 178,561 LMD-full files
(seed 0). Both scripts only read the MIDI files.

```sh
python lakh_survey.py ../../Dataset/lmd_full results 2000 0
python lakh_middle_band.py results/rows.jsonl
```

## What Lakh contains

- 16 folders (`0`–`f`) of `.mid` files named by MD5 checksum (5.9 GB). No titles
  or song IDs; `md5_to_paths.json` gives original file names and
  `match_scores.json` links 45,129 files to Million Song Dataset entries.
- The median file is 3.3 minutes long, with 8 instrument parts. 70% have a piano
  part, 69% drums, 64% bass, and 57% guitar; 13% carry karaoke lyrics.
- Time signatures: explicit constant 4/4 in 65% of files; none in 9.5%; another
  single meter in 11%; meter changes in 14%.
- Tempo: one tempo in 73%; 13% sit at exactly 120 BPM, the MIDI default and a
  sign of files played in without a click. The median first tempo is 120 BPM
  and the middle half run 92–132 BPM.
- 2.1% of files do not open.

## Cleaning funnel on the sample

| Step | Files left |
|---|---|
| Sampled | 2,000 |
| Opens without error | 1,958 |
| At least 30 s and 50 pitched notes | 1,797 |
| Explicit, constant 4/4 | 1,182 |
| 90% of note starts exactly on a 16th/triplet position | 499 |
| Snare not on beats 1 & 3 | 475 |

The exact-grid rule was too strict. Of the 683 4/4 files it rejected, 390 are
humanized: at least 90% of note starts lie within a 64th note of the grid, so
their beat labels are still right (`results/example_humanized.png`: median
offset 4 ms). Another 44 use 32nds or 16th-triplets, and 5 carry one constant
offset. Only 244 are genuinely played in. Random note starts would fall within
a 64th note of the grid 67% of the time, so the 90% bar stays informative.

**Adopted rule:** at least 90% of note starts within a 64th note of a 16th or
triplet position, or at least 90% exactly on a grid that adds 32nds and
16th-triplets. Under it about 44% of all files, and about 58% of the matched
files, pass every check.

The drum check flags files whose snare lands on beats 1 and 3. In
`results/example_shifted.png`, every chord also changes one beat before the bar
line, so the bars really are off by one beat.

## Song identity and test-set leakage

- Assigning each matched file to its best-scoring MSD id gives about 19,000
  songs. Linking files through every shared match instead chains 18,273 files
  into one group, because Lakh's matching is noisy.
- JazzSAMBA titles appear as whole words in 539 original file names (99 in the
  matched set). Short titles such as "Four" also catch unrelated songs; all are
  removed, because SAMBA is the test set.

## Follow-up: are other meters trustworthy, and are versions duplicates?

The first build kept only constant 4/4 files and one version per song. Both
choices were questioned, so we measured them (`meter_and_duplicates.py`,
`results/meter_and_duplicates.json`).

**Meters.** On 11,940 random matched files, kick-drum and bass-instrument onsets
were tested against each file's stated bar: does the pattern repeat every 3 or
every 4 quarter notes, and is the stated beat 1 the most accented position?

| Stated meter | Share of files | Repeats every 3 quarters | Stated beat 1 strongest |
|---|---|---|---|
| 4/4 (reference) | 74% | 10% | 84% |
| 3/4 | 2.9% | 92% | 96% |
| 6/8 | 1.0% | 100% | 97% |
| 12/8 | 0.5% | 93% | 91% |
| 2/4 | 1.8% | 17% | 86% |
| Meter changes | 12.5% | — | — |
| No time signature | 6.5% | 26% (a mix) | — |

Stated 3/4, 6/8 and 12/8 agree with the drums and bass at least as well as
4/4, so label reliability does not justify dropping them. Meter-change files
are mostly 4/4 with a few odd bars: the main meter covers 95% or more of 72%
of them, and the most common combination is 2/4 + 4/4 (one short bar). Files
without a time signature are a mixture and stay excluded.

**Decision:** keep every meter whose bars can be counted from the
time-signature events (bars of at most 8 quarter notes). The beat is the
time signature's denominator note (quarter in x/4, eighth in x/8, so 6/8 has
six beats per bar). Constant 2/4 is kept too; it is about 2% of files.

**Duplicates.** Among the 25,494 files that passed the first build, 13,557
were extra versions of an existing song. 6,289 of them (46%) have note-for-note
identical content to another version (note starts to 1/48 quarter note and
pitches); the other 7,268 are genuinely different arrangements. Skew is
moderate: the top 1% of songs hold 8.9% of files (largest: 154 versions).
745 identical contents appear under different song ids.

**Decision:** remove note-identical copies, keep distinct arrangements, and
sample one arrangement per song group per training epoch. Files sharing a song
id or identical content form one group, so copies under different ids land in
the same split.

## Files

- `lakh_survey.py`: per-file survey, funnel, and example plots.
- `lakh_middle_band.py`: why 4/4 files fail the exact-grid rule.
- `meter_and_duplicates.py`: the meter and duplicate measurements above.
- `results/summary.json`, `results/example_*.png`, `results/meter_and_duplicates.json`:
  the runs reported above.
  `rows.jsonl` (per-file rows with local paths) is regenerated by the survey
  and not committed.
