# ASAP pipeline: next-stage data contract

This document records what the standalone TCN needs from the next data stage.
It does not claim that ASAP has been downloaded, audited, or used to train the
baseline. No full data pipeline is implemented here yet.

## Required source material

Use performance MIDI with that same performance's corrected beat/downbeat
annotations from the [official ASAP repository](https://github.com/fosfrancesco/asap-dataset).
Do not pair a performance's timestamps with a different performance or with
the quantized score MIDI. Preserve the original annotation files and record
their path, source revision, and file hash in the prepared-data manifest.

Each manifest item should identify:

- composition and performance IDs;
- performance MIDI path and corrected annotation path;
- training, validation, or test split;
- accepted meter segments and excluded annotation regions;
- any recorded offsets, verification notes, or preprocessing options.

Split by composition before cropping or augmentation. Performances of the
same composition and all transformed versions must stay in the same split.
The test split remains excluded from model selection and JazzSAMBA fine-tuning.

## First supported setting

Start with piano performance MIDI in complete, valid 4/4 measure regions.
Screen missing annotations, non-increasing beat times, pickup measures, meter
changes, incomplete terminal measures, and MIDI/annotation alignment offsets.
Exclude or mask ambiguous regions instead of silently assuming beat zero is
the first note or that every crop begins on a downbeat. Other meters require
an explicit labeling policy before inclusion.

## Input representation

Produce binary note-active piano-roll frames with 88 columns for MIDI pitches
21--108 and a 20-Hz timestamp grid. Use the same event ordering, overlapping
note counting, and note-off convention as `MidiFrameRasterizer`. Training can
read a complete file to schedule events, but each frame may contain only
events whose timestamps are at or before that frame. Future note durations
must not appear as input features. The first adapter does not model sustain
pedal, so any later pedal-aware representation needs matching training and
inference behavior.

The network input is a float tensor `[batch, time, 88]`. An optional BPM input
and its availability mask use the model's supported shapes. Beat annotations
and reference phase are targets, not additional input channels.

## Phase targets

For each frame, corrected neighboring beat annotations define fractional
progress through a beat. Downbeat flags and meter define the beat number
inside the measure. Together they define:

- a beat-phase class in 32 bins;
- a measure-phase class in 32 bins;
- separate masks indicating which targets are valid.

Both class counts match the initial code configuration; only the measure-phase
count is explicitly specified in the Maezawa paper. Target interpolation
between beat annotations is our training convention and should be stored in
the preprocessing configuration. Using future reference beats to construct
offline training targets is acceptable; the reference times cannot feed the
inference model or its BPM hint.

In 4/4, measure phase should follow the annotated sequence of beats and their
fractional progress. Uniformly dividing a whole measure's duration can hide
within-measure tempo variation. Retain the original phase when making crops.
Mask timestamps outside reliable annotated intervals, including startup and
trailing partial measures as necessary.

## Approximate BPM conditioning

Define how a BPM hint would be available in a live performance before training
with it. Acceptable sources include a manually supplied starting tempo or a
tempo estimate computed strictly from past input. Do not use a full-performance
reference-derived tempo trajectory during testing. Simulated hints may be
noisy and randomly absent during training, following the paper's motivation;
their noise, dropout, and update policy must be explicit. Missing hints must
use the model's validity mask rather than pretending that zero BPM is valid.

## Augmentation

Apply global tempo changes and smooth time warps to notes and reference
timestamps together. Note insertion/deletion and mild onset perturbation can
simulate performance differences while retaining the intended pulse targets.
Pitch transposition must respect the supported MIDI range. All random seeds
and transformation parameters should be reproducible; augmentation is
training-only unless an explicitly labeled stress test is requested.

## Training handoff and acceptance checks

The loader must return piano roll, optional BPM hint/availability, both phase
targets, both label masks, performance ID, and source timestamps. Loss should
be computed only on valid targets and on observations assigned to the current
rolling window. Any initial classification-loss/head weighting is an explicit
implementation choice; it is not stated in the demonstration paper.

Before full training:

1. Visually inspect notes and targets for several performances, including
   pickups, sparse playing, and tempo changes.
2. Verify that train/validation/test composition IDs do not overlap.
3. Compare event replay with saved piano rolls at known note-on/off boundaries.
4. Confirm that time warps change notes and labels consistently.
5. Overfit a few real excerpts and verify causal streaming predictions.

Synthetic framework success is insufficient evidence of musical accuracy.
Held-out ASAP results and transfer to jazz must be reported separately.

## JazzSAMBA follow-up

Native JazzSAMBA piano MIDI can test jazz transfer or supply a song-separated
fine-tuning subset. Its bar timestamps support reference downbeat times, but
not independently annotated individual beats. Interpolated within-bar targets
are approximate labels and must be labeled accordingly. A manually checked
beat-annotation subset would support a more precise timing evaluation.
