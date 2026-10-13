# Maezawa-inspired MIDI timing baseline

This directory contains the first, standalone timing component of a Yamaha/
Maezawa-inspired lead-sheet follower. It receives incoming piano MIDI and
predicts local musical phase. It does not yet identify a bar in a chart,
estimate tempo, match harmony, or schedule accompaniment.

The distinction matters: the 32 measure-phase values describe locations within
one measure. The same phase occurs in every measure. In the complete paper's
system, a temporal follower combines these observations with harmonic evidence
to estimate chart position and tempo. Tempo is fed back to the network as an
input condition. It is not a direct output of this TCN.

## Input and output

1. MIDI note-on/off events update an 88-key binary piano roll (MIDI 21--108).
2. The roll is sampled at 20 Hz: one frame every 50 ms.
3. At each frame, the streaming wrapper supplies up to 120 past frames and an
   optional approximate BPM. Short startup histories are padded with zeros.
4. A causal TCN returns beat-phase and measure-phase distributions. The wrapper
   emits the current prediction's timestamp, both probability lists, and a
   flag saying whether the rolling context is full.

For example, in 4/4, measure-phase bin 16 of 32 is approximately the start of
beat three. It does not tell us which bar contains that beat. Output softmax
normalizes the network scores; an untrained distribution is not a validated
confidence estimate.

## What comes from the paper, and what we chose

Source: Maezawa (2023), *Automatic Accompaniment from a Lead-sheet by Jointly
Tracking at Chord and Downbeat Level*, [ISMIR late-breaking demonstration](https://ismir2023program.ismir.net/lbd_365.html).

| Component | Paper specification | This implementation |
|---|---|---|
| Input sampling | Running piano roll, 20 frames/s | Binary active keys, MIDI 21--108; events only up to each frame |
| History | Stated six-second receptive field | Explicit 120-frame rolling inference window |
| Encoder | Seven residual TCN stages, two convolutions per stage, powers-of-two dilation, ELU, 512-dimensional output | Causal left padding, kernel size 2, fixed channel width; these missing details are explicit choices |
| Heads | Separate two-layer 1x1 beat/downbeat-phase heads; 32 downbeat-phase classes | 32 classes in both heads; beat class count and hidden width are reconstruction choices |
| Tempo condition | Eight-dimensional rough-BPM representation, linear/sigmoid multiplicative gate | Optional masked BPM conditioning; no tempo prediction head |
| Training | Popular-song scores with track selection, tempo drift, note insertion/deletion; BPM condition sometimes absent | Synthetic framework check first; annotated ASAP performance training is the next stage |
| Full following | Temporal state model plus phase and harmonic likelihoods; tempo feedback | Outside the current timing-module scope |

The architecture details in the paper do not uniquely determine a six-second
convolutional receptive field. Seven stages with two kernel-size-2 convolutions
and dilations 1--64 give a nominal receptive field of 255 frames (12.75 s).
We retain those stages but constrain each streaming prediction to a six-second
rolling input. The resulting model is a documented reconstruction, not a claim
to reproduce the original authors' unpublished implementation exactly.

Past context does not imply six seconds of future delay. New predictions can
be made every 50 ms, including during startup; `context_ready` becomes true
after the buffer is full. CPU processing time must still be measured against
that update interval. No real-time hardware guarantee or musical accuracy is
established by the framework tests.

## Files and execution

Run commands from this directory after installing its requirements into the
local environment:

The checked local `.venv` uses Python 3.10 and inherits the already installed
packages from `/Users/james/miniconda3/envs/AMT` without changing that environment.
The verified versions are PyTorch 2.10.0 and NumPy 1.26.4. On a different machine,
create a Python 3.10+ virtual environment and install `requirements.txt` there.

```sh
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python train_synthetic.py --steps 250 --channels 32
./.venv/bin/python demo.py --checkpoint runs/synthetic/checkpoint.pt
```

- `yamaha_timing/config.py`: explicit model dimensions and timing settings.
- `yamaha_timing/model.py`: causal residual TCN and the two phase heads.
- `yamaha_timing/streaming.py`: rolling inference and MIDI frame sampling.
- `train_synthetic.py`: a small optimization smoke test, not music training.
- `demo.py`: chronological synthetic-frame replay and processing-time
  measurement, with a checkpoint or explicitly untrained weights. It is not
  yet a live MIDI transport or a MIDI-file parser.
- `tests/test_framework.py`: causal behavior, streaming parity, gradients,
  shape/validation, and MIDI timing checks.
- `asap_data_contract.md`: required data and label decisions for the next stage.

## Streaming contract

`StreamingTimingTracker(model).push_frame(frame, timestamp_s, bpm_hint=None)`
returns a JSON-serializable record containing `timestamp_s`,
`beat_phase_probabilities`, `bar_phase_probabilities`, `context_frames_seen`,
and `context_ready`. Frames must arrive on the configured sampling grid;
timestamps refer to the input frame, not the completion time of processing.
`reset()` clears past context before a new performance.
The wrapper retains the BPM hint available at each frame; a later hint never
rewrites the conditioning of earlier input frames.

`MidiFrameRasterizer.push_event(event_type, pitch, velocity, timestamp_s,
channel=0)` queues a MIDI note event. `advance_to(timestamp_s)` emits sampled
frames up to that time. Queue all simultaneous events before advancing to
their timestamp. This prevents a later event from changing an already emitted
frame. Velocity-zero note-on is treated as note-off; overlapping held notes
are counted per channel/key. Out-of-piano-range pitches are ignored and counted.
Unmatched note-offs are counted rather than creating negative note counts.
Pedal and controller events are not modeled in this first adapter.

The rasterizer uses neither future note-off times nor full-file beat metadata.
The same sampling convention must be used for training and inference.

## Training and inference

Training compares the two heads' logits with phase labels and updates network
parameters. Inference keeps those parameters fixed and produces new
distributions as input changes. Synthetic overfitting verifies that the code
can optimize and execute; it does not show that the network understands timing
in human performance. Real-data training, held-out musical evaluation, and
integration with the position--tempo filter remain next steps.

## Verified framework run (2026-10-11)

- All 21 tests passed, including future-input invariance, event boundaries,
  rolling-window equivalence, label masks, and checkpoint restoration.
- A reduced-width, 32-channel model overfit the deliberately simple periodic
  keyboard example in 250 optimization steps. Both heads reached 100% accuracy
  on those same toy windows, with and without a BPM hint. This is a training-path
  check, not a held-out music result.
- The full 512-channel model produced finite gradients in a training backward
  pass. Its untrained streaming computation was profiled for 160 updates on
  local CPU with one PyTorch thread: median 24.76 ms, 95th percentile 26.28 ms,
  maximum 33.20 ms. No update exceeded the 50-ms sampling interval in that run.
  This excludes a live MIDI driver, scheduling load, and end-to-end latency.
- `runs/synthetic/metrics.json` and `checkpoint.pt` contain the toy-training
  result and its explicitly synthetic provenance. `runs/default_cpu/summary.json`
  contains the compute profile; `runs/synthetic_stream/phase_stream.jsonl` shows
  chronological phase output from the toy-trained checkpoint.
- `runs/verification.json` collects the results and hashes of the verified
  source files. No ASAP download or real-performance training is recorded.

ASAP will provide piano MIDI and corrected beat/downbeat annotation files.
We will first establish a composition-separated 4/4 dataset with audited
phase targets. JazzSAMBA may later support jazz transfer/fine-tuning, but its
bar timestamps do not independently label every performed beat. See the data
contract before constructing or reporting those targets.
