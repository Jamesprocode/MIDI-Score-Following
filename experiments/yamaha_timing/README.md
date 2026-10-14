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
3. At each frame, the streaming wrapper supplies up to 128 past frames (6.4 s,
   the network's whole receptive field) and an optional approximate BPM.
   Short startup histories are padded with silent frames.
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
Where it is silent we follow the two TCN beat trackers it cites: Davies & Böck
(2019), *Temporal convolutional networks for musical audio beat tracking*
(EUSIPCO), and Oyama, Ishizuka & Yoshii (2021), *Phase-aware joint beat and
downbeat estimation based on periodicity of metrical structure* (ISMIR).

| Component | Paper specification | This implementation |
|---|---|---|
| Input sampling | Running piano roll, 20 frames/s | Binary active keys, MIDI 21--108; events only up to each frame |
| History | Six-second receptive field | 128 frames (6.4 s); the streaming window equals the receptive field |
| Encoder | "The backbone takes the piano-roll": seven residual TCN layers of "two layers of 1d convolution with dilation 2^i", ELU, 512-dimensional output | Cited TCN layer (Oyama Fig. 3): dilated causal conv, ELU, dropout 0.1, 1x1 conv; 1x1 conv on the residual path; kernel size 2, dilations 1..64. No separate input layer: the first TCN layer maps the 88 keys to 512 channels |
| Heads | Separate two-layer 1x1 beat/downbeat-phase heads; 32 downbeat-phase classes | Oyama's decoder (dropout, dense, ELU, dropout, dense) as 1x1 convs; 32 bar-phase classes; 32 beat-phase classes per beat (unstated in the paper; Oyama et al. use the same class count for beat and downbeat phase); hidden width 128 is our choice |
| Tempo condition | Eight-dimensional rough-BPM representation, linear/sigmoid multiplicative gate | As written: exp(-(b - 30*2^i)^2), linear layer, sigmoid gate; masked when absent |
| Training data | Popular-song scores; random combination of tracks; random tempo drift; random note insertion/deletion; BPM condition given with 50% probability | Cleaned Lakh LMD-matched songs (`prepare_lakh.py`); the four operations in `yamaha_timing/lakh.py`, with sizes we chose (`SynthesisConfig`) |
| Loss and schedule | Not stated | Cross-entropy per head, summed. Oyama: one whole song per step, LR halved after 15 epochs without validation gain, stop after 50 such epochs or 200 epochs, gradients clipped at 0.5. Learning rate 1e-4, not Oyama's 1e-3: at 1e-3 this 512-channel model stayed at chance with loss spikes (first run, 2026-10-13); on 2,000 songs 1e-4 reached validation loss 6.69 and beat F 0.20, 1e-3 stayed at 6.93 and 0 |
| Full following | Temporal state model plus phase and harmonic likelihoods; tempo feedback | Outside the current timing-module scope |

Maezawa's "two layers of 1d convolution with dilation 2^i" and "six seconds"
only agree under the cited papers' layer: one dilated convolution plus a 1x1
convolution. With kernel size 2 and dilations 1..64 that gives 1 + 127 = 128
frames (6.4 s). The cited papers use kernel size 5, which would give 25 s, so
the six seconds fix the kernel size at 2. The earlier version of this code put
the dilation on both convolutions (255 frames, 12.75 s) and cut the input to
6 s; it is superseded (checkpoint schema 1). Oyama et al. use blurred one-hot
targets at 360 phase classes; their blur spans about 1% of a cycle, narrower
than one of our 32 bins, so plain one-hot cross-entropy is the closest match.
The cited networks are offline (non-causal); ours is causal for live use.

A whole-song causal pass equals live streaming only if the song is preceded by
127 silent frames: per-layer zero padding is not the same as silent input.
Training examples therefore start with that unlabeled lead-in
(`tests/test_framework.py` checks the equality exactly). The model remains a
documented reconstruction, not the original authors' unpublished code.

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
./.venv/bin/python prepare_lakh.py --lmd-full ../../Dataset/lmd_full \
    --metadata ../../Dataset/lmd_metadata \
    --samba-catalog "../../Dataset/SAMBA dataset/JazzSAMBA-catalog/songs.csv" --out data/lakh_v2
./.venv/bin/python train_lakh.py --data data/lakh_v2 --out runs/lakh_r1
./.venv/bin/python train_synthetic.py --steps 250 --channels 32
./.venv/bin/python demo.py --checkpoint runs/lakh_r1/best_model.pt
```

- `yamaha_timing/config.py`: explicit model dimensions and timing settings.
- `yamaha_timing/model.py`: causal residual TCN and the two phase heads.
- `yamaha_timing/streaming.py`: rolling inference and MIDI frame sampling.
- `yamaha_timing/lakh.py`: Lakh reading, label-quality checks, packed storage,
  and per-example performance synthesis with frame labels.
- `prepare_lakh.py`: builds `data/lakh_v2` (`train.npz`, `val.npz`,
  `files.csv` with every file's decision, `manifest.json` with input hashes,
  rules and the cleaning funnel). Needs `mido`; training does not.
- `yamaha_timing/evaluation.py`: causal phase-wrap decoder (phase outputs to beat
  and downbeat times; ours) and mir_eval beat-tracking metrics.
- `yamaha_timing/plots.py`: validation figure for one song (piano roll, beat/bar
  phase probabilities with the true phase, true vs predicted beats). Each epoch
  draws 2 random validation songs (`--plot-songs`) into `RUN_DIR/plots/` and W&B.
  `--warmup-steps N` adds a linear learning-rate warm-up (off by default).
- `train_lakh.py`: whole-song training with validation (phase loss and accuracy,
  plus beat/downbeat F-measure, CMLt, AMLt and the other mir_eval measures), best-model export, and
  exact resume (SIGUSR1/SIGTERM checkpoint and exit, periodic checkpoints,
  `DONE` marker) for PACE ICE job chains.
- `../lakh_survey/`: the 2,000-file survey behind the cleaning rules.
- `train_synthetic.py`: a small optimization smoke test, not music training.
- `demo.py`: chronological synthetic-frame replay and processing-time
  measurement, with a checkpoint or explicitly untrained weights. It is not
  yet a live MIDI transport or a MIDI-file parser.
- `tests/test_framework.py`: causal behavior, streaming parity, gradients,
  shape/validation, and MIDI timing checks.
- `tests/test_lakh.py`: fast rasterizer equals the live rasterizer; labels stay
  on the notes under tempo drift; lead-in; determinism; each cleaning rule on
  generated MIDI files; pack round trip.
- `asap_data_contract.md`: ASAP data and label decisions; deferred, since the
  pilot trains on Lakh as the paper does.

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
distributions as input changes.

**Data.** `prepare_lakh.py` keeps Lakh files matched to the Million Song
Dataset, removes any file whose name contains a JazzSAMBA title (SAMBA is the
test set), and applies the survey's checks: opens; at least 30 s and 50 pitched
notes; a time signature whose events fall on bar lines, with bars of at most 8
quarter notes; at least 90% of note starts within a 64th note of the
16th/triplet grid; snare not on beats 1 and 3 of 4/4 bars. All meters are kept;
bars follow the time-signature events and the beat is the denominator note
(quarter in x/4, eighth in x/8). Files sharing a song id or identical note
content form one group; note-identical copies are removed and distinct
arrangements kept. Groups are split 95/5 into train and validation by a stable
hash. Drum-channel and sound-effect notes are dropped; the input is a
pianist's keys. The evidence for these rules is in `../lakh_survey/`.

**Each training example** is one whole song rendered fresh; each epoch visits
every training group once, with one of its arrangements chosen at random:
- a random number of the song's pitched parts, chosen at random;
- each note deleted with probability ~U(0, 0.1), and ~U(0, 0.1) extra notes per
  kept note inserted at random times and pitches within the selection's range;
- tempo drift: the score's own tempo map times exp(w), where w is a
  per-quarter-note random walk (step std ~U(0, 0.03), pulled back by 2% per
  quarter note, capped at +-0.4);
- labels from the same beat-to-time map, so drift never misaligns them; beat
  phase from the position within the beat (the denominator note), bar phase
  from the position within the current bar;
- the BPM condition (local drifted tempo, in the meter's beats per minute)
  given for 50% of examples, as in the paper.
Notes are cut to at most 10 s after timing (`max_note_seconds`): Lakh has
stuck notes (377 training songs hold a note over 60 s, up to 45 min).
The paper names these operations but not their sizes; the sizes are ours.
Validation uses each group's best-matched arrangement with one fixed rendering,
and reports results for 4/4 and other meters separately.

## Training on PACE ICE

Run `ice` (`~/.claude/skills/pace-ice/scripts/ice`) from the repo root.
`.ice.env` holds the settings: project name `MIDI-Score-Following` (the local
folder name has spaces), the `midi-score-following` conda env (Python 3.10,
torch 2.10.0+cu128, numpy 1.26.4, wandb 0.25.0, mir_eval 0.8.2, matching the
local env), any NVIDIA GPU (`gpu:1` with the `nvidia-gpu` node feature, so never
the AMD MI210; the model is small and the V100 works), 8 CPUs, 32 GB, and the
data-ready marker. `ice submit` refuses
uncommitted code, so commit and push first.

```sh
ice sync                     # ICE checkout fast-forwards to the pushed commit
# GPU check (5 min)
ice sh 'srun -p ice-gpu --gres=gpu:1 --constraint=nvidia-gpu -t 5 bash -c "source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate midi-score-following && python -c \"import torch; print(torch.cuda.get_device_name())\""'
# Data: copy, verify against manifest.json, mark ready
rsync -a --partial --progress experiments/yamaha_timing/data/lakh_v2 ice:scratch/MIDI-Score-Following/data/
ice sh 'source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate midi-score-following && python experiments/yamaha_timing/verify_data.py ~/scratch/MIDI-Score-Following/data/lakh_v2 --mark'
# Smoke run: 300 song groups, one epoch; measures GPU speed, W&B, checkpoints
ice submit lakh_smoke -t 20 -- python train_lakh.py --data /home/hice1/jwang3180/scratch/MIDI-Score-Following/data/lakh_v2 --limit-train 300 --limit-val 50 --epochs 1
# Full run (resubmits itself until DONE)
ice submit lakh_r1 -- python train_lakh.py --data /home/hice1/jwang3180/scratch/MIDI-Score-Following/data/lakh_v2
ice status; ice log lakh_r1; ice fetch lakh_r1 best_model.pt
```

`train_lakh.py` writes to `$RUN_DIR` (set by the job chain), resumes from its
newest checkpoint, and saves and exits on the chain's SIGUSR1 before the wall
time. Data builds are reproducible: rebuilding from the same inputs and code
gives byte-identical packs (checked on the first build).

## Verified framework run (2026-10-11), earlier architecture

This run used the superseded two-dilated-convolution layer (checkpoint schema
1); its toy checkpoint no longer loads.

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

The pilot trains on Lakh, as the paper trains on popular-song scores, and
tests on JazzSAMBA piano MIDI (all 76 songs have recorded piano MIDI with bar
timestamps). ASAP is deferred; see the data contract if it is added later.
JazzSAMBA bar timestamps label downbeats exactly but not every performed beat.
