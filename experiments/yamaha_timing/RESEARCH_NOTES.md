# Research notes: Yamaha baseline on jazz

Running log of findings and directions to revisit. Each entry states the
evidence it rests on and what would test it.

## 2026-10-14: real-data results, and three directions

### Results these rest on

The timing network (Maezawa 2023 recreation) was trained on synthetic Lakh
performances and scored with `eval_real.py`. Details: W&B runs `samba_r3b`,
`samba_r2`, `asap_r3b` and `asap_r2`, and `~/scratch/MIDI-Score-Following/runs/`
on ICE.

| BPM hint given | Beat F | Downbeat F |
|---|---|---|
| Synthetic validation (Lakh) | 0.78 | 0.585 |
| SAMBA async (click, 100 takes) | 0.51 | 0.30 |
| SAMBA sync (live band, repaired timing, 52) | 0.54 | 0.37 |
| ASAP classical (1,067, oracle tempo) | 0.58 | 0.25 |

- The two training configurations tie on real data (within 0.02). They are
  `configs/oyama.json` (lakh_r2) and `configs/batch16_cosine.json` (lakh_r3b).
- No SAMBA take reaches downbeat F 0.8; the median is 0.31. The figures show
  the tracker locking on for stretches, then drifting or settling on the wrong
  beat of the bar.
- Withholding the BPM hint roughly halves beat continuity (SAMBA async CMLt
  0.27 to 0.15).
- SAMBA breakdown:
  - Swung and straight songs score the same (beat F 0.52 both; downbeat F
    0.31 vs 0.35).
  - Every genre lands near 0.5 beat F.
  - Continuity falls with tempo: beat CMLt is 0.56 below 120 BPM and 0.22
    above 170.
- Tempo drift is not the main difficulty. On 200 synthetic validation songs,
  removing drift entirely raises downbeat F only from 0.53 to 0.56. Our drift
  is also larger than a live band's: the median 4-bar tempo range is 34% vs
  14%.
- Real piano timing is looser than the training data. SAMBA async onsets near
  a beat sit a median 31 ms from the click, with a second cluster around the
  swung eighth. Lakh is mostly quantised.

### 1. A pop-trained baseline does not transfer directly to jazz

**Claim:**
- Maezawa trains on popular-song scores, and we did the same with Lakh.
- On jazz standards played on piano, beat F drops from 0.78 to about 0.5 and
  downbeat F from 0.585 to about 0.3.
- Directly transferring the pop-style baseline is therefore not enough. Jazz
  lead-sheet following needs its own work: an argument for opening up this
  problem.

**Limits:**
- This is one tracker with one training recipe, kept strict on purpose.
- A tracker trained on jazz could do better.
- State the general claim together with the structural argument in
  direction 2.

**To test:**
- **Controlled swing test:** delay the off-beat eighths of the synthetic
  validation songs to the swung position and compare. That isolates swing.
  The SAMBA breakdown above suggests swing alone is not the main factor;
  loose, syncopated comping in general may be.
- **Train or fine-tune on jazz:** use SAMBA's official train split (40 async
  + 20 sync songs) and test on its test split. Survey other jazz sources with
  beat annotations, e.g. the Weimar Jazz Database, PiJAMA and JAAH; each still
  needs its suitability checked.
- **Literature:** collect published jazz beat/downbeat tracking results to
  frame the claim.
- All of this goes beyond the strict replication. Keep it separate from the
  pilot.

### 2. Piano-only beat evidence is weak, so the cues need adaptive weighting

**Observation:** jazz piano comping is sparse, syncopated and off the beat,
and the tracker loses the downbeat on it.

**Hypothesis:**
- A steady bass and drums would give much stronger beat evidence.
- **Measured 2026-10-15:** piano + transcribed bass raises SAMBA beat/downbeat F from 0.52/0.33 to 0.61/0.39 (live takes: downbeat 0.37 to 0.48). Adding the horns does not add more. Bass alone marks beats but not bars. Drums are still untested, since the network never saw them. Details are in REPORT_2026-10-15.md §4b.
- With piano alone, the follower should lean on harmony.
- That is the case for weighting the cues by their reliability.

**Structural argument, true even for a perfect tracker:**
- Beat and bar phase are cyclic. They say where you are within a bar, not
  which bar of the form you're in.
- One missed downbeat shifts the bar count permanently.
- Harmony identifies the bar; the beat places you within it.

**To test:**
- **Tracker on more instruments:** compare piano alone, piano + bass and
  piano + bass + drums on SAMBA. Async drums MIDI is recorded; bass and horns
  are transcriptions in the derived package.
- **Step 3 follower comparison:** score position with a constant-tempo clock,
  harmony only, downbeat only, and harmony × downbeat as in the paper.
- **Adaptive weighting:** weight each cue by a reliability estimate, such as
  the entropy of the beat posterior, and compare with the fixed product.

### 3. Piano MIDI quality

**Found:**
- SAMBA piano MIDI was recorded from a MIDI piano, not transcribed.
- The sync takes' MIDI is mistimed as distributed. The ticks follow a 175 BPM
  session clock, but each file is stamped with the song's written tempo, and
  the MIDI starts 1 to 21 s off the audio.
- `samba_sync_timing.py` fixes this. It rescales to 175 BPM and fits each
  take's shift from its own piano audio, without annotations.
- On the correctly aligned async takes, the same method finds a shift of 0.
- Async MIDI is aligned as recorded.
- Report the fault to the SAMBA authors.

**Still open:**
- **Drift:** the fix is one global scale and shift per take. Check for drift
  along each take by aligning in windows.
- **Note completeness:** compare MIDI note onsets with onsets in the piano
  audio to find missing or extra notes.
- **Our own transcription:** if the MIDI proves unreliable, transcribe the
  piano from `piano_left/right.flac`. In sync takes the band bleeds into the
  piano mics at only 1 to 10 dB below the piano, which will hurt
  transcription.
- **Live use:** a live system working from audio needs a piano transcription
  model anyway.

### Smaller follow-ups

- **Lay-outs:** mask stretches where the piano lays out when scoring; the
  model gets no input there.
- **Written vs played tempo:** the written-tempo hint can be far from the
  played tempo (On Green Dolphin Street: written 165, played about 189).
- **Sync beats:** sync-take beats are interpolated within annotated bars, so
  they are approximate.

## 2026-10-18: does bounding the tempo fix the tempo spikes? (advisor's suggestion)

**Question:** the tempo maps of the decoded beats spike to 2–3x (two beats fired
0.1–0.3 s apart) and dip where a beat is missed. Would bounding the highest and
lowest tempo remove these, and would that make the beat tracker more accurate?

**Test:** `eval_tempo_bounds.py`, all 152 SAMBA takes, checkpoint lakh_r3b
(`runs/lakh_r3b/best_model.pt`). Its unbounded row reproduces samba_breakdown
exactly (0.532 / 0.337). Decoder variants, all our choice (Maezawa's follower
never decodes beats):
- **min-gap:** drop a beat that comes sooner than period / r after the last one,
  so the tempo can be at most r times the BPM condition. This is causal.
- **min-gap+fill:** also add a beat when none has come for period * r, so every
  interval lies within [period / r, period * r]. An added beat is decided up to
  (r − 1) beats late, so a live system would hear it late.

**Results**, BPM condition = played tempo, mir_eval ±70 ms:

| | beat F | beat CMLt | downbeat F | downbeat CMLt |
|---|---|---|---|---|
| all 152, current decoder | 0.532 | 0.301 | 0.337 | 0.325 |
| all 152, min-gap r=1.5 (causal) | 0.524 | 0.297 | 0.340 | 0.420 |
| all 152, min-gap+fill r=1.5 | 0.549 | 0.380 | 0.343 | 0.517 |
| all 152, min-gap+fill r=1.3 | 0.528 | 0.428 | 0.318 | 0.553 |
| live 52, current decoder | 0.584 | 0.356 | 0.404 | 0.412 |
| live 52, min-gap+fill r=1.5 | 0.606 | 0.434 | 0.410 | 0.583 |
| live 52, written tempo, current decoder | 0.544 | 0.293 | 0.366 | 0.321 |
| live 52, written tempo, min-gap+fill r=1.3 | 0.511 | 0.333 | 0.297 | 0.420 |

**What this shows:**
- **The obvious spikes go away**, by construction. Continuity improves a lot:
  beat CMLt +0.08 to +0.13, downbeat CMLt +0.19 to +0.23.
- **Hit rates barely move.** Beat F is +0.02 at best and downbeat F changes by
  ±0.02. Filled-in beats are extrapolated and often miss the ±70 ms window.
  The causal drop-only version does not raise F at all.
- **The downbeat problem is not a tempo problem.** Bounding cannot tell which
  beat is "1", and in 149 of 152 takes the "1" wanders rather than sitting at a
  fixed offset. Argument 2 (beat tracking alone can't follow a lead sheet) stands.
- **A bound is only as good as the tempo it is centred on.** Around the written
  tempo, tight bounds hurt the live takes (beat F 0.544 → 0.511, downbeat F
  0.366 → 0.297), because written tempos there are often 20–50% off.
- **The follower already has the bound where it matters.** Its tempo grid spans
  0.5–2x, and B7 adds a per-beat pull toward the target tempo (about ±15%). That
  is the change from B3 to B7: bar accuracy 0.68 → 0.85, lost takes 10 → 0.

Figure: `results/tempo_bounds/tempo_bounds_examples.png` (Bernie's Tune and
Misty, the slide examples).
