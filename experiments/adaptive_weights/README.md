# Adaptive cue weighting: how much is there to gain?

The B7 follower (`../yamaha_follower/yamaha_filter.py`) already combines its two
cues as a weighted sum of log-likelihoods:

    log phi(s) = elapsed * (w_h * log H(s) + w_d * log D(s))

with both weights fixed at 1 (Maezawa's product p_h * p_d). The thesis proposes
letting these weights change with the music. This folder measures how much that
could gain, and holds the demo material for the proposal talk.

## Files

- `sweep_fixed_weights.py`: B7 on the 52 live (sync) JazzSAMBA takes for every
  fixed (w_h, w_d) in {0, 0.25, 0.5, 1, 2, 4}^2. For each take it saves the
  predicted bar and position at every 200 ms check, plus the piano's role from
  the section annotations, to `runs/sweep_b7/frames/<take>.npz`. `--chunk`
  splits a take's settings over several workers.
- `analyze_sweep.py`: computes the summary and the figures (`runs/sweep_b7/figures/`)
  from the frame files.
- `weighted_filter.py`: the same filter with per-update weights, the hook for an
  adaptive layer. `tests/test_weighted_filter.py` checks it equals B7 exactly for
  constant weights.
- `demo_player/`: a copy of `../yamaha_follower/player` (unchanged there) with a
  lead-sheet view. It shows the whole form, section letters and chords, where the
  band is and where the follower thinks it is, and a roadmap of the performance.
  `build_demo.py` builds its data.
- `render_demo.py`: renders the same view as a video with the take's band
  recording as audio (the talk's demo clips).

## How the sweep was run (2026-10-18)

- Checkpoint `../yamaha_timing/runs/lakh_r3b/best_model.pt` (sha256 aac88515…, as on ICE).
- 38 takes ran on ICE (`ice-cpu`, 48 cores, run `sweep_b7`, commit 4aa8673). The
  job reached its 150 min wall time; each finished take had already been saved.
- The remaining 14 takes, the shortest, ran locally with the same code.
- Check: the (1, 1) setting reproduces the earlier B7 per-take bar accuracy
  (ICE run `follow_b67`) exactly on every take.

## Results (52 live takes, share of 200 ms checks in the correct bar)

| Follower | Bar accuracy | Lost takes |
|---|---|---|
| no evidence (tempo only) | 0.19 | |
| rhythm only (0, 1) | 0.61 | 12 |
| harmony only (1, 0) | 0.76 | 0 |
| both, Maezawa (1, 1) = B7 | 0.852 | 0 |
| best single fixed setting (0.5, 0.5) | 0.858 | 0 |
| best setting per take (oracle) | 0.882 | |

- **Which cue is more reliable depends on the song.** Rhythm alone beats harmony
  alone in 18 takes, and harmony alone wins in 34.
  - When the rhythm cue works, it is very good (about 0.9–0.97).
  - When it fails, harmony carries the follower (`sweep_rhythm_vs_harmony.pdf`).
- **The best balance varies:** 17 distinct best settings over 52 takes.
  - It favours harmony in 14 takes, rhythm in 18, and is even in 20.
- **Choosing per take helps a few songs a lot, most songs little.** The median
  gain over (1, 1) is +0.016 and the mean +0.030. 11 takes gain more than 0.05,
  and 3 gain more than 0.10, up to +0.17 (Don't Fence Me In, alternate).
  - This is an optimistic upper bound: the best of 36 runs, picked after seeing
    the result.
- **Averaged by passage type, the best balances are close to even:**
  - head: (1, 0.5);
  - piano solo and comping: (0.5, 0.5);
  - piano laid out: (0.25, 0.25), the lowest, meaning trust the tempo model.
  The per-song differences cancel out in these averages. For example:
  - in Bye Bye Blackbird's piano solo, harmony is right and rhythm misleads;
  - in Almost Like Being in Love's piano solo, it is the reverse.
  So weights would have to follow the actual evidence, not just the passage label.
- **The overall scale matters, not only the ratio.** Many takes do best at
  (0.25, 0.25) or (0.5, 0.5), i.e. evidence counts less and the tempo model
  more. Weights should not be forced to sum to one.

**The two talk examples:**
- **La Vie en Rose (preferred):** about 0.97 with any setting; rhythm alone 0.975.
- **Bye Bye Blackbird (preferred):** (1, 1) 0.72; rhythm alone 0.12; harmony alone
  0.74; best (0.5, 0.25) 0.82.
  - In the piano solo (172–200 s) the fixed follower falls 3 bars behind while
    harmony alone stays in the right bar.

**Not yet measured:** weights that change within a take. The per-take oracle keeps
one setting per song, so it says nothing about moment-to-moment adaptation.
`weighted_filter.py` is ready for that test (a causal rule, then a learned
layer).
