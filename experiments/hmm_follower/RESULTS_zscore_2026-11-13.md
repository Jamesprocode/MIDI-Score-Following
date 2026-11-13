# Per-cue z-score in the follower (2026-11-13)

James's question: "do the mean and standard deviation on the 60 training songs, use them to z-score normalize the test
scores, and see if we have any improvement."

## What was run

**The rule** (`scaling.py`, fitted by `fit_scaling.py` into `params/zscore_train.json`):
- **Fit.** Each cue gets one mean and one std, pooled over every score it gave on the 60 training songs (120 takes).
  - The pool is every chart place at every 0.2 s check where the cue spoke, with each check counting the same ("pool B", 11-11).
  - The numbers come from the saved check_cue runs (`melody_evidence/runs/hmmf_check_*`, commit 3b89580).
  - They are frozen, then used unchanged on the 16 test songs (32 takes).
- **What the follower actually changes.** The follower renormalizes after every update, so the mean (and the moment's max) cancels. Each z rule is exactly the cue's "max" evidence times a fixed weight of 1/std.
  - Checked: a unit test, plus 30 s of a real take where z equals "max" with weight 1/std to 6e-16.

| Cue | z: std of (s − best) | zraw: std of raw s | zw: within-moment std |
|---|---|---|---|
| Harmony | 1.099 → weight 0.91 | 1.047 → 0.96 | 0.953 → 1.05 |
| Rhythm | 1.769 → 0.57 | 1.315 → 0.76 | 1.118 → 0.89 |
| Melody v2 | 0.925 → 1.08 | 0.453 → 2.21 | 0.358 → 2.79 |

**Labelled choices (ours):**
- **zw comes from the 11-13 code review.** z's and zraw's std also count how much a cue's level wanders from moment to moment, and the follower never sees that. zw keeps only the spread across places at one moment. Under zw, every cue's evidence has the same typical spread inside the follower.
- **The scores were made with the true tempo.** check_cue gave each cue the annotated tempo as its BPM hint. In a follower, the cue gets the follower's own tempo.
- **Melody's training scores are in-sample.** Its fitted tables (crepe_split) were fit on the same training takes.

**Followers** (test split, 32 takes, `--mode whole` and `--mode heads`):

| Follower | Harmony | Rhythm | Melody | Rule | Total evidence* |
|---|---|---|---|---|---|
| B5 | 1 | 1 | – | max | 2.07 |
| B5z | 1 | 1 | – | z | 1.50 |
| B5+v2 | 1 | 1 | 0.5 | max | 2.25 |
| B5+v2w1 | 1 | 1 | 1 | max | 2.43 |
| B5+v2z | 1 | 1 | 1 | z | 1.89 |
| B5+v2zraw | 1 | 1 | 1 | zraw | 2.55 |
| B5+v2zw | 1 | 1 | 1 | zw | 3.00 |
| B5+v2s | 0.838 | 0.838 | 0.419 | max | 1.89 (control) |

\*Total evidence = the sum over cues of the effective weight × the within-moment spread. It says how strongly all the evidence together pushes against the tempo model.

- **B5+v2s is the control.** It is B5+v2 turned down uniformly to B5+v2z's total. An older sweep (`adaptive_weights/README.md`, B7) found that turning all evidence down changes accuracy on its own. So a gain for B5+v2z only counts as a z-score effect if B5+v2z also beats B5+v2s.

**Fixed before the runs:**
- **The primary test** is B5+v2z vs B5+v2, whole-take mode, bar accuracy, all 16 test songs, paired Wilcoxon per song (`metrics.py`). Every other comparison is exploratory.
- **The live group has only 6 songs.** Its smallest possible p is 0.031, so a live-only "significant" result is out of reach by design.

## Results

(pending: PACE runs)
