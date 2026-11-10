# Code review 2026-11-09: making the follower modular

**Goal** (after the 11-05 meeting with Alex): swapping in a new harmony module or a new beat tracker should take one new file and two commands, one measuring the module alone and one measuring it inside the follower. This review covers the code a B5 + melody run actually uses. Nothing was changed.

## 1. How a cue reaches the follower today

Every runner (`run_melody_follower.py`, `run_melody_heads.py`, `run_jiang_resets.py`, `sweep_fixed_weights.py`, ...) repeats the same steps for each take:

1. **Load the take:** `follower_data.samba_follow_takes` gives a `data` dict (chord plan, truth) and a `take` object (piano notes). Horns and bass come from `SAMBA_CLEANED_MIDI`, an environment variable read in `yamaha_timing/real_data.py:110`.
2. **Build three time grids:** the data grid (100 ms), the filter grid (`STEP` = 50 ms, from `run_follow_b.py`) and the evaluation grid (200 ms).
3. **Harmony:** `run_study.get_features` (in `ensemble_benchmark/improvement_study/`) matches the piano roll against chord templates. `run_study.observation` then calls `run_benchmark.score_probabilities` (z-score, then softmax) to get chord-class probabilities per frame.
4. **Rhythm:** `NetworkPhase(sweep_fixed_weights._MODEL, take)`. The model is a module global loaded by `_init`.
5. **Melody:** `MelodyEvidence` or `NoteHMMEvidence(data, root, config)`.
6. **Follow:** `follow_weighted(...)` with 13 positional arguments.
7. **Save and score:** save frames, then summarise with one of several scripts.

**The three cues enter the filter at three different levels:**

| Cue | What the filter receives | Who turns it into a score per place | Scaling |
|---|---|---|---|
| Harmony | a frames × chord-class matrix, plus a chord id for each chord of the plan | the filter loop (`weighted_filter.py:92-97`) | z-score softmax (`run_benchmark.py:107`), a 2% floor (`probability_floor_mix`), minus the max |
| Rhythm | an object with `frame_at` and `bar_log_probs(frame, bpm)` over 32 bar-phase classes | the filter loop (`weighted_filter.py:98-102`) | the network's log-softmax, minus the max |
| Melody | an object with `log_likelihood(now, bpm, n_places)` | the cue itself | v1: log(floor + share); v2: mean log-likelihood per note; minus the max |

Only melody already gives a score per place. Harmony and rhythm need code inside the filter loop.

## 2. Findings, ordered by how much they slow down a swap

1. **The filter loop is copied.**
   - There are three live copies: `yamaha_filter.follow`, `weighted_filter.follow_weighted` and `run_jiang_resets.follow_with_resets`. Three older copies sit in `ensemble_benchmark/`.
   - **They have already drifted.** The tempo-drift fix (`exact_beat_crossing`, still uncommitted in `yamaha_filter.py` and `weighted_filter.py`) is not in `run_jiang_resets.py:113`. Turning it on would quietly skip the reset-ratio runs.
2. **Harmony and rhythm are wired into the loop.**
   - A new harmony model must output probabilities over `make_targets`' chord vocabulary (`run_benchmark.py:52`), in its exact column order.
   - The harmony check (`harmony_cue/chord_rank.py`) uses a different cue shape: (root, quality) keys. A model measured there can't be dropped into the follower without glue code.
   - A new beat tracker must imitate `NetworkPhase`'s two methods and give bar-phase classes. A tracker that only finds beats, with no downbeat, doesn't fit at all.
3. **The per-take setup is copied** in 7 scripts and 1 test: harmony features + z-score, the network, the melody cue and the time grids. These are `check_melody_evidence`, `run_melody_heads`, `run_jiang_resets`, `run_melody_follower`, `run_follow_b`, `sweep_fixed_weights`, `run_tempo_variants` and `tests/test_weighted_filter`.
4. **Scaling is spread over four places:**
   - the z-score in the benchmark code;
   - the 2% floor in `YamahaConfig`;
   - "minus the max" in the loop;
   - v1's floor in `MelodyConfig`.

   Alex's point (log probability, z-score vs bounded probability) would need edits in all four.
5. **The runner code is copied.**
   - argparse, sharding, resume, the process pool and the frame file appear in 4 runners.
   - Each runner has its own `VARIANTS` table with a different tuple shape: `(w_m, config)`, `((w_h, w_d, w_m), cue)`, or a weight grid.
6. **Section labels are defined three ways:**
   - `melody_evidence.section_kinds`: 4 kinds;
   - `chord_rank.section_kinds`: 8 kinds;
   - `sweep_fixed_weights.piano_roles`: 5 roles.
7. **Metrics are defined in four places:** `run_follow.position_metrics`, `metrics_from_frames.take_metrics`, `summarize_bar_beat.per_take` and `jiang_metrics`. They agree today (bar = same bar index, beat = same whole beat), but a new metric has to be added four times.
8. **Some configuration is hidden:**
   - the horn set comes from an environment variable;
   - `NoteHMMEvidence` picks its fitted file from the take's protocol (`note_hmm.py:313`);
   - the network sits in a module global.

   The melody runners write none of this into the run folder, so a run doesn't record which horn set, fit or commit it used.
9. **Imports work through `sys.path` side effects** (`import follower_data  # sets sys.path`). Private helpers (`_advance`, `_walk`, `_init`, `_MODEL`) are imported across folders, so moving one file can quietly break others.

**What to keep:**
- `follow_weighted` already takes weights per update. That is the hook for the confidence rule.
- The melody interface is the right shape: `log_likelihood(now, bpm, n_places)` returns one number per place, or None for "no opinion".
- `NetworkPhase` runs the backbone once and reapplies the tempo gate per frame. That is fast and stays.
- Tests already check filter equivalence (`tests/test_weighted_filter.py`) and beat crossing (`tests/test_beat_crossing.py`). They are the base for the golden tests below.

## 3. Proposed design

A new package, `experiments/follower/`. No existing file is edited. Old scripts keep working, and the new code is proven equal to the old on golden takes before anything uses it.

**Every cue is the same kind of object:**

```python
class Cue(Protocol):
    name: str
    def log_likelihood(self, now: float, bpm: float, grid: PlaceGrid) -> np.ndarray | None:
        """One number per place: log P(what this cue heard recently | the band is at that place). None = no opinion."""
```

**Files:**

| File | What it holds |
|---|---|
| `data.py` | `Take`: plan, piano, horns, bass, sections and split. The horn set is an explicit argument. One `section_kinds` with the 8 kinds; the coarser labels are derived from it. |
| `grid.py` | `PlaceGrid`: the places, and each place's measure, chord and bar phase. Also the time grids (filter 50 ms, evaluation 200 ms). |
| `cues/harmony.py` | `ChordCue(model)`: any model that gives chord-class probabilities per frame (today's template matcher, a future small model). The chord → place mapping is done once, here. |
| `cues/rhythm.py` | `PhaseCue(tracker)`: any tracker that gives bar-phase or beat-phase log-probabilities per frame for a tempo hint. `NetworkPhase` is one such tracker. |
| `cues/melody.py` | Adapters for v1 and v2, which already have the right shape. The fitted file is passed in explicitly. |
| `scaling.py` | How a cue's output becomes evidence. Today's rules are reproduced exactly (harmony z-score softmax + 2% floor, minus the max). **The probability redesign happens here, in one file.** |
| `filter.py` | One loop: the tempo model + Σ over cues of w_c(t) · elapsed · scaled cue. Options: start place, per-update weights, `exact_beat_crossing`, and a reset hook (Jiang). It must equal `follow`, `follow_weighted` and `follow_with_resets`. |
| `followers.py` | Named followers as data, e.g. `"B5": {"harmony_b5": 1, "rhythm_r3b": 1}` and `"B5+melodic": {..., "melody_v2": 1}`. Cue factories by name; each loads its own resources, cached per worker. |
| `run.py` | One runner: modes whole, heads and resets; `--split`, `--shard`, resume. It writes one frame format plus `run.json` (commit, followers, cue configs, horn set, fit files). |
| `metrics.py`, `compare.py` | Bar, beat, off, lost and ms; live / click / pooled; per-song Wilcoxon. |
| `check_cue.py` | Any cue on its own: place AUC, bar hit and beat hit by section kind. It replaces both `check_melody_evidence.py` and `chord_rank.py`. |
| `tests/` | Golden equality with the old code on a few takes, plus a toy perfect-place cue. |

**After this, adding a harmony model or a beat tracker:**
1. Write one class (chord probabilities, or phase log-probabilities, per frame).
2. Add one line to `followers.py`.
3. `check_cue.py --cue harmony_new` measures it on its own.
4. `run.py --followers B5 B5:harmony=harmony_new` measures it inside the follower against B5.

## 4. Build order (each step checked before the next)

1. **`grid`, `scaling`, `filter`, and B5's two cues as `Cue` objects.** Golden test against `follow` and `follow_weighted` on 3 takes. Light; runs on the Mac.
   - **Done 11-09** as `experiments/hmm_follower/`:
     - `hmm.py`: the follower, as a `Follower` class that takes one update at a time, plus `follow()`;
     - `cues/harmony.py`, `cues/rhythm.py`, `cues/melody.py`: the three plug-ins;
     - `scaling.py`, and `legacy.py` (the one door into the old folders).
   - **The time grids stay in the runner, for step 2.**
   - **`tests/test_same_as_old.py` passes (5 tests, 112 s).** The new follower gives the same output as the old code, bit for bit, on 30 s of 3 takes (live 3/4, live 4/4, click), for:
     - B5;
     - B7 weights, including one cue off;
     - the continuous tempo model;
     - `exact_beat_crossing`;
     - B5 + melody v2;
     - weights that change at every update;
     - a start place 20 s in;
     - Jiang resets.
2. **Melody adapters, `followers.py`, `run.py` (whole and heads).** Rerun the 32 split test takes on PACE and check that the frames match the saved `runs/split_heads_test` and `runs/split_whole_test` exactly.
   - **Built 11-10:**
     - `followers.py`: named followers as data (cue: weight, in the order the evidence is added, plus HMM settings and a scaling rule per cue), and `CUES`, one factory per cue. Followers can also be written on the command line: `label=cue:weight,...;setting=value`.
     - `takes.py`: the take loader, the horn set (`use_horns`, an explicit argument instead of the hidden environment variable), and the time grids in one place.
     - `run.py`: one runner, `--mode whole|heads`, `--split`, `--shard`, `--takes`, resume.
       - The fitted melody tables (`--melody-params`), the network and the lead sheets are explicit arguments.
       - `run.json` records the followers, the files, the horn set and the commit, and resuming with other settings is refused.
       - Each frame file adds `info`, the fitted file melody v2 used for that take.
     - `same_frames.py`: compares two runs bit for bit.
     - `tests/test_followers.py`: the names, the parsing, and that B5's settings are the old defaults.
   - **Mac check, against the frames PACE saved on 11-05:** on the shortest test take (click, Impressions, alternate), bit-identical in both modes:
     - heads: all 7 followers;
     - whole take: B5, and B5+v2w1 (the old "B5+H1").
   - **Done 11-10: the PACE rerun of all 32 split test takes is bit-identical to the saved runs.**
     - **Code:** commit `6bb0508`, with the same settings as 11-05.
     - **Jobs:** `hft_s0/1` (heads) and `wft_s0..2` (whole), each about 10–20 minutes on `ice-cpu`.
     - **Comparison** with `same_frames.py`:
       - heads: 32 of 32 takes identical to `runs/split_heads_test`, all 7 followers;
       - whole take: 32 of 32 identical to `runs/split_whole_test`, B5 and B5+v2w1.
     - **Saved** as `melody_evidence/runs/hmmf_split_{heads,whole}_test`, with each shard's `run.json`.
   - **From now on,** new follower runs use `python -m hmm_follower.run`. The old runners stay as the record of the runs they made.
3. **Resets mode, `metrics`, `compare`, `check_cue`.**
   - **Built 11-10** (the resets mode is not yet built):
     - **`sections.py`: one labelling of what the band is doing.** It gives the 8 kinds, the 4 coarse kinds and the piano's role, and equals all three old labellings on all 152 takes (`tests/test_sections.py`).
     - **`metrics.py`: one definition of the follower metrics, with `compare` folded in.**
       - It gives bar, beat, off, ms and lost; live, click and all; per-song Wilcoxon tests; and `--section` to keep one kind of section.
       - It equals `summarize_bar_beat`, `jiang_metrics.frame_metrics` and `metrics_from_frames` on every take of the saved split test runs (`tests/test_metrics.py`).
       - On `split_heads_test` it prints the same table as `summarize_split_heads.py`.
     - **`check_cue.py`: any cue in `followers.CUES` on its own,** at every 0.2 s check, against every place of the chart, with the annotated tempo as the hint.
       - Metrics: place AUC (as `chord_rank.py`, over grid places), measure AUC, top, bar hit and beat hit (as `check_melody_evidence.py`).
       - For Alex's scaling work it also saves the cue's raw numbers: s at the true place minus the max, as a log-softmax and as a z-score, and s at 8 random places in other measures.
       - **Checked against `chord_rank.py`** on live Black Orpheus (alternate), harmony: 928 of 934 checks give the same place AUC (0.688 vs 0.684 overall). The other 6 have the true place within 1/16 beat before a chord change, where rounding to the 1/8-beat grid lands in the next chord.
       - **One difference from `check_melody_evidence.py`:** checks where the tune does not cover the true bar are no longer skipped. A melody cue firing in an intro is therefore counted, and scored as wrong.
       - **Toy cues** (`tests/test_check_cue.py`): a perfect cue gives 1, a flat cue 0.5, and a cue that knows the measure but not the chorus gives measure AUC 1 with place AUC below 1.
4. **Then Alex's probability step:**
   - plot each cue's output distribution with `check_cue` (at the true place vs elsewhere, by section);
   - redesign `scaling.py` (log probabilities; z-score vs bounded);
   - run the followers.

## 5. Open questions

- **Harmony interface.** Proposal: a chord-probability adapter (`ChordCue`), so chord recognisers plug in without knowing about places. A module can still implement `Cue` directly.
- **`exact_beat_crossing` (tempo-drift fix).** It goes into the new filter as an option, off by default, so the default stays B5. The tempo-drift session's files are not touched.
- **Old scripts.** They are frozen as the record of past results, and new runs use the package. `ensemble_benchmark/` stays as it is.
