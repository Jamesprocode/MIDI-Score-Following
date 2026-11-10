"""The HMM follower with harmony, rhythm and melody as plug-in cues (CODE_REVIEW_2026-11-09.md).

  hmm.py          the follower: (tempo, place) HMM, one update at a time; any list of cues, each with a weight
  cues/           the plug-ins, one per file; each gives log_likelihood(now, bpm, positions)
  scaling.py      how a cue's log-likelihood becomes evidence (one rule per cue)
  followers.py    named followers as data (cues, weights, HMM settings) and the cue factories (CUES)
  takes.py        the takes, the horn set, and the time grids (update 50 ms, check 200 ms)
  run.py          one runner: --mode whole | heads, any followers; frames + run.json
  metrics.py      bar, beat, off, ms, lost from any run's frames; live / click / all; per-song paired tests
  check_cue.py    any one cue on its own: place AUC, measure AUC, bar and beat hit, and its raw numbers, by section
  sections.py     what the band is doing at each instant (8 kinds, 4 coarse kinds, the piano's role)
  same_frames.py  are two runs' frames identical?
  legacy.py       the one door into the older experiment folders (read-only)

Run from experiments/:

  yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_followers hmm_follower.tests.test_sections \
      hmm_follower.tests.test_check_cue                                              # seconds
  yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_metrics        # a minute
  yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_same_as_old    # about two minutes
  yamaha_timing/.venv/bin/python -m hmm_follower.run --help
"""
