"""metrics.py gives the old scripts' numbers on saved runs: summarize_bar_beat.per_take (bar, beat, and the heads of a
whole-take run), jiang_metrics.frame_metrics (off) and metrics_from_frames.take_metrics (ms, lost). Reads the
split test runs (32 takes each) and loads the takes for the note onsets: under a minute.

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_metrics
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2]
SAMBA = EXPERIMENTS.parent / "Dataset" / "SAMBA dataset"
HORNS = EXPERIMENTS.parent / "Dataset" / "horn_sets" / "crepe_all"
RUNS = EXPERIMENTS / "melody_evidence" / "runs"
HEADS, WHOLE = RUNS / "split_heads_test", RUNS / "split_whole_test"


@unittest.skipUnless(SAMBA.exists() and HORNS.exists() and HEADS.exists() and WHOLE.exists(),
                     "needs SAMBA, the horn set and the saved split test runs")
class SameNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SAMBA_CLEANED_MIDI", str(HORNS))  # as the runs (before real_data is imported)
        from hmm_follower import legacy  # noqa: F401
        from hmm_follower.takes import load
        cls.takes = {d["performance_id"]: d for d, _ in load(SAMBA)}

    def test_bar_beat_off(self):
        from hmm_follower.metrics import per_take
        from summarize_bar_beat import per_take as old_bar_beat
        import jiang_metrics
        for run in (HEADS, WHOLE):
            new = per_take(run, SAMBA, takes=self.takes)
            old, _ = jiang_metrics.frame_metrics(run)
            bars = old_bar_beat(run)
            for (p, s, t), followers in new.items():
                for v, m in followers.items():
                    with self.subTest(run=run.name, take=f"{p}/{s}/{t}", follower=v):
                        self.assertEqual((m["bar"], m["beat"]), bars[p][v][(s, t)])
                        self.assertEqual(m["off"], old[p][v][(s, t)]["off"])

    def test_ms_and_lost(self):
        sys.path.insert(0, str(EXPERIMENTS / "adaptive_weights"))
        from metrics_from_frames import take_metrics as old
        from hmm_follower.metrics import per_take
        new = per_take(WHOLE, SAMBA, notes=False)
        for f in sorted((WHOLE / "frames").glob("*.npz")):
            z = np.load(f)
            p, s, t = f.stem.split("__")
            for k, v in enumerate(str(x) for x in z["variants"]):
                expected = old(z["times"], z["truth_bar"], z["truth_position"], z["predicted_bar"][k],
                               z["predicted_position"][k])
                with self.subTest(take=f.stem, follower=v):
                    for m in ("bar", "beat", "ms", "lost"):
                        self.assertEqual(new[(p, s, t)][v][m], expected[m], m)

    def test_heads_of_whole_takes(self):
        from hmm_follower.metrics import per_take
        from summarize_bar_beat import per_take as old_bar_beat
        new = per_take(WHOLE, SAMBA, section="head", notes=False)
        old = old_bar_beat(WHOLE, "head")
        for (p, s, t), followers in new.items():
            for v, m in followers.items():
                self.assertEqual((m["bar"], m["beat"]), old[p][v][(s, t)])
        self.assertEqual(len(new), sum(len(old[p]["B5"]) for p in old))


if __name__ == "__main__":
    unittest.main()
