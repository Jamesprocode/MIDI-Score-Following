"""check_cue's metrics on toy cues, on a real take's chart: a perfect cue scores 1 everywhere, a flat cue scores
chance, a cue that knows the measure but not the chorus gets measure AUC 1 and bar 1 but a place AUC below 1. Seconds.

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_check_cue
"""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2]
SAMBA = EXPERIMENTS.parent / "Dataset" / "SAMBA dataset"
TAKE = "sync/53-black_orpheus/alternate"


class Toy:
    """log-likelihood from the true place: "perfect" (0 there, -10 elsewhere), "flat" (0 everywhere), "measure"
    (0 at the true place in every chorus)."""
    name = "toy"

    def __init__(self, data, mode):
        self.data, self.mode = data, mode

    def log_likelihood(self, now, bpm, positions):
        truth = float(np.interp(now, self.data["times"], self.data["truth_position_beats"]))
        i = int(round(truth * 8))
        if self.mode == "flat":
            return np.zeros(len(positions))
        s = np.full(len(positions), -10.)
        if self.mode == "perfect":
            s[i] = 0.
        else:
            from hmm_follower.check_cue import chart
            _, inside, measure, in_bar = chart(self.data, 8)
            s[inside & (measure == measure[i]) & (np.abs(in_bar - in_bar[i]) < 1e-9)] = 0.
        return s


@unittest.skipUnless(SAMBA.exists(), "needs the SAMBA dataset")
class ToyCues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hmm_follower.takes import load
        cls.data, cls.take = load(SAMBA, ("sync",), only=(TAKE,))[0]

    def check(self, mode):
        from hmm_follower import check_cue, followers
        followers.CUES["toy"] = lambda x: Toy(x.data, mode)
        try:
            r = check_cue.check_take(self.data, self.take, "toy", SAMBA, followers.Resources())
        finally:
            del followers.CUES["toy"]
        f = r["fired"]
        self.assertGreater(f.mean(), .9)
        return {m: float(np.mean(r[m][f])) for m in check_cue.SUMMARY + ("truth_minus_max",)}

    def test_perfect(self):
        got = self.check("perfect")
        for m in ("place_auc", "measure_auc", "top", "bar", "beat"):
            self.assertAlmostEqual(got[m], 1., msg=m)
        self.assertEqual(got["truth_minus_max"], 0.)

    def test_flat(self):
        got = self.check("flat")
        self.assertAlmostEqual(got["place_auc"], .5)
        self.assertAlmostEqual(got["measure_auc"], .5)

    def test_knows_the_measure_not_the_chorus(self):
        got = self.check("measure")
        self.assertAlmostEqual(got["measure_auc"], 1.)
        self.assertAlmostEqual(got["bar"], 1.)
        self.assertLess(got["place_auc"], 1.)
        self.assertGreater(got["place_auc"], .9)


if __name__ == "__main__":
    unittest.main()
