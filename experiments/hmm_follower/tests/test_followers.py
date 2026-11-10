"""Followers as data (followers.py) and the time grids (takes.py): fast, no dataset needed.

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_followers
"""
from __future__ import annotations

from dataclasses import asdict
import unittest

from hmm_follower.followers import FOLLOWERS, parse
from hmm_follower.hmm import HMMConfig


class Followers(unittest.TestCase):
    def test_b5_settings_are_the_old_defaults(self):
        """A named follower's HMM is YamahaConfig()'s (B5), and the update step is run_follow_b's."""
        from hmm_follower import legacy  # noqa: F401
        from run_follow_b import STEP
        from yamaha_filter import YamahaConfig
        from hmm_follower.takes import STEP as NEW_STEP
        self.assertEqual(asdict(HMMConfig()), asdict(HMMConfig.like(YamahaConfig())))
        self.assertEqual(NEW_STEP, STEP)

    def test_old_names(self):
        """The old runners' variants, as named followers: harmony, rhythm, melody in that order."""
        self.assertEqual(FOLLOWERS["B5"].cues, {"harmony": 1., "rhythm": 1.})
        self.assertEqual(list(FOLLOWERS["B5+v2w1"].cues.items()), [("harmony", 1.), ("rhythm", 1.), ("melody_v2", 1.)])
        self.assertEqual(FOLLOWERS["B5+v2"].cues["melody_v2"], .5)
        self.assertEqual(list(FOLLOWERS["v2+rhythm"].cues), ["rhythm", "melody_v2"])

    def test_written_out(self):
        label, spec = parse("mine=harmony@max:0.5,rhythm:2,melody_v2;exact_beat_crossing=true;beat_tempo_reversion=0.02")
        self.assertEqual(label, "mine")
        self.assertEqual(spec.cues, {"harmony": .5, "rhythm": 2., "melody_v2": 1.})
        self.assertEqual(spec.scaling, {"harmony": "max"})
        self.assertTrue(spec.config().exact_beat_crossing)
        self.assertEqual(spec.config().beat_tempo_reversion, .02)
        self.assertEqual(parse("B5")[1].record(), FOLLOWERS["B5"].record())

    def test_mistakes_are_refused(self):
        for text in ("B6", "x=beat_new:1", "x=harmony:1;no_such_setting=1", "x=harmony@zz:1"):
            with self.subTest(text), self.assertRaises(ValueError):
                parse(text)


if __name__ == "__main__":
    unittest.main()
