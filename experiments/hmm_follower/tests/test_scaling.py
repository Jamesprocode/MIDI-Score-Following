"""The per-cue z-score (scaling.py "z", "zraw"): fast, no dataset needed (the last test reads the check_cue runs and
the committed fit where they exist).

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_scaling
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hmm_follower.followers import FOLLOWERS, needs, parse
from hmm_follower.hmm import Follower, Plugged
from hmm_follower.scaling import ZScore, resolve

EXPERIMENTS = Path(__file__).resolve().parents[2]


class Toy:
    """A cue with made-up log-likelihoods that change with time; silent now and then."""

    def __init__(self, seed, scale, silent_every=0):
        self.seed, self.scale, self.silent_every = seed, scale, silent_every

    def log_likelihood(self, now, bpm, positions):
        k = int(round(now * 20))
        if self.silent_every and k % self.silent_every == 0:
            return None
        rng = np.random.default_rng(self.seed * 100003 + k)
        return self.scale * rng.normal(size=len(positions)) - 3.


def run(plugged, steps=200):
    follower = Follower(plugged, 64., 120., start_time=0.)
    out = [follower.step(k * .05) for k in range(1, steps + 1)]
    return follower.posterior, out


class Scaling(unittest.TestCase):
    def test_z_is_a_weight_of_one_over_std(self):
        """The follower renormalises every update, so (s - max - mean)/std and (s - mean)/std follow exactly as the
        "max" rule with weight 1/std."""
        cues = [Toy(1, 1.), Toy(2, 2.5, silent_every=7), Toy(3, .4)]
        z = [Plugged(cues[0], 1., ZScore(-1.5, 1.1, "max")), Plugged(cues[1], 1., ZScore(-4.2, 1.3, "raw")),
             Plugged(cues[2], 2., ZScore(0., .36, "within"))]
        weights = [Plugged(cues[0], 1 / 1.1), Plugged(cues[1], 1 / 1.3), Plugged(cues[2], 2 / .36)]
        (pz, ez), (pw, ew) = run(z), run(weights)
        np.testing.assert_allclose(pz, pw, rtol=1e-9, atol=1e-300)
        self.assertEqual([e["position_beats"] for e in ez], [e["position_beats"] for e in ew])

    def test_max_rule_unchanged(self):
        """A rule name and the resolved "max" give the same follower, bit for bit."""
        cue = Toy(4, 1.)
        self.assertEqual(resolve("max", "harmony"), "max")
        (a, _), (b, _) = run([Plugged(cue, 1., "max")]), run([Plugged(cue, 1.)])
        np.testing.assert_array_equal(a, b)

    def test_resolve_reads_the_cue_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fit.json"
            path.write_text(json.dumps({"cues": {"rhythm": {"max": {"mean": -2.5, "std": 1.8},
                                                            "raw": {"mean": -4.2, "std": 1.3},
                                                            "within": {"mean": 0., "std": 1.1}}}}))
            z = resolve("z", "rhythm", str(path))
            self.assertEqual((z.mean, z.std, z.base), (-2.5, 1.8, "max"))
            zraw = resolve("zraw", "rhythm", str(path))
            self.assertEqual((zraw.mean, zraw.std, zraw.base), (-4.2, 1.3, "raw"))
            s = np.array([-1., -3., -2.])
            np.testing.assert_allclose(z(s), (s + 1. + 2.5) / 1.8)
            np.testing.assert_allclose(zraw(s), (s + 4.2) / 1.3)
            np.testing.assert_allclose(resolve("zw", "rhythm", str(path))(s), (s + 2.) / 1.1)
            for rule, cue, params in (("z", "melody_v2", str(path)), ("z", "rhythm", ""), ("zz", "rhythm", "")):
                with self.subTest(rule=rule, cue=cue), self.assertRaises(ValueError):
                    resolve(rule, cue, params)

    def test_followers_name_the_rule(self):
        self.assertEqual(FOLLOWERS["B5+v2z"].cues, {"harmony": 1., "rhythm": 1., "melody_v2": 1.})
        self.assertEqual(set(FOLLOWERS["B5+v2zraw"].scaling.values()), {"zraw"})
        self.assertEqual(set(FOLLOWERS["B5+v2zw"].scaling.values()), {"zw"})
        self.assertEqual(FOLLOWERS["B5+v2s"].scaling, {})
        self.assertIn("scaling_params", needs([FOLLOWERS["B5z"]]))
        self.assertNotIn("scaling_params", needs([FOLLOWERS["B5"], FOLLOWERS["B5+v2w1"]]))
        label, spec = parse("x=harmony@z:1,rhythm:1")
        self.assertEqual(spec.record()["scaling"], {"harmony": "z", "rhythm": "max"})

    @unittest.skipUnless((EXPERIMENTS / "melody_evidence/runs/hmmf_check_rhythm/frames").exists(),
                         "needs the check_cue runs")
    def test_committed_fit_is_the_train_pool(self):
        """params/zscore_train.json is fit_scaling's output on the training takes of the three check_cue runs."""
        from hmm_follower.fit_scaling import fit
        runs = {c: str(EXPERIMENTS / f"melody_evidence/runs/hmmf_check_{c}") for c in ("harmony", "rhythm", "melody_v2")}
        committed = json.loads((EXPERIMENTS / "hmm_follower/params/zscore_train.json").read_text())
        self.assertEqual(committed["split"], "train")
        again = fit(runs, "train")
        for cue in runs:
            self.assertEqual(again["sources"][cue]["takes"], 120)
            for base in ("max", "raw", "within"):
                for key in ("mean", "std"):
                    self.assertAlmostEqual(again["cues"][cue][base][key], committed["cues"][cue][base][key], 12)
            numbers = committed["cues"][cue]  # the within-moment variance is the same part of both bases' variance
            for base in ("max", "raw"):
                self.assertAlmostEqual(numbers[base]["std"] ** 2 * numbers[base]["within_moment_share"],
                                       numbers["within"]["std"] ** 2, 12)


if __name__ == "__main__":
    unittest.main()
