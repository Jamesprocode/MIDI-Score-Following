"""follow_weighted with constant weights must equal yamaha_filter.follow exactly (one short SAMBA take).

Needs the local SAMBA data and the lakh_r3b checkpoint; skipped if either is missing.

  ../yamaha_timing/.venv/bin/python -m unittest discover -s tests
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "yamaha_follower"))
SAMBA = HERE.parent.parent / "Dataset" / "SAMBA dataset"
CHECKPOINT = HERE.parent / "yamaha_timing" / "runs" / "lakh_r3b" / "best_model.pt"
TAKE = "sync/51-amazing_grace/preferred"


@unittest.skipUnless(SAMBA.exists() and CHECKPOINT.exists(), "needs SAMBA data and the checkpoint")
class WeightedFilterTests(unittest.TestCase):
    def test_constant_weights_match_follow(self):
        import follower_data
        import run_study
        from evidence import NetworkPhase
        from run_follow_b import STEP, played_bpm
        from yamaha_filter import YamahaConfig, follow
        from yamaha_timing.training import load_checkpoint
        from weighted_filter import follow_weighted

        data, take = next(t for t in follower_data.samba_follow_takes(SAMBA, protocols=("sync",))
                          if t[0]["performance_id"] == TAKE)
        model = load_checkpoint(CHECKPOINT, "cpu")[0]
        times = np.round(np.arange(data["times"][0], data["times"][-1] + 1e-9, STEP), 6)[:1200]
        case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
        scores, available, mass = case["features"]["active_pooled"]
        probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
        network = NetworkPhase(model, take)
        base = YamahaConfig(beat_tempo_reversion=0.02)
        args = (times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], played_bpm(data))
        for w_h, w_d in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.5, 2.0)]:
            with self.subTest(w_h=w_h, w_d=w_d):
                expected = follow(*args, start_time=data["start_time"], network=network if w_d else None,
                                  config=replace(base, emission_power_per_second=w_h, downbeat_power_per_second=w_d))
                got = follow_weighted(*args, np.tile([w_h, w_d], (len(times), 1)), start_time=data["start_time"],
                                      network=network, config=base)
                for key in ("position_beats", "tempo_bpm", "position_confidence", "bpm_condition", "bar_index"):
                    np.testing.assert_array_equal(got[key], expected[key])


if __name__ == "__main__":
    unittest.main()
