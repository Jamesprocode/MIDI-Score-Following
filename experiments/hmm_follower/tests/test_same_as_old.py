"""The new follower (hmm.py with the three plug-in cues) gives exactly the old followers' output, bit for bit.

The old followers are yamaha_filter.follow (B5, B7, the tempo models), weighted_filter.follow_weighted (per-update
weights, melody, a start place) and run_jiang_resets.follow_with_resets (hard resets). The new cues are built by their
own factories (b5_harmony, network_downbeat, melody_v2), so the cue code is checked too. Each case follows 600
updates (30 s): from the top of a take, or from 20 s in for the start-place case. Three takes: live Amazing Grace
(3/4), live Black Orpheus, click Au Privave. Needs the local SAMBA data, the CREPE horn set and the lakh_r3b
checkpoint; skipped if any is missing. About a minute on a laptop.

  cd experiments && yamaha_timing/.venv/bin/python -m unittest hmm_follower.tests.test_same_as_old
"""
from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2]
SAMBA = EXPERIMENTS.parent / "Dataset" / "SAMBA dataset"
HORNS = EXPERIMENTS.parent / "Dataset" / "horn_sets" / "crepe_all"
CHECKPOINT = EXPERIMENTS / "yamaha_timing" / "runs" / "lakh_r3b" / "best_model.pt"
PARAMS = EXPERIMENTS / "melody_evidence" / "params" / "crepe_split"
TAKES = ("sync/51-amazing_grace/preferred", "sync/53-black_orpheus/alternate", "async/05-au_privave/preferred")
N = 600
KEYS = ("position_beats", "tempo_bpm", "mean_tempo_bpm", "bpm_condition", "position_confidence", "bar_index")


@unittest.skipUnless(SAMBA.exists() and HORNS.exists() and CHECKPOINT.exists(), "needs SAMBA, the horn set, the model")
class SameAsOld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SAMBA_CLEANED_MIDI", str(HORNS))  # before real_data is imported, as in the runs
        from hmm_follower import legacy  # noqa: F401
        sys.path.insert(0, str(EXPERIMENTS / "adaptive_weights"))
        import follower_data
        from yamaha_timing.training import load_checkpoint
        cls.model = load_checkpoint(CHECKPOINT, "cpu")[0]
        cls.takes = {d["performance_id"]: (d, t) for d, t in follower_data.samba_follow_takes(SAMBA, ("sync", "async"))
                     if d["performance_id"] in TAKES}

    def setup(self, performance_id, n=N):
        """The old followers' inputs and the new cues for the first ``n`` updates of a take."""
        import run_study
        from evidence import NetworkPhase
        from note_hmm import NoteHMMConfig, NoteHMMEvidence
        from run_follow_b import STEP, played_bpm
        from hmm_follower.cues.harmony import b5_harmony
        from hmm_follower.cues.melody import melody_v2
        from hmm_follower.cues.rhythm import network_downbeat
        data, take = self.takes[performance_id]
        times = np.round(np.arange(data["times"][0], data["times"][-1] + 1e-9, STEP), 6)[:n]
        case = run_study.get_features(dict(data, times=times), {"triad"})["triad"]
        scores, available, mass = case["features"]["active_pooled"]
        probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
        old = {"args": (times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                        case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], played_bpm(data)),
               "network": NetworkPhase(self.model, take),
               "melody": NoteHMMEvidence(data, SAMBA, NoteHMMConfig(params=str(PARAMS)))}
        new = {"cues": [b5_harmony(data, times), network_downbeat(self.model, take, data["beats_per_bar"]),
                        melody_v2(data, SAMBA, PARAMS)],
               "plan": (data["chord_ends_beats"][-1], data["bar_starts_beats"], played_bpm(data))}
        return data, times, old, new

    def same(self, got, expected, label):
        for key in KEYS:
            np.testing.assert_array_equal(got[key], expected[key], err_msg=f"{label}: {key}")

    def test_b5_on_three_takes(self):
        from hmm_follower.hmm import HMMConfig, follow
        from yamaha_filter import YamahaConfig, follow as old_follow
        for performance_id in TAKES:
            data, times, old, new = self.setup(performance_id)
            with self.subTest(performance_id):
                expected = old_follow(*old["args"], start_time=data["start_time"], network=old["network"],
                                      config=YamahaConfig())
                got = follow(times, [(c, w) for c, w in zip(new["cues"], (1., 1., 0.))], *new["plan"],
                             start_time=data["start_time"], config=HMMConfig.like(YamahaConfig()))
                self.same(got, expected, performance_id)

    def test_weights_and_tempo_models(self):
        from hmm_follower.hmm import HMMConfig, follow
        from yamaha_filter import YamahaConfig, follow as old_follow
        data, times, old, new = self.setup(TAKES[0])
        b7 = YamahaConfig(beat_tempo_reversion=.02)
        cases = [("B7 h1 d0", b7, 1., 0.), ("B7 h0 d1", b7, 0., 1.), ("B7 h0.5 d2", b7, .5, 2.),
                 ("continuous", YamahaConfig(tempo_model="continuous"), 1., 1.)]
        if "exact_beat_crossing" in {f.name for f in fields(YamahaConfig)}:  # the tempo-drift fix, when present
            cases.append(("exact beat crossing", YamahaConfig(exact_beat_crossing=True), 1., 1.))
        for label, config, w_h, w_d in cases:
            with self.subTest(label):
                expected = old_follow(*old["args"], start_time=data["start_time"], network=old["network"] if w_d else None,
                                      config=replace(config, emission_power_per_second=w_h, downbeat_power_per_second=w_d))
                got = follow(times, [(new["cues"][0], w_h), (new["cues"][1], w_d)], *new["plan"],
                             start_time=data["start_time"], config=HMMConfig.like(config))
                self.same(got, expected, label)

    def test_melody_and_per_update_weights(self):
        from hmm_follower.hmm import HMMConfig, follow
        from weighted_filter import follow_weighted
        from yamaha_filter import YamahaConfig
        weights = {"B5 + melody v2 at 1": np.array([1., 1., 1.]),
                   "weights changing every update": np.random.default_rng(0).uniform(0, 2, (N, 3))}
        for performance_id in (TAKES[0], TAKES[2]):
            data, times, old, new = self.setup(performance_id)
            for label, w in weights.items():
                with self.subTest(performance_id=performance_id, case=label):
                    expected = follow_weighted(*old["args"], w, start_time=data["start_time"], network=old["network"],
                                               config=YamahaConfig(), melody=old["melody"])
                    got = follow(times, new["cues"], *new["plan"], start_time=data["start_time"],
                                 config=HMMConfig(), weights=w)
                    self.same(got, expected, f"{performance_id}, {label}")

    def test_start_place(self):
        """As the heads-only runs: start at the true place 20 s in."""
        from hmm_follower.hmm import HMMConfig, follow
        from weighted_filter import follow_weighted
        from yamaha_filter import YamahaConfig
        data, times, old, new = self.setup(TAKES[1], n=400 + N)
        start = float(np.interp(times[400], data["times"], data["truth_position_beats"]))
        args = (times[400:], old["args"][1][400:], old["args"][2][400:]) + old["args"][3:]
        expected = follow_weighted(*args, (1., 1., 1.), start_time=times[400], network=old["network"],
                                   config=YamahaConfig(), melody=old["melody"], start_position=start)
        got = follow(times[400:], new["cues"], *new["plan"], start_time=times[400], start_position=start,
                     config=HMMConfig())
        self.same(got, expected, "start place")

    def test_resets(self):
        """Jiang et al.'s hard reset: back to the true place and tempo when a note finds the follower > 0.1 beat off
        (a small threshold, so that resets happen in 30 s)."""
        import run_jiang_resets
        from hmm_follower.hmm import HMMConfig, follow
        from yamaha_filter import YamahaConfig
        data, times, old, new = self.setup(TAKES[0])
        truth = np.interp(times, data["times"], data["truth_position_beats"])
        truth_bpm = run_jiang_resets.bar_tempo(data, times)
        notes_at = (np.arange(N) % 5 == 0).astype(int)
        expected = run_jiang_resets.follow_with_resets(
            *old["args"], (1., 1., 1.), start_time=data["start_time"], truth_position=truth, truth_bpm=truth_bpm,
            notes_at=notes_at, threshold=.1, network=old["network"], config=YamahaConfig(), melody=old["melody"])

        def reset(frame, estimate):
            if notes_at[frame] and abs(estimate["position_beats"] - truth[frame]) > .1:
                return truth[frame], truth_bpm[frame]

        got = follow(times, new["cues"], *new["plan"], start_time=data["start_time"], config=HMMConfig(), reset=reset)
        self.assertGreater(expected["reset"].sum(), 0)
        np.testing.assert_array_equal(got["reset"], expected["reset"])
        self.same(got, expected, "resets")


if __name__ == "__main__":
    unittest.main()
