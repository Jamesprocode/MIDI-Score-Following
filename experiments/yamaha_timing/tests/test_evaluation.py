"""Beat/downbeat decoding and tracking metrics, against known references."""
import unittest
import warnings

import numpy as np

from yamaha_timing.config import TimingConfig
from yamaha_timing.evaluation import decode_events, expected_phase, p_score, tracking_metrics
from yamaha_timing.plots import plot_tracking
from yamaha_timing.lakh import SynthesisConfig, render_example

from test_lakh import song_on_beats  # same folder; discovery puts tests/ on sys.path

TIMING = TimingConfig(channels=16, head_hidden=8, dilations=(1, 2, 4), context_frames=8)


def one_hot(bins, classes=32):
    probabilities = np.zeros((len(bins), classes))
    probabilities[np.arange(len(bins)), bins] = 1.0
    return probabilities


class DecoderTests(unittest.TestCase):
    def test_sawtooth_phase_gives_exact_event_times(self):
        times = np.arange(400) / 20
        phase = np.mod(times / 0.55 + 0.3, 1.0)  # beats every 0.55 s, first at 0.385 s
        events = decode_events(phase, 20)
        expected = (np.arange(1, 50) - 0.3) * 0.55
        np.testing.assert_allclose(events, expected[expected < times[-1]], atol=1e-9)

    def test_jitter_around_the_wrap_fires_once(self):
        phase = np.array([0.6, 0.9, 0.98, 0.01, 0.99, 0.02, 0.97, 0.03, 0.3, 0.6, 0.95, 0.05])
        events = decode_events(phase, 20)
        self.assertEqual(len(events), 2)  # the first wrap, then the one after real progress
        np.testing.assert_allclose(events[0], (2 + 0.02 / 0.03) / 20)

    def test_expected_phase_is_circular(self):
        p = np.zeros((1, 32))
        p[0, [31, 0]] = 0.5  # half on the last bin, half on the first: phase near 0
        self.assertLess(min(expected_phase(p)[0], 1 - expected_phase(p)[0]), 1e-9)


class MetricTests(unittest.TestCase):
    def test_perfect_empty_and_shifted_estimates(self):
        reference = np.arange(0, 60, 0.5)
        perfect = tracking_metrics(reference, reference)
        self.assertEqual(perfect["f_measure"], 1.0)
        self.assertEqual(perfect["cmlt"], 1.0)
        self.assertEqual(perfect["amlt"], 1.0)
        self.assertEqual(tracking_metrics(reference, [])["f_measure"], 0.0)
        offbeat = tracking_metrics(reference, reference + 0.25)  # half a beat late
        self.assertEqual(offbeat["f_measure"], 0.0)
        self.assertGreater(offbeat["amlt"], 0.9)  # off-beat counts at "any metric level"


class ReferenceTimeTests(unittest.TestCase):
    def test_reference_times_follow_the_meter(self):
        quiet = SynthesisConfig(max_delete=0, max_insert=0, drift_step_max=0)
        four = render_example(song_on_beats(), np.random.default_rng(0), TIMING, quiet, lead_in=False)
        np.testing.assert_allclose(four["beat_times"][:4], [0, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(four["downbeat_times"][:3], [0, 2.0, 4.0])
        six = render_example(song_on_beats(meter=((0.0, 6, 8),)), np.random.default_rng(0), TIMING, quiet,
                             lead_in=False)
        np.testing.assert_allclose(six["beat_times"][:3], [0, 0.25, 0.5])   # eighth-note beats
        np.testing.assert_allclose(six["downbeat_times"][:3], [0, 1.5, 3.0])

    def test_decoding_the_true_labels_scores_perfectly(self):
        synth = SynthesisConfig(max_delete=0, max_insert=0, drift_step_max=0.03)
        example = render_example(song_on_beats(beats=300, bpm=100.0, meter=((0.0, 4, 4), (100.0, 3, 4))),
                                 np.random.default_rng(3), TIMING, synth)
        song = example["mask"]
        for labels, reference in (("beat", "beat_times"), ("bar", "downbeat_times")):
            events = decode_events(expected_phase(one_hot(example[labels][song])), 20)
            scores = tracking_metrics(example[reference], events)
            self.assertEqual(scores["f_measure"], 1.0, labels)
            self.assertEqual(scores["cmlt"], 1.0, labels)


class PlotTests(unittest.TestCase):
    def test_validation_figure_is_written(self):
        import tempfile
        from pathlib import Path
        example = render_example(song_on_beats(beats=120), np.random.default_rng(0), TIMING)
        song = example["mask"]
        probs = {name: one_hot(example[name][song]) for name in ("beat", "bar")}
        decoded = {name: decode_events(expected_phase(probs[name]), 20) for name in probs}
        figure = {
            "title": "test song", "roll": example["roll"][song],
            "beat_probs": probs["beat"], "bar_probs": probs["bar"],
            "beat_labels": example["beat"][song], "bar_labels": example["bar"][song],
            "true_beats": example["beat_times"], "true_downbeats": example["downbeat_times"],
            "pred_beats": decoded["beat"], "pred_downbeats": decoded["bar"],
            "scores": {"beat": tracking_metrics(example["beat_times"], decoded["beat"]),
                       "downbeat": tracking_metrics(example["downbeat_times"], decoded["bar"])},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figure.png"
            plot_tracking(figure, path)
            self.assertGreater(path.stat().st_size, 10_000)


class PScoreTests(unittest.TestCase):
    def test_matches_mir_eval_exactly(self):
        import mir_eval

        rng = np.random.default_rng(3)
        for case in range(300):
            period = rng.uniform(0.25, 1.2)
            reference = np.cumsum(np.full(int(rng.integers(0, 120)), period) + rng.normal(0, 0.01, 1).clip(-0.1))
            reference = reference + rng.uniform(0, 8)
            kind = case % 4
            estimated = (reference + rng.normal(0, rng.uniform(0, 0.08), len(reference)) if kind == 0
                         else reference[::2] + rng.uniform(-0.2, 0.2) if kind == 1
                         else np.sort(rng.uniform(0, 70, int(rng.integers(0, 150)))) if kind == 2
                         else np.sort(np.concatenate([reference[rng.random(len(reference)) < 0.7],
                                                      rng.uniform(0, 70, 10)])))
            reference, estimated = np.sort(reference), np.sort(estimated)
            with self.subTest(case=case):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    expected = mir_eval.beat.p_score(reference, estimated)
                    self.assertEqual(p_score(reference, estimated), expected)


if __name__ == "__main__":
    unittest.main()
