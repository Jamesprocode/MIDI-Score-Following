"""Meaningful framework checks; these do not establish musical accuracy."""

import json
import unittest

import torch
import torch.nn.functional as F

from yamaha_timing.config import TimingConfig
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.streaming import MidiFrameRasterizer, StreamingTimingTracker
from yamaha_timing.training import masked_phase_loss


def tiny_config(**overrides):
    values = dict(channels=16, head_hidden=12, dilations=(1, 2, 4), context_frames=16)
    values.update(overrides)
    return TimingConfig(**values)


class FrameworkTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def assert_distribution(self, values, bins):
        probabilities = torch.as_tensor(values)
        self.assertEqual(probabilities.shape[-1], bins)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities.sum(dim=-1)),
            atol=1e-6,
            rtol=1e-6,
        )


class ConfigTests(FrameworkTestCase):
    def test_published_dimensions_and_explicit_context(self):
        config = TimingConfig()
        self.assertEqual(config.frame_rate, 20)
        self.assertEqual(config.input_pitches, 88)
        self.assertEqual(config.midi_low, 21)
        self.assertEqual(config.channels, 512)
        self.assertEqual(config.bar_bins, 32)
        self.assertEqual(config.dropout, 0.1)
        # Seven kernel-2 layers, dilations 1..64: 6.4 s, the paper's "6 seconds".
        self.assertEqual(config.receptive_field_frames, 128)
        self.assertEqual(config.context_frames, config.receptive_field_frames)
        self.assertEqual(config.context_seconds, 6.4)

    def test_json_round_trip(self):
        original = tiny_config()
        restored = TimingConfig(**json.loads(json.dumps(original.to_dict())))
        self.assertEqual(original, restored)

    def test_invalid_configuration(self):
        cases = [
            {"frame_rate": 0}, {"frame_rate": float("nan")},
            {"context_frames": 0}, {"channels": 0}, {"input_pitches": 0},
            {"head_hidden": 0}, {"kernel_size": 0}, {"beat_bins": 1},
            {"bar_bins": 0}, {"midi_low": -1}, {"midi_low": 80},
            {"dilations": ()}, {"dilations": (1, 0)},
            {"context_frames": 3.5}, {"frame_rate": True},
            {"dropout": 1.0}, {"dropout": -0.1}, {"dropout": True},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TimingConfig(**kwargs)


class ModelTests(FrameworkTestCase):
    def test_shapes_and_probabilities_with_optional_bpm(self):
        config = tiny_config()
        model = YamahaTimingTCN(config).eval()
        roll = torch.rand(2, 17, 88)
        for bpm in (None, 120.0, torch.tensor([90.0, 150.0]), torch.full((2, 17), 110.0)):
            with self.subTest(bpm=bpm):
                outputs = model(roll, bpm_hint=bpm)
                self.assertEqual(set(outputs), {"beat_logits", "bar_logits"})
                self.assertEqual(outputs["beat_logits"].shape, (2, 17, config.beat_bins))
                self.assertEqual(outputs["bar_logits"].shape, (2, 17, config.bar_bins))
                self.assertTrue(torch.isfinite(outputs["beat_logits"]).all())
                probabilities = model.probabilities(roll, bpm_hint=bpm)
                self.assert_distribution(probabilities["beat_probs"], config.beat_bins)
                self.assert_distribution(probabilities["bar_probs"], config.bar_bins)

    def test_missing_hint_mask_bypasses_conditioning(self):
        model = YamahaTimingTCN(tiny_config()).eval()
        roll = torch.rand(2, 12, 88)
        absent = model(roll)
        masked = model(roll, bpm_hint=torch.full((2, 12), float("nan")), hint_valid=False)
        for name in absent:
            torch.testing.assert_close(absent[name], masked[name], atol=0, rtol=0)

    def test_phase_loss_backpropagates_through_both_heads_and_conditioning(self):
        model = YamahaTimingTCN(tiny_config())
        outputs = model(torch.rand(2, 18, 88), bpm_hint=torch.tensor([100.0, 140.0]))
        loss = sum(
            F.cross_entropy(outputs[name].reshape(-1, 32), torch.randint(0, 32, (36,)))
            for name in ("beat_logits", "bar_logits")
        )
        loss.backward()
        # All trainable pathways participate when both heads and BPM are used.
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(any(parameter.grad.abs().sum() > 0 for parameter in model.parameters()))

    def test_future_notes_and_hints_do_not_change_prefix(self):
        model = YamahaTimingTCN(tiny_config()).eval()
        roll = torch.rand(1, 25, 88)
        bpm = torch.full((1, 25), 120.0)
        cutoff = 12
        altered_roll = roll.clone()
        altered_roll[:, cutoff:] = torch.rand_like(altered_roll[:, cutoff:]) * 10
        altered_bpm = bpm.clone()
        altered_bpm[:, cutoff:] = 240.0
        first = model(roll, bpm_hint=bpm)
        second = model(altered_roll, bpm_hint=altered_bpm)
        for name in first:
            torch.testing.assert_close(first[name][:, :cutoff], second[name][:, :cutoff], atol=1e-6, rtol=1e-6)

    def test_invalid_model_input(self):
        model = YamahaTimingTCN(tiny_config())
        for invalid in (torch.zeros(3, 88), torch.zeros(1, 4, 87), torch.zeros(1, 0, 88)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                model(invalid)
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 5, 88, dtype=torch.long))
        for bpm in (0.0, -120.0, float("nan"), torch.ones(2, 2, 2)):
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                model(torch.rand(1, 5, 88), bpm_hint=bpm)
        with self.assertRaises(TypeError):
            model(torch.rand(1, 5, 88), bpm_hint=120.0, hint_valid=1.0)


class StreamingTests(FrameworkTestCase):
    def test_streaming_matches_each_past_only_rolling_window(self):
        config = tiny_config(context_frames=8)
        model = YamahaTimingTCN(config).eval()
        tracker = StreamingTimingTracker(model)
        frames = torch.rand(13, 88)
        hint_history = [None, 100.0, 100.0, None, 120.0, 120.0, 150.0, None, None, 110.0, 110.0, 90.0, None]
        for index, frame in enumerate(frames):
            record = tracker.push_frame(frame, index / config.frame_rate, bpm_hint=hint_history[index])
            begin = max(0, index + 1 - config.context_frames)
            count = index + 1 - begin
            padding = config.context_frames - count
            roll = torch.cat((torch.zeros(padding, 88), frames[begin:index + 1])).unsqueeze(0)
            hints = [0.0] * padding + [value or 0.0 for value in hint_history[begin:index + 1]]
            valid = [False] * padding + [value is not None for value in hint_history[begin:index + 1]]
            outputs = model(roll, bpm_hint=torch.tensor([hints]), hint_valid=torch.tensor([valid]))
            expected_beat = outputs["beat_logits"][0, -1].softmax(dim=-1)
            expected_bar = outputs["bar_logits"][0, -1].softmax(dim=-1)
            torch.testing.assert_close(torch.tensor(record["beat_phase_probabilities"]), expected_beat, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(torch.tensor(record["bar_phase_probabilities"]), expected_bar, atol=1e-6, rtol=1e-6)
            self.assertEqual(record["context_frames_seen"], min(index + 1, config.context_frames))
            self.assertEqual(record["context_ready"], index + 1 >= config.context_frames)
            self.assertEqual(record["timestamp_s"], index / config.frame_rate)
            json.dumps(record, allow_nan=False)

    def test_window_equal_to_receptive_field_matches_whole_sequence_pass(self):
        # Training runs whole songs in one causal pass; live use runs a rolling
        # window over silent (zero) startup frames. They agree exactly when the
        # window equals the receptive field and the whole sequence is preceded
        # by RF-1 silent frames: per-layer zero padding is not the same as
        # zero input, so training data carries that silent lead-in.
        config = tiny_config()
        config = tiny_config(context_frames=config.receptive_field_frames)
        lead = config.receptive_field_frames - 1
        model = YamahaTimingTCN(config).eval()
        tracker = StreamingTimingTracker(model)
        frames = (torch.rand(40, 88) < 0.1).float()
        hints = [None if index % 7 == 3 else 100.0 + index for index in range(40)]
        padded = torch.cat((torch.zeros(lead, 88), frames)).unsqueeze(0)
        with torch.no_grad():
            whole = model(padded,
                          bpm_hint=torch.tensor([[120.0] * lead + [value or 120.0 for value in hints]]),
                          hint_valid=torch.tensor([[False] * lead + [value is not None for value in hints]]))
        for index, frame in enumerate(frames):
            record = tracker.push_frame(frame, index / config.frame_rate, bpm_hint=hints[index])
            for name in ("bar", "beat"):
                torch.testing.assert_close(torch.tensor(record[f"{name}_phase_probabilities"]),
                                           whole[f"{name}_logits"][0, lead + index].softmax(-1),
                                           atol=1e-5, rtol=1e-5)

    def test_startup_silence_and_reset(self):
        config = tiny_config(context_frames=4)
        model = YamahaTimingTCN(config).eval()
        tracker = StreamingTimingTracker(model)
        first = tracker.push_frame(torch.zeros(88), 0.0)
        self.assertFalse(first["context_ready"])
        for index in range(1, 6):
            record = tracker.push_frame(torch.zeros(88), index / config.frame_rate)
            # Same zero-padded silent window: no synthetic pitch evidence appears.
            self.assertEqual(first["bar_phase_probabilities"], record["bar_phase_probabilities"])
            self.assert_distribution(record["beat_phase_probabilities"], config.beat_bins)
        tracker.reset()
        after_reset = tracker.push_frame(torch.zeros(88), 10.0)
        self.assertEqual(after_reset["context_frames_seen"], 1)
        self.assertFalse(after_reset["context_ready"])
        self.assertEqual(first["bar_phase_probabilities"], after_reset["bar_phase_probabilities"])

    def test_reject_invalid_frame_or_sampling_time(self):
        tracker = StreamingTimingTracker(YamahaTimingTCN(tiny_config()))
        with self.assertRaises(ValueError):
            tracker.push_frame(torch.zeros(87), 0.0)
        with self.assertRaises(ValueError):
            tracker.push_frame(torch.full((88,), float("nan")), 0.0)
        tracker.push_frame(torch.zeros(88), 0.0)
        for timestamp in (0.0, -0.05, 0.1, float("nan")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                tracker.push_frame(torch.zeros(88), timestamp)


class TrainingContractTests(FrameworkTestCase):
    def test_masked_labels_do_not_contribute_loss_or_gradients(self):
        beat = torch.randn(1, 4, 32, requires_grad=True)
        bar = torch.randn(1, 4, 32, requires_grad=True)
        targets = torch.tensor([[2, -100, 3, -100]])
        mask = torch.tensor([[True, False, True, False]])
        losses = masked_phase_loss(
            {"beat_logits": beat, "bar_logits": bar}, targets, targets, mask, mask
        )
        expected = F.cross_entropy(beat[mask], targets[mask]) + F.cross_entropy(bar[mask], targets[mask])
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertEqual(int(torch.count_nonzero(beat.grad[~mask])), 0)
        self.assertEqual(int(torch.count_nonzero(bar.grad[~mask])), 0)
        self.assertTrue((beat.grad[mask].abs().sum() > 0).item())

    def test_missing_one_task_is_allowed_but_empty_training_batch_is_rejected(self):
        outputs = {name: torch.randn(1, 3, 32, requires_grad=True) for name in ("beat_logits", "bar_logits")}
        targets = torch.zeros(1, 3, dtype=torch.long)
        empty = torch.zeros(1, 3, dtype=torch.bool)
        losses = masked_phase_loss(outputs, targets, targets, bar_mask=empty)
        self.assertEqual(losses["bar"].detach().item(), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(outputs["bar_logits"].grad)
        self.assertEqual(int(torch.count_nonzero(outputs["bar_logits"].grad)), 0)
        with self.assertRaises(ValueError):
            masked_phase_loss(outputs, targets, targets, beat_mask=empty, bar_mask=empty)


class MidiRasterizerTests(FrameworkTestCase):
    def active_pitch(self, record, pitch):
        return record["frame"][pitch - 21]

    def test_exact_event_boundaries_and_no_future_note_leakage(self):
        rasterizer = MidiFrameRasterizer()
        rasterizer.push_event("note_on", 60, 100, 0.1)
        early = rasterizer.advance_to(0.05)
        self.assertEqual([frame["timestamp_s"] for frame in early], [0.0, 0.05])
        self.assertTrue(all(not any(frame["frame"]) for frame in early))
        onset = rasterizer.advance_to(0.1)
        self.assertEqual(len(onset), 1)
        self.assertEqual(self.active_pitch(onset[0], 60), 1.0)
        rasterizer.push_event("note_off", 60, 0, 0.15)
        release = rasterizer.advance_to(0.15)
        self.assertEqual(self.active_pitch(release[0], 60), 0.0)

    def test_overlapping_notes_and_channels_preserve_held_key(self):
        rasterizer = MidiFrameRasterizer()
        rasterizer.push_event("note_on", 60, 100, 0.0, channel=0)
        rasterizer.push_event("note_on", 60, 80, 0.0, channel=0)
        rasterizer.push_event("note_on", 60, 90, 0.0, channel=1)
        self.assertEqual(self.active_pitch(rasterizer.advance_to(0.0)[0], 60), 1.0)
        for timestamp, channel in ((0.05, 0), (0.1, 0)):
            rasterizer.push_event("note_off", 60, 0, timestamp, channel=channel)
            self.assertEqual(self.active_pitch(rasterizer.advance_to(timestamp)[0], 60), 1.0)
        rasterizer.push_event("note_on", 60, 0, 0.15, channel=1)
        self.assertEqual(self.active_pitch(rasterizer.advance_to(0.15)[0], 60), 0.0)

    def test_out_of_range_unmatched_off_and_reset_diagnostics(self):
        rasterizer = MidiFrameRasterizer()
        rasterizer.push_event("note_on", 20, 100, 0.0)
        rasterizer.push_event("note_on", 109, 100, 0.0)
        rasterizer.push_event("note_off", 60, 0, 0.0)
        record = rasterizer.advance_to(0.0)[0]
        self.assertFalse(any(record["frame"]))
        self.assertEqual(rasterizer.diagnostics["ignored_out_of_range_events"], 2)
        self.assertEqual(rasterizer.diagnostics["unmatched_note_off_events"], 1)
        rasterizer.reset(start_time_s=2.0)
        self.assertEqual(rasterizer.diagnostics["ignored_out_of_range_events"], 0)
        self.assertEqual(rasterizer.diagnostics["unmatched_note_off_events"], 0)
        self.assertEqual(rasterizer.advance_to(2.0)[0]["timestamp_s"], 2.0)

    def test_late_events_cannot_revise_emitted_frames(self):
        rasterizer = MidiFrameRasterizer()
        rasterizer.advance_to(0.0)
        with self.assertRaises(ValueError):
            rasterizer.push_event("note_on", 60, 100, 0.0)
        rasterizer.advance_to(0.04)
        with self.assertRaises(ValueError):
            rasterizer.push_event("note_on", 60, 100, 0.03)


if __name__ == "__main__":
    unittest.main()
