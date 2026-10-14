"""Check missing-label behavior and preservation of trained model parameters."""
import tempfile
from pathlib import Path
import unittest

import torch

from yamaha_timing.config import TimingConfig
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.synthetic import overfit_windows
from yamaha_timing.training import load_checkpoint, masked_phase_loss, save_checkpoint
from train_lakh import TRAINING_ARGS, pad_batch, parse_args, scheduled_lr  # run from the experiment folder


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_padded_batch_matches_songs_run_alone(self):
        timing = TimingConfig(channels=16, head_hidden=8, dilations=(1, 2, 4), context_frames=8)
        model = YamahaTimingTCN(timing).eval()
        items = []
        for length, given in ((30, True), (17, False)):
            items.append({"roll": (torch.rand(length, timing.input_pitches) < 0.2).float(),
                          "beat": torch.randint(0, 32, (length,)), "bar": torch.randint(0, 32, (length,)),
                          "mask": torch.ones(length, dtype=torch.bool), "bpm": torch.full((length,), 97.0),
                          "hint_valid": torch.full((length,), given)})
        batch = pad_batch(items)
        self.assertEqual(tuple(batch["roll"].shape), (2, 30, timing.input_pitches))
        self.assertFalse(batch["mask"][1, 17:].any())
        with torch.no_grad():
            together = model(batch["roll"], bpm_hint=batch["bpm"], hint_valid=batch["hint_valid"])
            for k, item in enumerate(items):
                alone = model(item["roll"][None], bpm_hint=item["bpm"][None], hint_valid=item["hint_valid"][None])
                for name in ("beat_logits", "bar_logits"):
                    torch.testing.assert_close(together[name][k, :len(item["roll"])], alone[name][0])

    def test_config_files(self):
        defaults = parse_args(["--data", "d"])
        oyama = parse_args(["--data", "d", "--config", "configs/oyama.json"])
        for key in TRAINING_ARGS:  # lakh_r2 ran on the defaults
            self.assertEqual(getattr(oyama, key), getattr(defaults, key), key)
        batched = parse_args(["--data", "d", "--config", "configs/batch16_cosine.json", "--lr", "5e-4"])
        self.assertEqual((batched.batch_size, batched.schedule, batched.lr), (16, "cosine", 5e-4))
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.json"
            bad.write_text('{"lr": 1e-3, "workers": 2}')
            with self.assertRaises(SystemExit):
                parse_args(["--data", "d", "--config", str(bad)])

    def test_learning_rate_schedules(self):
        self.assertAlmostEqual(scheduled_lr(0, 1e-4, 10, "cosine", 110), 1e-5)
        self.assertAlmostEqual(scheduled_lr(9, 1e-4, 10, "cosine", 110), 1e-4)
        self.assertAlmostEqual(scheduled_lr(10, 1e-4, 10, "cosine", 110), 1e-4)
        self.assertAlmostEqual(scheduled_lr(60, 1e-4, 10, "cosine", 110, 1e-6), 1e-6 + (1e-4 - 1e-6) / 2)
        self.assertAlmostEqual(scheduled_lr(110, 1e-4, 10, "cosine", 110, 1e-6), 1e-6)
        self.assertAlmostEqual(scheduled_lr(5, 1e-4, 10, "plateau", 110), 6e-5)
        self.assertIsNone(scheduled_lr(10, 1e-4, 10, "plateau", 110))
        self.assertIsNone(scheduled_lr(0, 1e-4, 0, "plateau", 110))  # Oyama default: untouched

    def test_masked_labels_have_no_gradient_and_no_effect_on_loss(self):
        logits = torch.randn(1, 3, 4, requires_grad=True)
        outputs = {"beat_logits": logits, "bar_logits": logits.clone()}
        mask = torch.tensor([[True, False, True]])
        labels = torch.tensor([[0, -1, 3]])
        loss = masked_phase_loss(outputs, labels, labels, mask, mask)["total"]
        changed = labels.clone()
        changed[0, 1] = 999
        other = masked_phase_loss(outputs, changed, changed, mask, mask)["total"]
        torch.testing.assert_close(loss, other, atol=0, rtol=0)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.0)

    def test_one_missing_task_is_valid_but_empty_batch_is_rejected(self):
        outputs = {"beat_logits": torch.randn(1, 2, 4, requires_grad=True),
                   "bar_logits": torch.randn(1, 2, 4, requires_grad=True)}
        labels = torch.tensor([[1, 2]])
        absent = torch.zeros_like(labels, dtype=torch.bool)
        result = masked_phase_loss(outputs, labels, labels, beat_mask=absent)
        self.assertEqual(result["beat"].detach().item(), 0.0)
        result["total"].backward()
        self.assertGreater(float(outputs["bar_logits"].grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            masked_phase_loss(outputs, labels, labels, absent, absent)

    def test_checkpoint_roundtrip_preserves_logits_and_provenance(self):
        config = TimingConfig(channels=16, head_hidden=12, dilations=(1, 2), context_frames=8)
        model = YamahaTimingTCN(config).eval()
        inputs = torch.rand(2, 8, 88)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.pt"
            provenance = {"training_data": "synthetic_only", "music_benchmark": False}
            save_checkpoint(target, model, provenance)
            restored, description = load_checkpoint(target)
            self.assertEqual(description, provenance)
            self.assertEqual(restored.config, config)
            self.assertFalse(restored.training)
            for key, output in model(inputs, bpm_hint=120.).items():
                torch.testing.assert_close(output, restored(inputs, bpm_hint=120.)[key], atol=0, rtol=0)

    def test_synthetic_supervision_matches_last_frame_window_contract(self):
        config = TimingConfig(channels=16, head_hidden=12)
        data = overfit_windows(config)
        self.assertEqual(data["piano_roll"].shape[1:], (config.context_frames, 88))
        self.assertTrue(data["beat_mask"][:, -1].all())
        self.assertFalse(data["beat_mask"][:, :-1].any())
        self.assertTrue(torch.equal(data["beat_mask"], data["bar_mask"]))
        self.assertEqual(int(data["bar_targets"][:, -1].unique().numel()), 32)
        self.assertTrue(((data["piano_roll"] == 0) | (data["piano_roll"] == 1)).all())


if __name__ == "__main__":
    unittest.main()
