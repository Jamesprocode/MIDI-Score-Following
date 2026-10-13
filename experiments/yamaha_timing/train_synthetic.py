"""Overfit a deliberately easy MIDI pattern to verify learning and checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from yamaha_timing.config import TimingConfig
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.synthetic import overfit_windows
from yamaha_timing.training import load_checkpoint, masked_phase_loss, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic"))
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or args.lr <= 0:
        parser.error("steps, threads and learning rate must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = TimingConfig(channels=args.channels, head_hidden=args.channels)
    model = YamahaTimingTCN(config)
    data = overfit_windows(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial = masked_phase_loss(model(data["piano_roll"], bpm_hint=data["bpm_hint"]),
                                    data["beat_targets"], data["bar_targets"],
                                    data["beat_mask"], data["bar_mask"])
        initial_loss = float(initial["total"])
    history = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Match the paper's 50% BPM-condition availability as an explicit toy policy.
        hint_valid = torch.rand(len(data["piano_roll"])) < 0.5
        outputs = model(data["piano_roll"], bpm_hint=data["bpm_hint"], hint_valid=hint_valid)
        losses = masked_phase_loss(outputs, data["beat_targets"], data["bar_targets"],
                                   data["beat_mask"], data["bar_mask"])
        losses["total"].backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "total_loss": float(losses["total"].detach())}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    evaluation = {}
    with torch.no_grad():
        for condition in ("given_bpm", "missing_bpm"):
            outputs = model(data["piano_roll"], bpm_hint=data["bpm_hint"] if condition == "given_bpm" else None)
            losses = masked_phase_loss(outputs, data["beat_targets"], data["bar_targets"],
                                       data["beat_mask"], data["bar_mask"])
            evaluation[condition] = {
                "total_loss": float(losses["total"]),
                "beat_last_frame_accuracy": float((outputs["beat_logits"][:, -1].argmax(-1) == data["beat_targets"][:, -1]).float().mean()),
                "bar_last_frame_accuracy": float((outputs["bar_logits"][:, -1].argmax(-1) == data["bar_targets"][:, -1]).float().mean()),
            }
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    provenance = {"training_data": "synthetic_periodic_keyboard_only", "seed": args.seed,
                  "steps": args.steps, "music_benchmark": False}
    save_checkpoint(checkpoint, model, provenance)
    restored, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        original = model(data["piano_roll"][:1])
        roundtrip = restored(data["piano_roll"][:1])
        checkpoint_equal = all(torch.equal(original[key], roundtrip[key]) for key in original)
    result = {
        "purpose": "implementation overfit check; no music generalization claim",
        "config": config.to_dict(), "seed": args.seed, "steps": args.steps,
        "learning_rate": args.lr, "threads": args.threads,
        "initial_given_bpm_loss": initial_loss, "evaluation": evaluation,
        "checkpoint_roundtrip_exact": checkpoint_equal,
        "elapsed_seconds": elapsed, "history": history,
        "torch_version": str(torch.__version__), "python": platform.python_version(),
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"evaluation": evaluation, "checkpoint_roundtrip_exact": checkpoint_equal,
                      "elapsed_seconds": elapsed, "metrics": str(args.output / "metrics.json")}), flush=True)
    if not checkpoint_equal or evaluation["given_bpm"]["total_loss"] >= initial_loss * 0.5:
        raise SystemExit("Synthetic check failed: loss did not fall sufficiently or checkpoint differed")


if __name__ == "__main__":
    main()
