"""Save streaming phase output and measured compute time on synthetic MIDI.

Random weights are explicitly marked untrained. A toy-trained checkpoint also
does not imply accuracy on real performances. BPM in this demo is an input,
not a prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from yamaha_timing.config import TimingConfig
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.streaming import StreamingTimingTracker
from yamaha_timing.synthetic import periodic_roll
from yamaha_timing.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    if args.frames < 1 or args.threads < 1:
        parser.error("frames and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    if args.checkpoint:
        model, provenance = load_checkpoint(args.checkpoint)
    else:
        model = YamahaTimingTCN(TimingConfig(channels=args.channels))
        provenance = {"training_data": "none", "untrained": True}
    tracker = StreamingTimingTracker(model)
    roll, _, _ = periodic_roll(model.config, args.frames)
    # Separate startup/warmup from the reported steady-call compute distribution.
    for frame in range(5):
        tracker.push_frame(np.zeros(model.config.input_pitches, dtype=np.float32), frame / model.config.frame_rate, bpm_hint=120.)
    tracker.reset()
    records, latencies = [], []
    for frame, activity in enumerate(roll):
        start = time.perf_counter()
        record = tracker.push_frame(activity, frame / model.config.frame_rate, bpm_hint=120.)
        latencies.append((time.perf_counter() - start) * 1000.0)
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "phase_stream.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    result = {
        "purpose": "local CPU per-update compute profile; not beat accuracy or end-to-end latency",
        "config": model.config.to_dict(), "provenance": provenance,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "frames": args.frames, "threads": args.threads, "device": "cpu",
        "compute_ms_median": float(np.median(latencies)),
        "compute_ms_p95": float(np.percentile(latencies, 95)),
        "compute_ms_max": float(np.max(latencies)),
        "update_interval_ms": 1000.0 / model.config.frame_rate,
        "updates_exceeding_interval": sum(value > 1000.0 / model.config.frame_rate for value in latencies),
        "context_frames": model.config.context_frames,
        "torch_version": str(torch.__version__), "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
