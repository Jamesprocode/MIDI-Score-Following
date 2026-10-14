"""Can the timing network run live? Per-frame latency of causal streaming inference.

Live use pushes one 20 fps frame (50 ms) at a time and needs the phase
probabilities for that frame before the next one arrives. Each tracker sees
only the current and past frames, with the BPM condition always given (as in
deployment). Measured per frame, after warm-up:

  window  StreamingTimingTracker: re-runs the full 128-frame context per frame
  cached  CachedStreamingTracker: one time step per layer per frame (same outputs)

on every available device (CPU at 1 thread and at the default thread count,
Apple MPS, CUDA). The window tracker cannot run on MPS (its tempo gate uses
float64). Content of the frames does not change the cost; sparse random frames
are used. Writes bench_realtime.json to --out (default $RUN_DIR or .).

  python bench_realtime.py --checkpoint runs/lakh_r3b/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from yamaha_timing.streaming import CachedStreamingTracker, StreamingTimingTracker
from yamaha_timing.training import load_checkpoint

BUDGET_MS = 50.0  # one frame at 20 fps


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(tracker, device, frames, warmup, bpm, rate):
    rng = np.random.default_rng(0)
    rolls = (rng.random((warmup + frames, 88)) < 0.05).astype(np.float32)
    times = []
    for k, frame in enumerate(rolls):
        _sync(device)
        start = time.perf_counter()
        tracker.push_frame(frame, k / rate, bpm)  # returns CPU lists: includes the device sync
        if k >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max()),
            "over_budget": float(np.mean(t > BUDGET_MS)), "frames": int(len(t))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600, help="timed frames per setting (30 s of input)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", ".")))
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    rate = model.config.frame_rate
    settings = [("cpu", 1), ("cpu", torch.get_num_threads())]
    if torch.backends.mps.is_available():
        settings.append(("mps", None))
    if torch.cuda.is_available():
        settings.append(("cuda", None))
    host = {"machine": platform.machine(), "processor": platform.processor() or platform.platform(),
            "cpu_threads_default": torch.get_num_threads(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else
                   ("Apple MPS" if torch.backends.mps.is_available() else None)}
    print(json.dumps(host), flush=True)
    results = []
    default_threads = torch.get_num_threads()
    for device_name, threads in settings:
        device = torch.device(device_name)
        torch.set_num_threads(threads or default_threads)
        for kind in ("window", "cached"):
            if kind == "window" and device.type == "mps":
                results.append({"device": "mps", "threads": None, "tracker": kind,
                                "skipped": "float64 tempo gate is unsupported on MPS"})
                continue
            if kind == "window":
                tracker = StreamingTimingTracker(model.to(device))
            else:
                tracker = CachedStreamingTracker(model, device=None if device.type == "cpu" else device)
            row = {"device": device_name, "threads": threads, "tracker": kind,
                   **measure(tracker, device, args.frames, args.warmup, 120.0, rate)}
            results.append(row)
            print(json.dumps(row), flush=True)
        model = model.to("cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench_realtime.json").write_text(json.dumps({"host": host, "budget_ms": BUDGET_MS,
                                                             "results": results}, indent=1))
    print(f"| device | threads | tracker | median ms | p99 ms | max ms | over {BUDGET_MS:.0f} ms |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        if "skipped" in r:
            print(f"| {r['device']} | - | {r['tracker']} | skipped: {r['skipped']} | | | |")
        else:
            print(f"| {r['device']} | {r['threads'] or '-'} | {r['tracker']} | {r['median_ms']:.2f} | "
                  f"{r['p99_ms']:.2f} | {r['max_ms']:.2f} | {100 * r['over_budget']:.1f}% |")


if __name__ == "__main__":
    main()
