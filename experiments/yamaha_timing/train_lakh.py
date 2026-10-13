"""Train the Maezawa (2023) beat/downbeat-phase TCN on synthetic Lakh performances.

Where Maezawa is silent, training follows Oyama et al. (2021), the phase-based
tracker Maezawa cites: Adam at 1e-3, one whole song per step, learning rate
halved after 15 epochs without validation improvement, stop after 50 such
epochs or at 200 epochs, gradients clipped at 0.5 (by value). Each epoch
visits every training song once with fresh augmentation; validation songs use
fixed augmentation, so the validation loss is comparable across epochs.

Resumable (PACE ICE contract): the newest checkpoint in --out is loaded on
start; SIGUSR1/SIGTERM save a checkpoint and exit 0; a checkpoint is written
every --checkpoint-minutes; DONE is touched when training ends.

Curves go to W&B (project midi-score-following). On ICE the job chain sets the
run id, name and a temporary W&B directory, so every resumed job extends one
run. For local smoke tests set WANDB_MODE=disabled.

  python train_lakh.py --data data/lakh_v1 --out runs/lakh_r1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from yamaha_timing.config import TimingConfig
from yamaha_timing.lakh import SongPack, SynthesisConfig, render_example
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.training import masked_phase_loss, save_checkpoint

FIXED_EPOCH = 10**9  # augmentation key for validation songs: identical every evaluation
STOP = {"signal": None}


class SongExamples(Dataset):
    """Item ``(epoch, index)`` -> one rendered song; the key fixes the augmentation."""

    def __init__(self, pack_path, timing, synth, seed, limit=0):
        self.pack_path, self.timing, self.synth, self.seed = Path(pack_path), timing, synth, seed
        self.pack = None
        with np.load(self.pack_path, allow_pickle=False) as data:
            total = len(data["ids"])
        self.size = min(total, limit) if limit else total

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        epoch, index = key
        if self.pack is None:  # load lazily so each DataLoader worker reads its own copy
            self.pack = SongPack(self.pack_path)
        example = render_example(self.pack.song(index), np.random.default_rng([self.seed, epoch, index]),
                                 self.timing, self.synth)
        return {
            "roll": torch.from_numpy(example["roll"]).float(),
            "beat": torch.from_numpy(example["beat"]), "bar": torch.from_numpy(example["bar"]),
            "mask": torch.from_numpy(example["mask"]), "bpm": torch.from_numpy(example["bpm"]),
            "hint_valid": torch.from_numpy(example["hint_valid"]),
        }


def loader(dataset, keys, workers, device):
    # A private generator: creating a DataLoader iterator otherwise draws from the
    # global torch RNG, so a resumed run (one extra iterator) would get different
    # dropout masks. Augmentation is keyed by (seed, epoch, index), not by this.
    return DataLoader(dataset, batch_size=None, sampler=keys, num_workers=workers,
                      pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None,
                      generator=torch.Generator().manual_seed(0))


def forward(model, item, device):
    to = lambda name: item[name].to(device, non_blocking=True).unsqueeze(0)
    outputs = model(to("roll"), bpm_hint=to("bpm"), hint_valid=to("hint_valid"))
    return outputs, to("beat"), to("bar"), to("mask"), to("hint_valid")


@torch.no_grad()
def validate(model, dataset, workers, device):
    """Frame-weighted loss and accuracy over all validation songs, split by hint."""
    model.eval()
    sums = {}
    for item in loader(dataset, [(FIXED_EPOCH, i) for i in range(len(dataset))], workers, device):
        outputs, beat, bar, mask, hint = forward(model, item, device)
        group = "hint" if bool(hint[mask].any()) else "no_hint"
        for task, target in (("beat", beat), ("bar", bar)):
            logits, labels = outputs[f"{task}_logits"][mask], target[mask]
            bins = logits.shape[-1]
            error = (logits.argmax(-1) - labels).remainder(bins)
            near = torch.minimum(error, bins - error) <= 1
            for key in ("all", group):
                stats = sums.setdefault(f"{key}/{task}", np.zeros(4))
                stats += [F.cross_entropy(logits, labels, reduction="sum").item(), len(labels),
                          (error == 0).sum().item(), near.sum().item()]
    result = {}  # plain floats: checkpoints are loaded with weights_only=True
    for name, (loss, count, exact, near) in sums.items():
        result[f"{name}_loss"] = float(loss / count)
        result[f"{name}_acc"] = float(exact / count)
        result[f"{name}_acc_within1"] = float(near / count)
    result["loss"] = result["all/beat_loss"] + result["all/bar_loss"]
    model.train()
    return result


def save_state(path, state):
    """Atomic write; the previous checkpoint is kept as ckpt_previous.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    if path.exists():
        os.replace(path, path.with_name("ckpt_previous.pt"))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="folder with train.npz, val.npz, manifest.json")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUN_DIR", "runs/lakh")))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--stop-patience", type=int, default=50)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-minutes", type=float, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--limit-train", type=int, default=0, help="smoke tests only")
    parser.add_argument("--limit-val", type=int, default=0, help="smoke tests only")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    for name in ("SIGUSR1", "SIGTERM"):
        signal.signal(getattr(signal, name), lambda signum, _: STOP.update(signal=signal.Signals(signum).name))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())

    timing = TimingConfig(channels=args.channels, head_hidden=args.head_hidden)
    synth = SynthesisConfig()
    train = SongExamples(args.data / "train.npz", timing, synth, args.seed, args.limit_train)
    val = SongExamples(args.data / "val.npz", timing, synth, args.seed, args.limit_val)
    torch.manual_seed(args.seed)
    model = YamahaTimingTCN(timing).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=args.lr_patience)
    state = {"epoch": 0, "next_index": 0, "best_val": math.inf, "bad_epochs": 0, "history": [], "steps": 0}

    latest = args.out / "ckpt_latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved["config"] != timing.to_dict() or saved["synthesis"] != synth.to_dict():
            sys.exit(f"{latest} was made with a different model or synthesis config; use a new --out")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state = {key: saved[key] for key in state}
        print(json.dumps({"resumed": str(latest), "epoch": state["epoch"], "next_index": state["next_index"]}), flush=True)
    else:
        print(json.dumps({"start": True, "device": str(device), "train_songs": len(train), "val_songs": len(val),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "config": timing.to_dict(), "synthesis": synth.to_dict(),
                          "data": manifest.get("packs"), "torch": torch.__version__,
                          "python": platform.python_version()}), flush=True)
    # No dir/id/resume here: on ICE the job chain supplies them through WANDB_* variables.
    wandb.init(project="midi-score-following", config={
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": timing.to_dict(), "synthesis": synth.to_dict(), "data": manifest.get("packs"),
        "data_funnel": manifest.get("funnel"), "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "torch": torch.__version__, "train_songs": len(train), "val_songs": len(val),
    })

    def checkpoint(reason):
        save_state(latest, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "config": timing.to_dict(), "synthesis": synth.to_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "reason": reason, **state,
        })

    last_save = time.time()
    while True:
        epoch = state["epoch"]
        if epoch >= args.epochs or state["bad_epochs"] >= args.stop_patience:
            break
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train))
        keys = [(epoch, int(i)) for i in order[state["next_index"]:]]
        window = {"loss": 0.0, "steps": 0, "frames": 0, "start": time.time()}
        for item in loader(train, keys, args.workers, device):
            outputs, beat, bar, mask, _ = forward(model, item, device)
            loss = masked_phase_loss(outputs, beat, bar, mask, mask)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
            optimizer.step()
            state["next_index"] += 1
            state["steps"] += 1
            window["loss"] += loss.item()
            window["steps"] += 1
            window["frames"] += int(mask.sum())
            if window["steps"] == args.log_every:
                elapsed = time.time() - window["start"]
                row = {"epoch": epoch, "song": state["next_index"], "of": len(train),
                       "train_loss": round(window["loss"] / window["steps"], 4),
                       "lr": optimizer.param_groups[0]["lr"],
                       "songs_per_s": round(window["steps"] / elapsed, 2),
                       "frames_per_s": round(window["frames"] / elapsed)}
                print(json.dumps(row), flush=True)
                wandb.log({"train/loss": row["train_loss"], "lr": row["lr"],
                           "train/songs_per_s": row["songs_per_s"], "train/frames_per_s": row["frames_per_s"],
                           "epoch": epoch + state["next_index"] / len(train)}, step=state["steps"])
                window = {"loss": 0.0, "steps": 0, "frames": 0, "start": time.time()}
            if STOP["signal"]:
                checkpoint(STOP["signal"])
                print(json.dumps({"stopped_by": STOP["signal"], "epoch": epoch, "next_index": state["next_index"]}),
                      flush=True)
                wandb.finish()
                return
            if time.time() - last_save > args.checkpoint_minutes * 60:
                checkpoint("periodic")
                last_save = time.time()

        metrics = validate(model, val, args.workers, device)
        scheduler.step(metrics["loss"])
        improved = metrics["loss"] < state["best_val"]
        if improved:
            state["best_val"], state["bad_epochs"] = metrics["loss"], 0
            save_checkpoint(args.out / "best_model.pt", model, {
                "training_data": "Lakh LMD-matched, cleaned (prepare_lakh.py), synthetic performances",
                "data_packs": manifest.get("packs"), "epoch": epoch, "validation": metrics,
                "synthesis": synth.to_dict(), "music_benchmark": False})
        else:
            state["bad_epochs"] += 1
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "improved": improved,
                  "bad_epochs": state["bad_epochs"], **{k: round(v, 5) for k, v in metrics.items()}}
        state["history"].append(record)
        with open(args.out / "metrics.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "val/best_loss": state["best_val"],
                   "val/bad_epochs": state["bad_epochs"], "lr": optimizer.param_groups[0]["lr"],
                   "epoch": epoch + 1}, step=state["steps"])
        state["epoch"], state["next_index"] = epoch + 1, 0
        checkpoint("epoch_end")
        last_save = time.time()
        if STOP["signal"]:
            wandb.finish()
            return

    checkpoint("finished")
    (args.out / "DONE").touch()
    wandb.summary.update({"best_val_loss": state["best_val"], "epochs": state["epoch"]})
    wandb.finish()
    print(json.dumps({"finished": True, "epochs": state["epoch"], "best_val": state["best_val"]}), flush=True)


if __name__ == "__main__":
    main()
