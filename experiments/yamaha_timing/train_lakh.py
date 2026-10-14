"""Train the Maezawa (2023) beat/downbeat-phase TCN on synthetic Lakh performances.

Where Maezawa is silent, training follows Oyama et al. (2021), the phase-based
tracker Maezawa cites: Adam (at 1e-4, not Oyama's 1e-3, which left this
512-channel model at chance), one whole song per step, learning rate
halved after 15 epochs without validation improvement, stop after 50 such
epochs or at 200 epochs, gradients clipped at 0.5 (by value). Those are the
defaults. Options beyond Oyama, to use the GPU better: --batch-size songs per
step, each a random --crop-seconds excerpt with the 6.4 s before it as
unlabelled context (end-padded and masked, exact for a causal model); a linear
--warmup-steps; and --schedule cosine (decay to --min-lr by --epochs, early
stopping kept). Each epoch visits every training song group once, using one of
its arrangements chosen at random, with fresh augmentation. Validation uses each group's best-matched
arrangement with fixed augmentation, so the validation loss is comparable
across epochs. Validation also decodes beat and downbeat times from the phase
outputs and reports mir_eval's beat-tracking metrics (F-measure, CMLt, AMLt,
...); the best checkpoint is still chosen by validation loss.

Resumable (PACE ICE contract): the newest checkpoint in --out is loaded on
start; SIGUSR1/SIGTERM save a checkpoint and exit 0; a checkpoint is written
every --checkpoint-minutes; DONE is touched when training ends.

Curves go to W&B (project midi-score-following). On ICE the job chain sets the
run id, name and a temporary W&B directory, so every resumed job extends one
run. For local smoke tests set WANDB_MODE=disabled.

  python train_lakh.py --data data/lakh_v2 --out runs/lakh_r1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from yamaha_timing.evaluation import decode_events, expected_phase, tracking_metrics
from yamaha_timing.plots import plot_tracking
from yamaha_timing.lakh import IGNORE_INDEX, SongPack, SynthesisConfig, crop_example, render_example
from yamaha_timing.model import YamahaTimingTCN
from yamaha_timing.training import masked_phase_loss, save_checkpoint

FIXED_EPOCH = 10**9  # augmentation key for validation songs: identical every evaluation
STOP = {"signal": None}


class SongExamples(Dataset):
    """Item ``(epoch, index)`` -> one rendered song; the key fixes the augmentation.

    ``groups`` lists each song group's file indices, best-matched first.
    """

    def __init__(self, pack_path, timing, synth, seed, limit=0, crop_frames=0):
        self.pack_path, self.timing, self.synth, self.seed = Path(pack_path), timing, synth, seed
        self.crop_frames = crop_frames  # 0 = whole songs
        self.pack = None
        with np.load(self.pack_path, allow_pickle=False) as data:
            group, self.meters, self.ids = data["meta_group"], data["meta_meter"], data["ids"]
        self.groups = [list(np.flatnonzero(group == g)) for g in range(group.max() + 1)]
        if limit:
            self.groups = self.groups[:limit]

    def __len__(self):
        return len(self.meters)

    def __getitem__(self, key):
        epoch, index = key
        if self.pack is None:  # load lazily so each DataLoader worker reads its own copy
            self.pack = SongPack(self.pack_path)
        rng = np.random.default_rng([self.seed, epoch, index])
        example = render_example(self.pack.song(index), rng, self.timing, self.synth)
        if self.crop_frames:
            example = crop_example(example, rng, self.crop_frames, self.timing.context_frames - 1)
        item = {
            "roll": torch.from_numpy(example["roll"]).float(),
            "beat": torch.from_numpy(example["beat"]), "bar": torch.from_numpy(example["bar"]),
            "mask": torch.from_numpy(example["mask"]), "bpm": torch.from_numpy(example["bpm"]),
            "hint_valid": torch.from_numpy(example["hint_valid"]),
            "four_four": torch.tensor(self.meters[index] == "4/4"),
        }
        for name in ("beat_times", "downbeat_times"):  # whole songs only (validation)
            if name in example:
                item[name] = torch.from_numpy(example[name])
        return item

    def epoch_keys(self, epoch):
        """One randomly chosen arrangement per group, groups in random order."""
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(len(self.groups))
        return [(epoch, int(self.groups[g][rng.integers(len(self.groups[g]))])) for g in order]

    def fixed_keys(self):
        """Each group's best-matched arrangement, for validation."""
        return [(FIXED_EPOCH, int(members[0])) for members in self.groups]


def loader(dataset, keys, workers, device):
    # A private generator: creating a DataLoader iterator otherwise draws from the
    # global torch RNG, so a resumed run (one extra iterator) would get different
    # dropout masks. Augmentation is keyed by (seed, epoch, index), not by this.
    return DataLoader(dataset, batch_size=None, sampler=keys, num_workers=workers,
                      pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None,
                      generator=torch.Generator().manual_seed(0))


TRAIN_FIELDS = (("roll", 0.0), ("beat", IGNORE_INDEX), ("bar", IGNORE_INDEX), ("mask", False),
                ("bpm", 120.0), ("hint_valid", False))


def pad_batch(items):
    """Stack songs into one batch, padding at the end. The TCN is causal, so
    trailing padding cannot change any real frame's output; it is masked out."""
    length = max(len(item["roll"]) for item in items)
    batch = {}
    for name, fill in TRAIN_FIELDS:
        first = items[0][name]
        out = torch.full((len(items), length, *first.shape[1:]), fill, dtype=first.dtype)
        for k, item in enumerate(items):
            out[k, :len(item[name])] = item[name]
        batch[name] = out
    return batch


def train_loader(dataset, keys, batch_size, workers, device):
    """Batches of songs (collated in the workers); same private generator as ``loader``."""
    return DataLoader(dataset, batch_size=batch_size, sampler=keys, collate_fn=pad_batch, num_workers=workers,
                      pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None,
                      generator=torch.Generator().manual_seed(0))


def scheduled_lr(step, peak, warmup, schedule, total, minimum=0.0):
    """Learning rate for optimizer step ``step``: linear warm-up to ``peak``, then
    either cosine decay to ``minimum`` at ``total`` steps (``cosine``) or None
    (``plateau``: ReduceLROnPlateau keeps control after the warm-up)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "cosine":
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
    return None


def forward_batch(model, batch, device):
    to = lambda name: batch[name].to(device, non_blocking=True)
    outputs = model(to("roll"), bpm_hint=to("bpm"), hint_valid=to("hint_valid"))
    return outputs, to("beat"), to("bar"), to("mask")


def forward(model, item, device):
    to = lambda name: item[name].to(device, non_blocking=True).unsqueeze(0)
    outputs = model(to("roll"), bpm_hint=to("bpm"), hint_valid=to("hint_valid"))
    return outputs, to("beat"), to("bar"), to("mask"), to("hint_valid")


@torch.no_grad()
def validate(model, dataset, workers, device, sample=()):
    """Frame-weighted loss and accuracy over the validation groups, plus beat and
    downbeat tracking metrics (mir_eval, averaged per song) from the decoded
    phase outputs; everything also split by hint given / withheld and by meter
    (4/4 / other). Returns ``(metrics, examples)``: ``examples`` holds what
    plots.plot_tracking needs for the validation groups at positions ``sample``."""
    model.eval()
    sums, tracking, examples = {}, {}, []
    keys = dataset.fixed_keys()
    for position, item in enumerate(loader(dataset, keys, workers, device)):
        outputs, beat, bar, mask, hint = forward(model, item, device)
        hint_key = "hint" if bool(hint[mask].any()) else "no_hint"
        meter_key = "meter_4_4" if bool(item["four_four"]) else "meter_other"
        song_frames = mask[0]  # labelled frames: the song after the silent lead-in
        decoded = {}
        for task, reference, event in (("beat", "beat_times", "beat"), ("bar", "downbeat_times", "downbeat")):
            probabilities = outputs[f"{task}_logits"][0][song_frames].softmax(-1).float().cpu().numpy()
            estimated = decode_events(expected_phase(probabilities), dataset.timing.frame_rate)
            scores = tracking_metrics(item[reference].numpy(), estimated)
            decoded[event] = (probabilities, estimated, scores)
            for key in ("all", hint_key, meter_key):
                totals = tracking.setdefault(f"{key}/{event}", [defaultdict(float), 0])
                for metric, value in scores.items():
                    totals[0][metric] += value
                totals[1] += 1
        if position in sample:
            index, song = keys[position][1], item["mask"]
            examples.append({
                "title": f"{dataset.ids[index][:12]}   meter {dataset.meters[index]}   "
                         f"BPM hint {'given' if hint_key == 'hint' else 'withheld'}",
                "roll": item["roll"][song].numpy(),
                "beat_probs": decoded["beat"][0], "bar_probs": decoded["downbeat"][0],
                "beat_labels": item["beat"][song].numpy(), "bar_labels": item["bar"][song].numpy(),
                "true_beats": item["beat_times"].numpy(), "true_downbeats": item["downbeat_times"].numpy(),
                "pred_beats": decoded["beat"][1], "pred_downbeats": decoded["downbeat"][1],
                "scores": {"beat": decoded["beat"][2], "downbeat": decoded["downbeat"][2]},
            })
        for task, target in (("beat", beat), ("bar", bar)):
            logits, labels = outputs[f"{task}_logits"][mask], target[mask]
            bins = logits.shape[-1]
            error = (logits.argmax(-1) - labels).remainder(bins)
            near = torch.minimum(error, bins - error) <= 1
            for key in ("all", hint_key, meter_key):
                stats = sums.setdefault(f"{key}/{task}", np.zeros(4))
                stats += [F.cross_entropy(logits, labels, reduction="sum").item(), len(labels),
                          (error == 0).sum().item(), near.sum().item()]
    result = {}  # plain floats: checkpoints are loaded with weights_only=True
    for name, (loss, count, exact, near) in sums.items():
        result[f"{name}_loss"] = float(loss / count)
        result[f"{name}_acc"] = float(exact / count)
        result[f"{name}_acc_within1"] = float(near / count)
    for name, (totals, songs) in tracking.items():
        for metric, total in totals.items():
            result[f"{name}_{metric}"] = float(total / songs)
    result["loss"] = result["all/beat_loss"] + result["all/bar_loss"]
    model.train()
    return result, examples


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
    # Oyama's 1e-3 (for a 20-channel TCN) left this 512-channel model at chance
    # with loss spikes; on 2,000 songs 1e-4 learned (val loss 6.69 vs 6.93).
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear learning-rate warm-up from 0 to --lr over the first N optimizer steps; 0 = off")
    parser.add_argument("--schedule", default="plateau", choices=("plateau", "cosine"),
                        help="after warm-up: plateau = halve on --lr-patience epochs without improvement (Oyama); "
                             "cosine = decay to --min-lr by the end of --epochs")
    parser.add_argument("--min-lr", type=float, default=0.0, help="cosine floor")
    parser.add_argument("--batch-size", type=int, default=1, help="songs per optimizer step (1 = Oyama)")
    parser.add_argument("--crop-seconds", type=float, default=0,
                        help="train on a random excerpt of each song this long, with the 6.4 s before it as "
                             "unlabelled context; 0 = whole songs (Oyama). Validation always uses whole songs")
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--stop-patience", type=int, default=50)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-minutes", type=float, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--plot-songs", type=int, default=2,
                        help="validation songs drawn at random each epoch for the predicted-vs-true beat figure")
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
    train = SongExamples(args.data / "train.npz", timing, synth, args.seed, args.limit_train,
                         crop_frames=round(args.crop_seconds * timing.frame_rate))
    steps_per_epoch = math.ceil(len(train.groups) / args.batch_size)
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
        print(json.dumps({"start": True, "device": str(device), "train_groups": len(train.groups),
                          "train_files": len(train), "val_groups": len(val.groups),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "config": timing.to_dict(), "synthesis": synth.to_dict(),
                          "data": manifest.get("packs"), "torch": torch.__version__,
                          "python": platform.python_version()}), flush=True)
    # No dir/id/resume here: on ICE the job chain supplies them through WANDB_* variables.
    wandb.init(project="midi-score-following", config={
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": timing.to_dict(), "synthesis": synth.to_dict(), "data": manifest.get("packs"),
        "data_funnel": manifest.get("funnel"), "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "torch": torch.__version__, "train_groups": len(train.groups),
        "train_files": len(train), "val_groups": len(val.groups),
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
        keys = train.epoch_keys(epoch)[state["next_index"]:]
        window = {"loss": 0.0, "steps": 0, "songs": 0, "frames": 0, "start": time.time()}
        for batch in train_loader(train, keys, args.batch_size, args.workers, device):
            outputs, beat, bar, mask = forward_batch(model, batch, device)
            loss = masked_phase_loss(outputs, beat, bar, mask, mask)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
            lr = scheduled_lr(state["steps"], args.lr, args.warmup_steps, args.schedule,
                              args.epochs * steps_per_epoch, args.min_lr)  # from the saved step count on resume
            if lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.step()
            state["next_index"] += len(batch["roll"])
            state["steps"] += 1
            window["loss"] += loss.item()
            window["steps"] += 1
            window["songs"] += len(batch["roll"])
            window["frames"] += int(mask.sum())
            if window["steps"] == args.log_every:
                elapsed = time.time() - window["start"]
                row = {"epoch": epoch, "song": state["next_index"], "of": len(train.groups),
                       "train_loss": round(window["loss"] / window["steps"], 4),
                       "lr": optimizer.param_groups[0]["lr"],
                       "songs_per_s": round(window["songs"] / elapsed, 2),
                       "frames_per_s": round(window["frames"] / elapsed)}
                print(json.dumps(row), flush=True)
                wandb.log({"train/loss": row["train_loss"], "lr": row["lr"],
                           "train/songs_per_s": row["songs_per_s"], "train/frames_per_s": row["frames_per_s"],
                           "epoch": epoch + state["next_index"] / len(train.groups)}, step=state["steps"])
                window = {"loss": 0.0, "steps": 0, "songs": 0, "frames": 0, "start": time.time()}
            if STOP["signal"]:
                checkpoint(STOP["signal"])
                print(json.dumps({"stopped_by": STOP["signal"], "epoch": epoch, "next_index": state["next_index"]}),
                      flush=True)
                wandb.finish()
                return
            if time.time() - last_save > args.checkpoint_minutes * 60:
                checkpoint("periodic")
                last_save = time.time()

        picks = np.random.default_rng([args.seed, epoch, 7]).choice(
            len(val.groups), size=min(args.plot_songs, len(val.groups)), replace=False)
        metrics, examples = validate(model, val, args.workers, device, sample=set(int(p) for p in picks))
        images = []
        for k, example in enumerate(examples):  # figures must never stop training
            try:
                path = args.out / "plots" / f"epoch{epoch:03d}_song{k}.png"
                path.parent.mkdir(exist_ok=True)
                plot_tracking(example, path, timing.frame_rate)
                images.append(wandb.Image(str(path), caption=example["title"]))
            except Exception as exc:
                print(json.dumps({"plot_failed": repr(exc)}), flush=True)
        if args.schedule == "plateau" and state["steps"] >= args.warmup_steps:
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
                   "epoch": epoch + 1, **({"val/examples": images} if images else {})}, step=state["steps"])
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
