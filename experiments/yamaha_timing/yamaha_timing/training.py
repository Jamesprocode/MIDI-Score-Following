"""Proposed phase-classification loss and reproducible checkpoint helpers.

Cross entropy and equal loss weights are reconstruction choices. Maezawa's
2023 demonstration does not specify its loss. Reference timing belongs in
targets, never in the streaming inputs or inferred BPM condition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import TimingConfig
from .model import YamahaTimingTCN

# 2: cited-paper TCN layer (dilated conv + 1x1 conv, 1x1 residual) and dropout.
# 3: no separate input projection; the first TCN layer takes the 88-key roll.
CHECKPOINT_SCHEMA = 3


def masked_phase_loss(outputs: dict[str, torch.Tensor],
                      beat_targets: torch.Tensor, bar_targets: torch.Tensor,
                      beat_mask: torch.Tensor | None = None,
                      bar_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Return equal-weight mean classification losses over valid [B,T] targets.

    Ambiguous/missing labels and synthetic startup frames can be masked per
    task. An entirely masked task contributes a differentiable zero; an entirely
    masked batch is rejected so it cannot masquerade as successful training.
    """
    losses: dict[str, torch.Tensor] = {}
    valid_count = 0
    for task, targets, mask in (("beat", beat_targets, beat_mask),
                                ("bar", bar_targets, bar_mask)):
        logits = outputs[f"{task}_logits"]
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError(f"{task} targets must match logits [B,T]")
        if targets.dtype != torch.long or targets.device != logits.device:
            raise ValueError(f"{task} targets must be torch.long on the logits device")
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        if mask.shape != targets.shape or mask.dtype != torch.bool or mask.device != logits.device:
            raise ValueError(f"{task} mask must be bool [B,T] on the logits device")
        selected = targets[mask]
        count = int(selected.numel())
        valid_count += count
        if count and ((selected < 0).any() or (selected >= logits.shape[-1]).any()):
            raise ValueError(f"{task} targets outside the phase-bin range")
        losses[task] = F.cross_entropy(logits[mask], selected) if count else logits.sum() * 0.0
    if not valid_count:
        raise ValueError("at least one phase target must be valid")
    losses["total"] = losses["beat"] + losses["bar"]
    return losses


def save_checkpoint(path: str | Path, model: YamahaTimingTCN,
                    provenance: dict[str, Any] | None = None) -> None:
    """Save config and CPU tensors; annotate synthetic checkpoints explicitly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "provenance": provenance or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[YamahaTimingTCN, dict[str, Any]]:
    """Load a tensor-only baseline checkpoint, never arbitrary pickled classes."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported timing checkpoint schema {payload.get('schema_version') if isinstance(payload, dict) else None}; "
            f"expected {CHECKPOINT_SCHEMA} (earlier schemas used a different layer layout)"
        )
    config = TimingConfig(**payload["config"])
    model = YamahaTimingTCN(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload.get("provenance", {})
