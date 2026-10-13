"""Causal MIDI phase network, not an absolute-position or tempo estimator.

This reconstructs the timing module described in Maezawa (2023), rather than
the complete Yamaha accompaniment system. It has no HHSMM, harmonic matcher,
chart position, or trained checkpoint. The streaming wrapper selects the
rolling context; ``forward`` processes the sequence it receives without
silently truncating it.
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import TimingConfig


class _CausalConv1d(nn.Module):
    """Left padding preserves length without observing later frames."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.conv(F.pad(features, (self.left_padding, 0)))


class _ResidualTCNBlock(nn.Module):
    """Dilated causal conv, ELU, dropout, 1x1 conv; 1x1 conv on the residual path.

    Layout of the TCN layer in Oyama et al. (2021, Fig. 3), the phase-based
    tracker Maezawa cites; no activation follows the residual sum. Causal left
    padding replaces the cited papers' offline (non-causal) convolution.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilated = _CausalConv1d(channels, kernel_size, dilation)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.residual = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        update = self.pointwise(self.dropout(self.activation(self.dilated(features))))
        return self.residual(features) + update


def _phase_head(channels: int, hidden: int, bins: int, dropout: float) -> nn.Sequential:
    """Oyama et al.'s decoder (dropout, dense, ELU, dropout, dense) as 1x1 convs.

    Softmax is applied by the loss or ``probabilities``; the hidden width is
    unspecified in all three papers.
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv1d(channels, hidden, kernel_size=1),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Conv1d(hidden, bins, kernel_size=1),
    )


class YamahaTimingTCN(nn.Module):
    """Predict beat and within-measure phase logits for every received frame.

    Input is floating ``[batch, frames, input_pitches]`` piano roll. Optional
    BPM and boolean availability accept a scalar, ``[batch]``, or
    ``[batch, frames]``. Missing/disabled BPM bypasses the tempo gate. Enabled
    hints must be positive and finite; disabled values may be placeholders.

    Outputs are ``beat_logits`` and ``bar_logits`` of shape
    ``[batch, frames, respective_bins]``. The paper specifies 32 bar bins;
    the default 32 beat bins is our choice. An untrained model's distributions
    have no demonstrated tracking accuracy or calibrated confidence.
    """

    def __init__(self, config: TimingConfig | None = None) -> None:
        super().__init__()
        self.config = TimingConfig() if config is None else config
        if not isinstance(self.config, TimingConfig):
            raise TypeError("config must be a TimingConfig")
        cfg = self.config
        self.input_projection = nn.Conv1d(cfg.input_pitches, cfg.channels, kernel_size=1)
        self.backbone = nn.Sequential(*(
            _ResidualTCNBlock(cfg.channels, cfg.kernel_size, dilation, cfg.dropout)
            for dilation in cfg.dilations
        ))
        self.tempo_projection = nn.Linear(8, cfg.channels)
        self.register_buffer(
            "tempo_centers", 30 * 2 ** torch.arange(8, dtype=torch.float64),
        )
        self.beat_head = _phase_head(cfg.channels, cfg.head_hidden, cfg.beat_bins, cfg.dropout)
        self.bar_head = _phase_head(cfg.channels, cfg.head_hidden, cfg.bar_bins, cfg.dropout)

    @staticmethod
    def _expand_hint(value: Any, batch: int, frames: int, name: str, device: torch.device) -> Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 0:
            return tensor.expand(batch, frames)
        if tensor.ndim == 1 and tensor.shape[0] == batch:
            return tensor[:, None].expand(batch, frames)
        if tensor.ndim == 2 and tensor.shape == (batch, frames):
            return tensor
        raise ValueError(f"{name} must be scalar, [batch], or [batch, frames]")

    def _condition(self, features: Tensor, bpm_hint: Any, hint_valid: Any) -> Tensor:
        if bpm_hint is None:
            return features
        batch, _, frames = features.shape
        bpm = self._expand_hint(bpm_hint, batch, frames, "bpm_hint", features.device)
        if bpm.dtype == torch.bool or bpm.is_complex():
            raise TypeError("bpm_hint must contain real numeric values")
        bpm = bpm.to(dtype=torch.float64)
        if hint_valid is None:
            valid = torch.ones((batch, frames), dtype=torch.bool, device=features.device)
        else:
            valid = self._expand_hint(hint_valid, batch, frames, "hint_valid", features.device)
            if valid.dtype != torch.bool:
                raise TypeError("hint_valid must contain boolean values")
        if not bool((torch.isfinite(bpm[valid]) & (bpm[valid] > 0)).all()):
            raise ValueError("enabled BPM hints must be positive and finite")

        # Invalid placeholders never enter the encoding. The following logits
        # equal -(b - center)^2 up to a shared constant. Removing b^2 avoids
        # underflow and cancellation; the final center has logit zero even for
        # extremely large positive hints. No new bandwidth is introduced.
        bpm = torch.where(valid, bpm, torch.full_like(bpm, 120.0))
        centers = self.tempo_centers.to(dtype=torch.float64)
        reference = centers[-1]
        encoding_logits = (
            bpm[..., None] * (2 * (centers - reference))
            + (reference.square() - centers.square())
        )
        encoded = F.softmax(encoding_logits, dim=-1).to(dtype=features.dtype)
        gate = torch.sigmoid(self.tempo_projection(encoded))
        gate = torch.where(valid[..., None], gate, torch.ones_like(gate))
        return features * gate.transpose(1, 2)

    def forward(
        self,
        piano_roll: Tensor,
        bpm_hint: Any = None,
        hint_valid: Any = None,
    ) -> dict[str, Tensor]:
        if not isinstance(piano_roll, Tensor):
            raise TypeError("piano_roll must be a torch.Tensor")
        if piano_roll.ndim != 3 or piano_roll.shape[-1] != self.config.input_pitches:
            raise ValueError("piano_roll must have shape [batch, frames, input_pitches]")
        if piano_roll.shape[0] == 0 or piano_roll.shape[1] == 0:
            raise ValueError("piano_roll must contain a batch and at least one frame")
        if not piano_roll.is_floating_point():
            raise TypeError("piano_roll must have a floating dtype")

        features = self.backbone(self.input_projection(piano_roll.transpose(1, 2)))
        features = self._condition(features, bpm_hint, hint_valid)
        return {
            "beat_logits": self.beat_head(features).transpose(1, 2),
            "bar_logits": self.bar_head(features).transpose(1, 2),
        }

    def probabilities(
        self,
        piano_roll: Tensor,
        bpm_hint: Any = None,
        hint_valid: Any = None,
    ) -> dict[str, Tensor]:
        """Normalize logits without disabling gradients or changing training mode."""
        outputs = self(piano_roll, bpm_hint=bpm_hint, hint_valid=hint_valid)
        return {
            "beat_probs": F.softmax(outputs["beat_logits"], dim=-1),
            "bar_probs": F.softmax(outputs["bar_logits"], dim=-1),
        }
