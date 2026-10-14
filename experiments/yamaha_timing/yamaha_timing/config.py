"""Configuration for a documented Maezawa-inspired timing reconstruction.

The paper states a 20 FPS piano roll, a six-second receptive field, seven
residual TCN layers of "two layers of 1d convolution with dilation 2^i", ELU,
a 512-dimensional feature, and 32 within-measure phase classes.

Each layer follows the TCN layer of the two TCN papers Maezawa cites (Davies &
Boeck 2019; Oyama et al. 2021, Fig. 3): one dilated convolution, ELU, dropout,
and a 1x1 convolution, with a 1x1 convolution on the residual path. With
kernel size 2 and dilations 1..64 the receptive field is 128 frames (6.4 s),
which matches the paper's six seconds; the cited papers' kernel size 5 would
give 25 s. Dropout 0.1 follows both cited papers. The beat-phase class count
is unstated; like Oyama et al., who quantize beat and downbeat phase into the
same number of classes, we use 32 for both cycles (in 4/4 the beat head is
therefore 4x finer in time than the bar head). Head width remains a
reconstruction choice. The rolling streaming window equals the receptive
field, so streaming and whole-sequence outputs agree.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class TimingConfig:
    """Immutable model and streaming settings; ``to_dict`` is JSON serializable."""

    frame_rate: float = 20
    input_pitches: int = 88
    midi_low: int = 21
    context_frames: int = 128
    channels: int = 512
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    kernel_size: int = 2
    head_hidden: int = 128
    beat_bins: int = 32  # positions within a beat (paper: unstated; same count as the bar, as in Oyama et al.)
    bar_bins: int = 32   # positions within a bar (paper: 32 segments)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_rate, bool)
            or not isinstance(self.frame_rate, Real)
            or not isfinite(self.frame_rate)
            or self.frame_rate <= 0
        ):
            raise ValueError("frame_rate must be positive and finite")

        for name in (
            "input_pitches", "context_frames", "channels", "kernel_size",
            "head_hidden", "beat_bins", "bar_bins",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.beat_bins < 2 or self.bar_bins < 2:
            raise ValueError("phase heads must contain at least two bins")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, Real)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if (
            isinstance(self.midi_low, bool)
            or not isinstance(self.midi_low, Integral)
            or self.midi_low < 0
            or self.midi_low + self.input_pitches > 128
        ):
            raise ValueError("midi_low and input_pitches must select valid MIDI keys")

        # JSON loads arrays as lists; normalize to an immutable tuple on input.
        try:
            dilations = tuple(self.dilations)
        except TypeError as exc:
            raise ValueError("dilations must be a nonempty sequence of positive integers") from exc
        if not dilations or any(
            isinstance(dilation, bool)
            or not isinstance(dilation, Integral)
            or dilation <= 0
            for dilation in dilations
        ):
            raise ValueError("dilations must be a nonempty sequence of positive integers")
        object.__setattr__(self, "dilations", dilations)

    @property
    def receptive_field_frames(self) -> int:
        """TCN history including the current frame: one dilated conv per layer; defaults to 128."""
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    @property
    def context_seconds(self) -> float:
        """Duration of the rolling streaming input window."""
        return self.context_frames / self.frame_rate

    def to_dict(self) -> dict[str, Any]:
        """Return dataclass fields suitable for ``json.dumps`` and reconstruction."""
        return asdict(self)
