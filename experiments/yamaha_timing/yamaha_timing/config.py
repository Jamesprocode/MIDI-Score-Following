"""Configuration for a documented Maezawa-inspired timing reconstruction.

The paper states a 20 FPS piano roll, seven residual TCN stages, a
512-dimensional feature, and 32 within-measure phase classes. Kernel size,
head width, causal padding, and beat-phase class count are reconstruction
choices. A six-second rolling input is distinct from the theoretical TCN
receptive field: the default two kernel-2 convolutions per stage see at most
255 frames if a longer input is supplied.
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
    context_frames: int = 120
    channels: int = 512
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    kernel_size: int = 2
    head_hidden: int = 128
    beat_bins: int = 32
    bar_bins: int = 32

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
        """Maximum TCN history, including the current frame; defaults to 255."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    @property
    def context_seconds(self) -> float:
        """Duration of the independently configured rolling input window."""
        return self.context_frames / self.frame_rate

    def to_dict(self) -> dict[str, Any]:
        """Return dataclass fields suitable for ``json.dumps`` and reconstruction."""
        return asdict(self)
