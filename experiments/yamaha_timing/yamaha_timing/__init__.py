"""Untrained Maezawa-inspired causal MIDI timing framework."""

from .config import TimingConfig
from .model import YamahaTimingTCN

__all__ = ["TimingConfig", "YamahaTimingTCN"]
