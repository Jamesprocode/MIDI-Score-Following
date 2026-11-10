"""The plug-in interface: what a cue must give the HMM follower.

A cue listens to some of the band (the piano's chords, the piano's rhythm, the horns' melody) and, at each update,
says how well what it heard fits each place on the chart:

    log_likelihood(now, bpm, positions) -> array, one value per place in ``positions``, or None (no opinion now)

  now        the update time, seconds from the start of the recording; only input up to ``now`` may be used
  bpm        the follower's fed-back tempo (its BPM hint), for cues that turn seconds into beats
  positions  the places, in beats from the top of the unfolded chart (the follower's grid; it can grow)

The value is log P(what the cue heard | the band is at that place), up to a constant: only differences between places
matter, and the follower scales it (scaling.py) and weights it. Three plug-ins exist, one per file: harmony.py,
rhythm.py, melody.py.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class Cue(Protocol):
    name: str

    def log_likelihood(self, now: float, bpm: float, positions: np.ndarray) -> Optional[np.ndarray]:
        ...
