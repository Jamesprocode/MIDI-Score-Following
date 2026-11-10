"""The one place this package reaches into the older experiment folders, which it imports and never edits: the take
loader (yamaha_follower/follower_data.py), B5's chord features (ensemble_benchmark), the timing network
(yamaha_follower/evidence.py, yamaha_timing), the melody matchers (melody_evidence/) and the piano-role labels
(adaptive_weights/sweep_fixed_weights.py). Importing this module puts those folders on sys.path.

The horn and bass parts come from SAMBA_CLEANED_MIDI when it is set (yamaha_timing/real_data.py reads it at import;
takes.use_horns sets it).
"""
from __future__ import annotations

from pathlib import Path
import sys

EXPERIMENTS = Path(__file__).resolve().parents[1]
for sub in ("melody_evidence", "yamaha_follower"):
    if str(EXPERIMENTS / sub) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS / sub))

if str(EXPERIMENTS / "adaptive_weights") not in sys.path:  # piano_roles (sweep_fixed_weights); last, no name clashes
    sys.path.append(str(EXPERIMENTS / "adaptive_weights"))

import follower_data  # noqa: E402,F401  (also puts yamaha_timing, ensemble_benchmark and harmony_pilot on sys.path)
