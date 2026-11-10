"""The takes a run follows, the horn set they are read with, and the three time grids, in one place.

Time grids (the old runners each built these themselves):
  data grid     data["times"], every 0.1 s: the plan and the truth (follower_data, from the benchmark)
  update grid   every STEP = 0.05 s from the first data instant: when the follower updates (run_follow_b.STEP)
  check grid    every second data instant (0.2 s): where a run is scored, and what the frame files keep

Horn set: the horn and bass parts come from a folder of cleaned MIDI (Dataset/horn_sets/...) or, with "raw", as the
dataset distributes them. yamaha_timing/real_data.py reads the choice from SAMBA_CLEANED_MIDI once, at import, so
``use_horns`` must run before anything loads a take. Run.py takes it as a required argument and records it.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

STEP = 0.05         # seconds between follower updates
CHECK_EVERY = 2     # data instants per check (0.2 s)


def use_horns(horns):
    """Read horns and bass from ``horns`` (a horn-set folder) or as distributed (``"raw"``). Returns the value
    recorded in run.json. Refuses to change a choice the loader has already read."""
    value = "" if str(horns) == "raw" else str(Path(horns).resolve())
    if value and not Path(value).is_dir():
        raise FileNotFoundError(f"horn set {value} does not exist")
    loaded = sys.modules.get("yamaha_timing.real_data")
    if loaded is not None and (loaded.CLEANED_MIDI or "") != value:
        raise RuntimeError(f"the take loader already read SAMBA_CLEANED_MIDI={loaded.CLEANED_MIDI!r}; "
                           f"set the horn set before loading takes")
    current = os.environ.get("SAMBA_CLEANED_MIDI", "")
    if current and str(Path(current).resolve()) != value:
        raise RuntimeError(f"SAMBA_CLEANED_MIDI={current} in the environment disagrees with --horns {horns}")
    if value:
        os.environ["SAMBA_CLEANED_MIDI"] = value
    else:
        os.environ.pop("SAMBA_CLEANED_MIDI", None)
    return value or "raw"


def load(samba, protocols=("sync", "async"), split="all", only=()):
    """(data, take) per SAMBA take of ``protocols`` in ``split`` (samba_split.py: all, train or test), optionally only
    the performance ids in ``only``. Call ``use_horns`` first."""
    from . import legacy  # noqa: F401
    import follower_data
    from samba_split import in_split
    takes = [t for t in follower_data.samba_follow_takes(samba, protocols=tuple(protocols))
             if in_split(t[0]["performance_id"], split)]
    if only:
        missing = set(only) - {d["performance_id"] for d, _ in takes}
        if missing:
            raise ValueError(f"not among the loaded takes: {sorted(missing)}")
        takes = [t for t in takes if t[0]["performance_id"] in set(only)]
    return takes


def update_times(data):
    """The follower's update times for a whole take."""
    return np.round(np.arange(data["times"][0], data["times"][-1] + 1e-9, STEP), 6)


def check_frames(data):
    """Indices into data["times"] of the checks."""
    return np.arange(0, len(data["times"]), CHECK_EVERY)


def update_index(times, check_times):
    """For each check time, the index of the same instant in ``times`` (an update grid, or a slice of one)."""
    return np.round((np.asarray(check_times) - times[0]) / STEP).astype(int)


def played_bpm(data):
    """The start tempo every run uses: the median annotated bar length as quarter notes per minute (oracle)."""
    from . import legacy  # noqa: F401
    from run_follow_b import played_bpm as old
    return old(data)


def head_stretches(samba, performance_id, times):
    """(first, last + 1) indices into ``times`` of each run of head instants (sections.csv is_head; heads that follow
    each other are one stretch), as run_melody_heads.py."""
    from . import legacy  # noqa: F401
    from melody_evidence import section_kinds
    head = section_kinds(samba, performance_id, times) == "head"
    edges = np.flatnonzero(np.diff(np.concatenate([[0], head.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def labels(samba, performance_id, times):
    """Per check: the piano's role (sweep_fixed_weights.piano_roles) and the section kind
    (melody_evidence.section_kinds), as the old frame files store them. Step 3 merges these into one labelling."""
    from . import legacy  # noqa: F401
    from melody_evidence import section_kinds
    from sweep_fixed_weights import piano_roles
    return (piano_roles(samba, performance_id, times).astype(str),
            section_kinds(samba, performance_id, times).astype(str))
