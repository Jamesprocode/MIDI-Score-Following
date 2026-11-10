"""Melody plug-in: a matcher listens to the horns; the cue scores how well their recent notes fit the tune at each place.

A matcher is anything with ``log_likelihood(now, bpm, n_places)`` over the first ``n_places`` places of the grid, or
None: melody_evidence.MelodyEvidence (v1) and note_hmm.NoteHMMEvidence (v2) already have it.
"""
from __future__ import annotations


class MelodyCue:
    name = "melody"

    def __init__(self, matcher):
        self.matcher = matcher

    def log_likelihood(self, now, bpm, positions):
        return self.matcher.log_likelihood(now, bpm, len(positions))


def melody_v2(data, samba, params, lead_sheets=None, combine="sum"):
    """Melody v2, the note HMM (note_hmm.py) with the lead-sheet tune. ``params``: a folder of fitted tables; the matcher
    picks the file for this take from it (assign.json, else the other protocol's fit) and keeps it in ``params_file``."""
    from .. import legacy  # noqa: F401  (melody_evidence on sys.path)
    from note_hmm import NoteHMMConfig, NoteHMMEvidence
    config = NoteHMMConfig(params=str(params), combine=combine,
                           **({"lead_sheets": str(lead_sheets)} if lead_sheets else {}))
    return MelodyCue(NoteHMMEvidence(data, samba, config))


def melody_v1(data, samba, lead_sheets=None):
    """Melody v1 (melody_evidence.py) with the lead-sheet tune."""
    from .. import legacy  # noqa: F401
    from melody_evidence import MelodyConfig, MelodyEvidence
    config = MelodyConfig(reference="written", **({"lead_sheets": str(lead_sheets)} if lead_sheets else {}))
    return MelodyCue(MelodyEvidence(data, samba, config))
