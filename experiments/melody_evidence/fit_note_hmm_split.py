"""Melody cue v2 parameters for the train/test split (samba_split.py): one fit on the heads of the train songs (60 songs,
120 takes, live and click together), used for every take. Results are then reported on the test songs only.

Writes <out>/fit_train.json and <out>/assign.json (every song -> fit_train.json; NoteHMMEvidence reads it). The fit is
note_hmm.fit with its defaults (6 Viterbi passes, add-one counts), the same as the cross-fit.

  SAMBA_CLEANED_MIDI=../../Dataset/horn_sets/crepe_all ../yamaha_timing/.venv/bin/python fit_note_hmm_split.py \\
      --samba "../../Dataset/SAMBA dataset" --lead-sheets ../../Dataset/JazzSAMBA-MIDI --out params/crepe_split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import note_hmm
import follower_data
from samba_split import in_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--lead-sheets", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    takes = [(d, None) for d, _ in follower_data.samba_follow_takes(args.samba, protocols=("sync", "async"))]
    train = [t for t in takes if in_split(t[0]["performance_id"], "train")]
    songs = {"/".join(t[0]["performance_id"].split("/")[:2]) for t in train}
    print(json.dumps({"takes": len(takes), "train_takes": len(train), "train_songs": len(songs)}), flush=True)
    assert len(songs) == 60 and len(train) == 120, "the split should give 60 train songs, 120 takes"
    params = note_hmm.fit(train, args.samba, args.lead_sheets, args.iterations, log=lambda s: print(s, flush=True))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fit_train.json").write_text(json.dumps(params, indent=1))
    keys = sorted({"/".join(t[0]["performance_id"].split("/")[:2]) for t in takes})
    (args.out / "assign.json").write_text(json.dumps({key: "fit_train.json" for key in keys}, indent=1))
    print(json.dumps(params["measured"], indent=1), flush=True)


if __name__ == "__main__":
    main()
