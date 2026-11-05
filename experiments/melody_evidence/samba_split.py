"""The train/test split decided on 2026-11-04 (James): fit on the official JazzSAMBA train songs, report on the official
val + test songs together (no validation set; "for this hmm training we dont need validation").

Song ids are copied from Dataset/SAMBA dataset/{sync,async}_audio/splits/{train,val,test}.txt (version v1, seed 1738,
"Do not reshuffle"). sync = live takes, async = click takes; both takes of a song are on the same side.
"""
from __future__ import annotations

TRAIN = {"sync": {50, 51, 52, 54, 55, 56, 58, 59, 61, 62, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75},
         "async": {0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 34,
                   36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49}}
TEST = {"sync": {53, 57, 60, 63, 66, 67},                       # official val 57 60 67 + test 53 63 66
        "async": {5, 7, 8, 14, 20, 24, 26, 33, 35, 40}}         # official val 5 8 20 24 40 + test 7 14 26 33 35
SPLITS = ("all", "train", "test")


def song_of(performance_id):
    """'sync/66-nardis/alternate' -> ('sync', 66)."""
    protocol, song = performance_id.split("/")[:2]
    return protocol, int(song.split("-")[0])


def in_split(performance_id, split):
    if split == "all":
        return True
    protocol, number = song_of(performance_id)
    return number in {"train": TRAIN, "test": TEST}[split][protocol]
