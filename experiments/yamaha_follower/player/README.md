# Follower replay player

A copy of `../../ground_truth_player` (unchanged there) that replays four
followers on the recorded piano MIDI of selected JazzSAMBA takes: harmony only,
harmony + rule-based onset cue, harmony + the timing network's downbeat phase
(Stage A), and the paper version (Stage B). The menus come from
`recordings.json`; per-take data (SAMBA notes, gitignored) is rebuilt with:

```sh
../../yamaha_timing/.venv/bin/python build_player.py --samba "../../../Dataset/SAMBA dataset" --checkpoint path/to/best_model.pt
python3 -m http.server 8770
```

then open http://localhost:8770 (the takes are listed in `TAKES` in build_player.py).
Blue: annotated position. Orange: the selected follower. Green: a fixed clock at
the written tempo. All followers are causal and see only piano notes.
