# Yamaha (Maezawa 2023) lead-sheet follower with the timing network

Integrates the downbeat-phase network from `../yamaha_timing` into the score
follower of `../ensemble_benchmark` (left unchanged; imported read-only) and
evaluates score following on JazzSAMBA with the recorded piano MIDI as input.

- `follower_data.py`: SAMBA takes in the benchmark's format (unfolded chord
  plan, 100 ms grid, truth positions), from the local annotation CSVs; equal to
  `ensemble_benchmark/common.load_performance` on all 124 takes it covers.
- `evidence.py`: the network's bar-phase posterior as p_d(s) for the grid
  filter's rhythm hook (causal: the last network frame at or before each update).
- `run_follow.py`: Stage A. The g00 harmonic follower with no rhythm evidence,
  the rule-based onset cue, or the network's p_d(s); 8 or 32 positions per bar.
  Bar/beat accuracy, errors, lag in ms, compute per update.

Async takes were played to a click at the catalog tempo, so a constant-tempo
clock already follows them perfectly; they check that a follower does not
wander off a steady beat. Sync takes (live band) are the real test.
