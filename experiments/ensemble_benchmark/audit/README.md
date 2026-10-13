# Local MIDI and audio-header audit

Snapshot: 2026-10-05. Source data were read without modification. Rerun `python3 audit_midi.py` to regenerate the JSON/CSV evidence in this directory. No package installation is required.

## Availability

| Local collection | Songs | Preferred/alternate groups | Core MIDI | Extra MIDI | Instruments |
|---|---:|---:|---:|---:|---|
| async | 49 | 98 | 294 | 48 | bass, trumpet, alto or tenor saxophone |
| sync | 26 | 52 | 182 | 0 | bass, drums, trumpet and/or alto saxophone |

The collections contain 75 distinct song directories and 524 MIDI files. There are 424 pitched core files, 52 drum core files, and 48 extra alto files. Async song ID 11 is absent locally. Sync IDs run from 50 through 75. Preferred and alternate are separate take groups; extras are additional instrument takes, not an independent complete ensemble.

Sync contains 13 songs with bass, drums, alto, and trumpet; 10 with bass, drums, and trumpet; and 3 with bass, drums, and alto. Counts describe local availability, not the full release. No local piano MIDI exists. The sync folder includes piano-left/right audio and 390 FLAC files total; audio does not supply the missing symbolic piano track.

All MIDI filenames identify transcription-derived exports. Every file is SMF type 1, 480 PPQ, with one metadata track and one instrument note track. Pitched files use zero-based channel 0; drums use zero-based channel 9 (MIDI channel 10). Each note-on velocity is 100. There are no sustain/controller/pitch-bend messages. Preserve instrument identity from filenames when combining streams.

## Parsing and note quality

All 524 files parse successfully: 347,589 complete notes, no unmatched note-offs, dangling note-ons, zero-duration notes, or byte-identical duplicate MIDI files.

Valid MIDI syntax does not establish a correct transcription. The fixed diagnostic flags are: pitched-file maximum polyphony greater than 4, any note longer than 10 seconds, or a pitch outside MIDI 21–108. They flag 71 files, including 63 core files across 51 core take groups. These thresholds were used only to identify files for inspection; no filtering has been applied. Drums are excluded from these pitched-instrument flags.

There are 19 async and 9 sync files with more than 16 simultaneous notes. For example, sync `57-chega_de_saudade/preferred/trumpet` contains 7,957 notes, reaches 65 simultaneous notes, and has a note lasting 365.10 seconds. Such artifacts can substantially affect instantaneous activity and event-partition weighting. Full per-file evidence is in `midi_files.json`; flagged paths are in `quality_flags.csv`.

## Timing

Each MIDI has exactly one tempo event at tick zero. Tempo maps agree across instruments within every local take group. Use these maps to convert stored tick values to seconds; their constancy does not prove constant performed tempo.

All 524 MIDI meter tags say 4/4. Catalog meter disagrees for seven songs: Alice in Wonderland, My Favorite Things, Take Five, The Great Pumpkin Waltz, Wait Till You See Her, Some Day My Prince Will Come, and Waltz for Debby. Do not use the export's time-signature tag as meter ground truth. The available sync annotation bar durations vary despite constant MIDI tempo metadata. `annotation_timing.csv` records the comparison for all 75 songs; the rows do not independently verify audio alignment.

All 390 local FLAC files are debleeded derived overlays (`*.debleeded.derived.flac`), not the original release mixtures/stems. Their headers report 48 kHz, 16-bit samples throughout. Within every sync take group, all non-mixture instrument overlays have exactly equal sample lengths. Debleeded mixture files are longer in 38 of 52 groups; the maximum difference is 5.504 seconds. These duration differences do not establish a mismatch between original release audio and annotations, and equal stem lengths do not verify shared content alignment. MIDI endpoints follow the final note-offs and vary between instruments, so neither first-note alignment nor rescaling to equal MIDI durations is justified. See [timing_provenance.md](timing_provenance.md) for verified official loader/demo behavior and the unresolved MIDI-to-annotation origin check.

## Common benchmark policy recommended

1. Exclude drum files/channel 10 from every harmony input. Keep them available only for a separately specified rhythm experiment. Do not identify percussion from program number alone.
2. Keep the raw valid note data for the primary benchmark and retain the same source subset and timestamp masks for every method. Report the quality flags and, if desired, a separately labeled sensitivity result using one fixed clean-group mask for all compared methods. Do not select filters from achieved chord accuracy.
3. Current files require no malformed-note repair. If future inputs fail structural parsing or contain unpaired events, record the exact failure and quarantine that source/group consistently across methods; do not silently close notes at annotation boundaries. Any later repair policy must be explicit and identical across methods.
4. Keep original time origins. Use annotation seconds for evaluation, not MIDI beat-number interpretation, event realignment, or fitted shifts. In particular, the documented copied async grid does not by itself justify applying preferred annotations to sync alternate performances.
5. The Maezawa-style 88-key representation excludes pitches outside MIDI 21–108. There are 144 such notes across the local corpus. Disclose this representational difference or run a common-range sensitivity condition; do not silently apply different data cleaning to competing scorers.
6. Treat preferred and alternate takes of the same composition as grouped data when splitting future training/evaluation sets. Preserve async/sync protocol labels and report their results separately.

## Evidence files

- `inventory.json`: local files, instruments, and song directories.
- `song_take_groups.json`: exact per-group inputs, tempo consistency, and endpoints.
- `midi_files.json`: hashes, tracks, messages, note statistics, and representative events.
- `quality_flags.csv`: fixed diagnostic flags, without exclusions.
- `audio_headers.json`: FLAC stream-header metadata only; audio was not decoded or listened to.
- `annotation_timing.csv`: catalog/MIDI timing metadata and reference bar-duration summaries.
- `summary.json`: compact counts and audit outcomes.
