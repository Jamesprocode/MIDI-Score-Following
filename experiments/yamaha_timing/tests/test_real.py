"""Real-data loaders and the evaluation's phase labels."""
import tempfile
from pathlib import Path
import unittest

import mido
import numpy as np

from yamaha_timing.real_data import beat_concentration, read_midi_notes, to_denominator_beats
from eval_real import phase_labels  # discovery runs from the experiment folder
from samba_sync_timing import RATE, fit_shift, onset_train


class RealDataTests(unittest.TestCase):
    def test_compound_beats_are_split_and_simple_kept(self):
        beats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        six_eight = to_denominator_beats(beats, [(0.0, "6/8", 2)])
        np.testing.assert_allclose(six_eight[:7], [0, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2])
        self.assertEqual(len(six_eight), 5 * 3 + 1)  # the last beat has no successor to split toward
        np.testing.assert_allclose(to_denominator_beats(beats, [(0.0, "3/4", 3)]), beats)
        mixed = to_denominator_beats(beats, [(0.0, "2/4", 2), (3.0, "6/8", 2)])
        np.testing.assert_allclose(mixed[:5], [0, 1, 2, 3, 10 / 3])

    def test_phase_labels(self):
        labels = phase_labels(np.array([-0.5, 0.0, 0.25, 0.99, 1.5, 2.5]), np.array([0.0, 1.0, 2.0]), 4, end=2.0)
        np.testing.assert_array_equal(labels, [-1, 0, 1, 3, 2, -1])

    def test_audio_shift_fit_and_beat_concentration(self):
        rng = np.random.default_rng(0)
        onsets = np.sort(rng.uniform(5, 115, 600))  # an irregular rhythm: a unique alignment
        envelope = onset_train(onsets, 120 * RATE) + rng.uniform(0, 0.3, 120 * RATE)
        shift, confidence = fit_shift(envelope, onsets + 3.2, max_shift=15)  # MIDI 3.2 s late
        self.assertAlmostEqual(shift, -3.2, delta=0.011)
        self.assertGreater(confidence, 1.5)
        beats = np.arange(0, 100, 0.5)
        self.assertGreater(beat_concentration(beats[2:-2] + 0.01, beats), 0.9)
        self.assertLess(beat_concentration(np.sort(rng.uniform(1, 99, 2000)), beats), 0.1)

    def test_midi_notes_held_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track += [mido.MetaMessage("set_tempo", tempo=500000, time=0),
                      mido.Message("note_on", note=60, velocity=80, time=0),
                      mido.Message("note_on", note=64, velocity=80, time=480),
                      mido.Message("note_on", note=60, velocity=0, time=480),
                      mido.Message("note_off", note=64, velocity=0, time=480),
                      mido.Message("note_on", note=10, velocity=80, time=0)]  # below the piano: dropped
            midi.tracks.append(track)
            midi.save(path)
            onsets, offsets, pitches = read_midi_notes(path)
        np.testing.assert_allclose(onsets, [0.0, 0.5])
        np.testing.assert_allclose(offsets, [1.0, 1.5])
        np.testing.assert_array_equal(pitches, [60, 64])


if __name__ == "__main__":
    unittest.main()
