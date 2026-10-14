"""Lakh preparation and synthesis checks: label alignment, live-input parity, filters."""
import tempfile
from pathlib import Path
import unittest

import mido
import numpy as np

from yamaha_timing.config import TimingConfig
from yamaha_timing.lakh import (
    IGNORE_INDEX, SongPack, SynthesisConfig, bar_and_beat_phase, check_score, crop_example, meter_segments,
    rasterize, read_score, render_example, save_pack, score_to_song,
)
from yamaha_timing.streaming import MidiFrameRasterizer

TIMING = TimingConfig(channels=16, head_hidden=8, dilations=(1, 2, 4), context_frames=8)
NO_AUGMENTATION = SynthesisConfig(max_delete=0, max_insert=0, drift_step_max=0)


def song_on_beats(beats=64, bpm=120.0, parts=1, meter=((0.0, 4, 4),)):
    """One note per quarter note per part, each half a quarter long, at a constant tempo."""
    starts = np.repeat(np.arange(beats, dtype=np.float32), parts)
    return {
        "start": starts, "end": starts + 0.5,
        "pitch": (60 + np.tile(np.arange(parts), beats)).astype(np.uint8),
        "part": np.tile(np.arange(parts), beats).astype(np.uint16),
        "tempo_beat": np.array([0.0], dtype=np.float32), "tempo_bpm": np.array([bpm], dtype=np.float32),
        "meter_beat": np.array([m[0] for m in meter], dtype=np.float64),
        "meter_num": np.array([m[1] for m in meter], dtype=np.uint8),
        "meter_den": np.array([m[2] for m in meter], dtype=np.uint8),
    }


def write_midi(path, meter=(4, 4), bpm=120, bars=40, onset_jitter_ticks=0, random_onsets=False,
               snare_beats=(1, 3), signature=True, extra_signatures=(), seed=0):
    rng = np.random.default_rng(seed)
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    signatures = ([(0, *meter)] if signature else []) + list(extra_signatures)
    last = 0
    for tick, num, den in sorted(signatures):
        meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=tick - last))
        last = tick
    mid.tracks.append(meta)
    events = []
    total_beats = bars * meter[0] * 4 // meter[1]  # in quarter notes
    for k in range(total_beats * 2):  # eighth notes
        on = rng.integers(0, total_beats * tpb) if random_onsets else k * tpb // 2
        on = max(0, int(on + rng.integers(-onset_jitter_ticks, onset_jitter_ticks + 1)))
        pitch = 60 + (k % 5)
        events += [(on, 1, mido.Message("note_on", note=pitch, velocity=80, channel=0)),
                   (on + tpb // 4, 0, mido.Message("note_off", note=pitch, velocity=0, channel=0))]
    for beat in range(total_beats):
        if beat % meter[0] in snare_beats:
            events += [(beat * tpb, 1, mido.Message("note_on", note=38, velocity=90, channel=9)),
                       (beat * tpb + 60, 0, mido.Message("note_off", note=38, velocity=0, channel=9))]
    track = mido.MidiTrack()
    last = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(msg.copy(time=tick - last))
        last = tick
    mid.tracks.append(track)
    mid.save(path)


class RasterizeTests(unittest.TestCase):
    def test_matches_live_rasterizer_event_for_event(self):
        rng = np.random.default_rng(1)
        onsets = np.round(rng.uniform(0, 3, 60), 3)
        offsets = onsets + np.round(rng.uniform(0.0, 0.4, 60), 3)
        offsets[:5] = onsets[:5]  # zero-length notes
        onsets[5:10] = np.round(onsets[5:10] * 20) / 20  # exactly on frame times
        pitches = rng.integers(55, 66, 60)
        frames = int(np.ceil(offsets.max() * 20)) + 1
        fast = rasterize(onsets, offsets, pitches, frames, 20)
        live = MidiFrameRasterizer()
        events = sorted([(t, 0, "note_off", p) for t, p in zip(offsets, pitches)] +
                        [(t, 1, "note_on", p) for t, p in zip(onsets, pitches)])
        # Zero-length notes: deliver their note-on before their note-off.
        events = sorted(events, key=lambda e: (e[0], e[1] if e[0] not in set(onsets[:5]) else -e[1]))
        for t, _, kind, p in events:
            live.push_event(kind, int(p), 80 if kind == "note_on" else 0, float(t))
        expected = np.array([f["frame"] for f in live.advance_to((frames - 1) / 20)], dtype=np.uint8)
        np.testing.assert_array_equal(fast, expected)


class RenderTests(unittest.TestCase):
    def test_constant_tempo_labels_are_exact(self):
        example = render_example(song_on_beats(bpm=120.0), np.random.default_rng(0), TIMING, NO_AUGMENTATION, lead_in=False)
        frames = np.flatnonzero(example["mask"])
        position = frames / 20 * 2.0  # 120 BPM: two beats per second, first note at beat 0
        np.testing.assert_array_equal(example["beat"][frames],
                                      np.floor((position % 1) * TIMING.beat_bins).astype(int))
        np.testing.assert_array_equal(example["bar"][frames],
                                      np.floor((position % 4) / 4 * TIMING.bar_bins).astype(int))
        # The bar position folded to one beat agrees with the beat position,
        # compared at the coarser of the two resolutions.
        per_beat = TIMING.bar_bins // 4
        coarse = min(per_beat, TIMING.beat_bins)
        np.testing.assert_array_equal(example["bar"][frames] % per_beat * coarse // per_beat,
                                      example["beat"][frames] * coarse // TIMING.beat_bins)
        np.testing.assert_allclose(example["bpm"][frames], 120.0, rtol=1e-6)
        # Notes start on beats: every half second at 120 BPM.
        np.testing.assert_array_equal(np.flatnonzero(example["roll"][:, 60 - 21])[:3], [0, 1, 2])

    def test_three_four_and_six_eight_labels(self):
        # 3/4: bar = 3 quarter notes, beat = quarter note.
        example = render_example(song_on_beats(meter=((0.0, 3, 4),)), np.random.default_rng(0), TIMING,
                                 NO_AUGMENTATION, lead_in=False)
        frames = np.flatnonzero(example["mask"])
        position = frames / 20 * 2.0
        np.testing.assert_array_equal(example["bar"][frames], np.floor((position % 3) / 3 * 32).astype(int))
        np.testing.assert_array_equal(example["beat"][frames], np.floor((position % 1) * 32).astype(int))
        np.testing.assert_allclose(example["bpm"][frames], 120.0, rtol=1e-6)
        # 6/8: bar = 3 quarter notes, beat = eighth note (the denominator), so the
        # beat cycle is half a quarter and the hint counts eighths per minute.
        example = render_example(song_on_beats(meter=((0.0, 6, 8),)), np.random.default_rng(0), TIMING,
                                 NO_AUGMENTATION, lead_in=False)
        np.testing.assert_array_equal(example["bar"][frames], np.floor((position % 3) / 3 * 32).astype(int))
        np.testing.assert_array_equal(example["beat"][frames], np.floor((position % 0.5) / 0.5 * 32).astype(int))
        np.testing.assert_allclose(example["bpm"][frames], 240.0, rtol=1e-6)

    def test_bars_follow_meter_changes(self):
        # One 4/4 bar, one 2/4 bar, then 4/4 again.
        meter = (np.array([0.0, 4.0, 6.0]), np.array([4, 2, 4], np.uint8), np.array([4, 4, 4], np.uint8))
        bar, beat, unit = bar_and_beat_phase([0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0], *meter)
        np.testing.assert_allclose(bar, [0.0, 0.5, 0.0, 0.5, 0.0, 0.25, 0.0])
        np.testing.assert_allclose(beat, 0.0)
        np.testing.assert_allclose(unit, 1.0)

    def test_meter_segments(self):
        raw = lambda *sigs: {"tpb": 480, "signatures": list(sigs)}
        self.assertEqual(meter_segments(raw((0, 4, 4), (1920, 2, 4), (2880, 4, 4))),
                         ([(0, 4, 4), (1920, 2, 4), (2880, 4, 4)], None))
        self.assertEqual(meter_segments(raw((0, 4, 4), (1920, 4, 4))), ([(0, 4, 4)], None))  # restatement
        self.assertEqual(meter_segments(raw((1920, 3, 4))), ([(0, 4, 4), (1920, 3, 4)], None))  # 4/4 before
        self.assertEqual(meter_segments(raw((0, 6, 8), (1440, 4, 4))), ([(0, 6, 8), (1440, 4, 4)], None))
        self.assertEqual(meter_segments(raw((0, 4, 4), (960, 3, 4)))[1], "meter_event_off_bar_line")
        self.assertEqual(meter_segments(raw((0, 9, 4)))[1], "unsupported_meter")  # 9 quarters > 8
        self.assertEqual(meter_segments(raw())[1], "no_time_signature")

    def test_stuck_notes_are_cut(self):
        song = song_on_beats(beats=64)
        song["end"][0], song["pitch"][0] = 200.0, 30  # held 100 s at 120 BPM (missing note-off), own key
        example = render_example(song, np.random.default_rng(0), TIMING, NO_AUGMENTATION, lead_in=False)
        held = np.flatnonzero(example["roll"][:, 30 - 21])
        self.assertEqual(held.min(), 0)
        self.assertEqual(held.max(), 10 * 20 - 1)  # sounding for exactly 10 s, not 100 s

    def test_labels_follow_notes_under_tempo_drift(self):
        synth = SynthesisConfig(max_delete=0, max_insert=0, drift_step_max=0.03)
        song = song_on_beats(beats=200, bpm=100.0)
        for seed in range(5):
            example = render_example(song, np.random.default_rng(seed), TIMING, synth, lead_in=False)
            column = example["roll"][:, 60 - 21].astype(int)
            onset_frames = np.flatnonzero(np.diff(np.concatenate(([0], column))) == 1)
            beats_at_onsets = example["beats"][onset_frames]
            # Each note starts on beat j; its first sounding frame lies within
            # one frame after it, so the labelled position is j + (<1 frame).
            offsets = beats_at_onsets - np.round(beats_at_onsets)
            self.assertTrue((offsets >= -1e-4).all() and (offsets < 0.2).all(), (seed, offsets.max()))
            self.assertTrue((np.diff(example["beats"]) > 0).all())
            self.assertFalse(np.allclose(example["bpm"], 100.0))  # tempo really drifted

    def test_lead_in_is_silent_unlabelled_and_deterministic(self):
        song = song_on_beats(parts=3)
        first = render_example(song, np.random.default_rng(42), TIMING)
        again = render_example(song, np.random.default_rng(42), TIMING)
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])
        lead = TIMING.context_frames - 1
        self.assertFalse(first["roll"][:lead].any())
        self.assertFalse(first["mask"][:lead].any() or first["hint_valid"][:lead].any())
        self.assertTrue((first["beat"][:lead] == IGNORE_INDEX).all())
        self.assertTrue(first["mask"][lead:].all())
        self.assertEqual(len(set(first["hint_valid"][lead:])), 1)  # all song frames given or all withheld

    def test_hint_given_about_half_the_time_and_tracks_vary(self):
        song = song_on_beats(parts=4)
        given, chosen_keys = [], set()
        for seed in range(200):
            example = render_example(song, np.random.default_rng(seed), TIMING)
            given.append(bool(example["hint_valid"][-1]))
            chosen_keys.add(tuple(np.flatnonzero(example["roll"].any(axis=0))))
        self.assertTrue(0.4 < np.mean(given) < 0.6)
        self.assertGreater(len(chosen_keys), 5)


    def test_crop_keeps_real_context_unlabelled(self):
        example = render_example(song_on_beats(beats=200), np.random.default_rng(0), TIMING, SynthesisConfig())
        context, frames = TIMING.context_frames - 1, 60
        crop = crop_example(example, np.random.default_rng(1), frames, context)
        start = int(np.random.default_rng(1).integers(0, len(example["roll"]) - context - frames + 1))
        window = slice(start, start + context + frames)
        self.assertEqual(len(crop["roll"]), context + frames)
        np.testing.assert_array_equal(crop["roll"], example["roll"][window])  # real past as context
        for name in ("beat", "bar"):
            self.assertTrue((crop[name][:context] == IGNORE_INDEX).all())
            np.testing.assert_array_equal(crop[name][context:], example[name][window][context:])
        self.assertFalse(crop["mask"][:context].any())
        np.testing.assert_array_equal(crop["mask"][context:], example["mask"][window][context:])
        np.testing.assert_array_equal(crop["bpm"], example["bpm"][window])
        short = crop_example(example, np.random.default_rng(1), 10**6, context)
        self.assertIs(short, example)


class CheckTests(unittest.TestCase):
    def check(self, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.mid"
            write_midi(path, **kwargs)
            return check_score(read_score(path))

    def test_rules(self):
        cases = [  # (name, file settings, expected reason)
            ("clean 4/4", {}, "ok"),
            ("humanized", {"onset_jitter_ticks": 12}, "ok"),
            ("3/4", {"meter": (3, 4)}, "ok"),
            ("6/8", {"meter": (6, 8)}, "ok"),
            ("one 2/4 bar inside 4/4", {"extra_signatures": [(16 * 480, 2, 4), (18 * 480, 4, 4)]}, "ok"),
            ("no time signature", {"signature": False}, "no_time_signature"),
            ("bar longer than 8 quarter notes", {"meter": (9, 4)}, "unsupported_meter"),
            ("meter event mid-bar", {"extra_signatures": [(2 * 480, 4, 4)]}, "meter_event_off_bar_line"),
            ("played in", {"random_onsets": True}, "off_grid"),
            ("snare on 1 and 3", {"snare_beats": (0, 2)}, "snare_on_1_3"),
            ("too short", {"bars": 10}, "too_short"),  # 80 notes but only 20 s
        ]
        for name, kwargs, expected in cases:
            with self.subTest(case=name):
                passed, reason, _ = self.check(**kwargs)
                self.assertEqual(reason, expected)
                self.assertEqual(passed, expected == "ok")

    def test_song_keeps_pitched_parts_only_and_pack_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.mid"
            write_midi(path)
            song = score_to_song(read_score(path))
            self.assertEqual(set(song["part"].tolist()), {0})        # drums removed
            self.assertEqual(len(song["start"]), 40 * 4 * 2)
            np.testing.assert_allclose(song["start"][:3], [0.0, 0.5, 1.0])
            other = song_on_beats(parts=2, meter=((0.0, 4, 4), (8.0, 3, 4)))
            save_pack(Path(folder) / "pack.npz", [song, other], ["a", "b"],
                      {"group": np.array([0, 0]), "meter": np.array(["4/4", "changes"])})
            pack = SongPack(Path(folder) / "pack.npz")
            self.assertEqual(list(pack.ids), ["a", "b"])
            self.assertEqual(list(pack.meta["meter"]), ["4/4", "changes"])
            self.assertEqual(list(pack.meta["group"]), [0, 0])
            for original, restored in ((song, pack.song(0)), (other, pack.song(1))):
                for key in original:
                    np.testing.assert_array_equal(original[key], restored[key])


if __name__ == "__main__":
    unittest.main()
