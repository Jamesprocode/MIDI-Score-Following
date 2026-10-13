"""Causal frame inference and chronological MIDI-event rasterization.

``MidiFrameRasterizer`` deliberately separates event delivery from clock
advancement. Queue every event at a timestamp before calling ``advance_to`` at
that timestamp; all those events affect the frame on that boundary. Events at
an already emitted boundary are rejected rather than silently changing history.
Events sharing a timestamp are applied in delivery order.

The roll represents keys held, not acoustic sustain. Pedal, controllers and
pitch bends are not modeled. MIDI file parsing and live MIDI transport belong
outside this module. A typical integration is::

    rasterizer.push_event("note_on", 60, 90, 0.0, channel=0)
    for item in rasterizer.advance_to(0.10):
        prediction = tracker.push_frame(item["frame"], item["timestamp_s"])

``advance_to`` emits silent frames as well, so model input stays at 20 Hz.
"""
from __future__ import annotations

from collections import deque
import math
from numbers import Integral

import torch

from .config import TimingConfig

__all__ = ["StreamingTimingTracker", "MidiFrameRasterizer"]


def _timestamp(value, name="timestamp_s"):
    """Validate recording-relative seconds, before mutating stream state."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite nonnegative seconds")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite nonnegative seconds") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return result


class StreamingTimingTracker:
    """Run one model call per frame using only the rolling past context.

    The model must expose ``config`` and the YamahaTimingTCN forward API.
    Frames contain ``config.input_pitches`` finite values. Timestamps must be
    strictly increasing at ``1 / config.frame_rate`` seconds (1 microsecond
    tolerance); use MidiFrameRasterizer to fill gaps with appropriate activity.

    A BPM hint is either None or a finite positive scalar available at that
    frame. Hints are stored with their frames: a new hint never rewrites earlier
    conditioning. Missing hints and left padding carry a false validity mask.
    ``context_frames_seen`` counts actual frames in the current context, capped
    at ``config.context_frames``; ``context_ready`` becomes true when it is full.

    This wrapper calls ``model.eval()`` and uses ``torch.no_grad()``. It does not
    restore training mode, train weights, decode beat events, or estimate tempo.
    Its JSON-compatible output contains probabilities for the current frame.
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.model.eval()
        self.reset()

    def reset(self):
        """Discard all frame, conditioning and timestamp history."""
        self._history = deque(maxlen=self.config.context_frames)
        self._last_timestamp = None

    def _device_and_dtype(self):
        # Buffers cover parameter-free test/inference modules too.
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            parameter = next(self.model.buffers(), None)
        if parameter is None:
            return torch.device("cpu"), torch.float32
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        return parameter.device, dtype

    def push_frame(self, frame, timestamp_s, bpm_hint=None):
        """Accept one regularly sampled frame and return its phase probabilities.

        Invalid input or a failed model call does not advance wrapper history.
        A full context is re-evaluated once per call; this is intentionally a
        simple reference implementation, not a cached convolution engine.
        """
        timestamp_s = _timestamp(timestamp_s)
        if self._last_timestamp is not None:
            interval = timestamp_s - self._last_timestamp
            if interval <= 0 or not math.isclose(
                interval, 1.0 / self.config.frame_rate, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    "Frame timestamps must increase at the configured frame rate; "
                    "use MidiFrameRasterizer to emit missing frames"
                )
        if bpm_hint is not None:
            if isinstance(bpm_hint, bool):
                raise ValueError("bpm_hint must be a finite positive scalar or None")
            try:
                bpm_hint = float(bpm_hint)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("bpm_hint must be a finite positive scalar or None") from exc
            if not math.isfinite(bpm_hint) or bpm_hint <= 0:
                raise ValueError("bpm_hint must be a finite positive scalar or None")

        with torch.no_grad():
            row = torch.as_tensor(frame, dtype=torch.float32, device="cpu")
            if row.shape != (self.config.input_pitches,):
                raise ValueError(f"frame must have shape [{self.config.input_pitches}]")
            if not torch.isfinite(row).all().item():
                raise ValueError("frame must contain only finite values")
            row = row.detach().clone()
            history = [*self._history, (row, bpm_hint)][-self.config.context_frames:]
            padding = self.config.context_frames - len(history)
            device, dtype = self._device_and_dtype()
            inputs = torch.zeros(
                (1, self.config.context_frames, self.config.input_pitches),
                dtype=dtype, device=device,
            )
            inputs[0, padding:] = torch.stack([item[0] for item in history]).to(
                device=device, dtype=dtype
            )
            # Positive placeholder avoids invalid arithmetic even when masked.
            hints = torch.full(
                (1, self.config.context_frames), 120.0, dtype=dtype, device=device
            )
            hint_valid = torch.zeros(
                (1, self.config.context_frames), dtype=torch.bool, device=device
            )
            for i, (_, hint) in enumerate(history, start=padding):
                if hint is not None:
                    hints[0, i] = hint
                    hint_valid[0, i] = True
            self.model.eval()
            output = self.model(inputs, bpm_hint=hints, hint_valid=hint_valid)
            probabilities = {}
            for name, classes in (
                ("beat", self.config.beat_bins), ("bar", self.config.bar_bins)
            ):
                logits = output[f"{name}_logits"]
                expected = (1, self.config.context_frames, classes)
                if tuple(logits.shape) != expected:
                    raise ValueError(f"{name}_logits must have shape {expected}")
                current = logits[0, -1].float()
                if not torch.isfinite(current).all().item():
                    raise ValueError(f"{name}_logits contains non-finite values")
                probabilities[f"{name}_phase_probabilities"] = (
                    torch.softmax(current, dim=-1).cpu().tolist()
                )
        self._history.append((row, bpm_hint))
        self._last_timestamp = timestamp_s
        return {
            "timestamp_s": timestamp_s,
            **probabilities,
            "context_frames_seen": len(history),
            "context_ready": len(history) == self.config.context_frames,
        }


class MidiFrameRasterizer:
    """Turn chronological note events into regularly sampled key-activity frames.

    ``push_event(event_type, pitch, velocity, timestamp_s, channel=0)`` queues a
    note_on/note_off without emitting a frame. ``advance_to(timestamp_s)`` emits
    every not-yet-emitted grid frame at or before that time. The grid starts at
    ``start_time_s`` (default zero). Deliver all simultaneous events first.

    Event timestamps must be nondecreasing. Time advancement cannot go backward.
    Events earlier than the advanced clock, or at/before the last emitted frame,
    are late and rejected. Queued future events are allowed. A 1-nanosecond
    tolerance is used only at floating-point grid boundaries.

    Overlaps are counted per (channel, pitch). A note-off closes one outstanding
    note-on on that channel; activity remains until all overlaps are released.
    note_on with velocity zero is a note-off. Out-of-piano-range integer pitches
    are ignored, including their releases. ``diagnostics`` reports those ignored
    events and unmatched in-range note-offs; ``reset`` clears both counters.
    Channels are zero-based 0..15; there is no automatic percussion exclusion.
    Callers choosing a pitched-only stream must exclude percussion explicitly.
    """

    _GRID_TOLERANCE = 1e-9

    def __init__(self, config=None, start_time_s=0.0):
        self.config = TimingConfig() if config is None else config
        self.reset(start_time_s=start_time_s)

    def reset(self, start_time_s=0.0):
        """Start a new empty event stream and clock at start_time_s."""
        self._start_time = _timestamp(start_time_s, "start_time_s")
        self._pending = deque()
        self._counts = {}
        self._pitch_counts = [0] * self.config.input_pitches
        self._next_frame_index = 0
        self._last_event_time = None
        self._last_frame_time = None
        self._advanced_to = None
        self._ignored_out_of_range = 0
        self._unmatched_note_off = 0

    @property
    def diagnostics(self):
        """A copy of counters; unmatched releases count when events are consumed."""
        return {
            "ignored_out_of_range_events": self._ignored_out_of_range,
            "unmatched_note_off_events": self._unmatched_note_off,
        }

    def push_event(self, event_type, pitch, velocity, timestamp_s, channel=0):
        """Queue one event; same-time events retain their delivery order."""
        timestamp_s = _timestamp(timestamp_s)
        if event_type not in ("note_on", "note_off"):
            raise ValueError("event_type must be 'note_on' or 'note_off'")
        if isinstance(pitch, bool) or not isinstance(pitch, Integral):
            raise ValueError("pitch must be an integer")
        if isinstance(velocity, bool) or not isinstance(velocity, Integral) or not 0 <= velocity <= 127:
            raise ValueError("velocity must be an integer in 0..127")
        if isinstance(channel, bool) or not isinstance(channel, Integral) or not 0 <= channel <= 15:
            raise ValueError("channel must be a zero-based integer in 0..15")
        if timestamp_s < self._start_time:
            raise ValueError("Event precedes the rasterizer start time")
        if self._last_event_time is not None and timestamp_s < self._last_event_time:
            raise ValueError("Event timestamps must be nondecreasing")
        if self._advanced_to is not None and timestamp_s < self._advanced_to - self._GRID_TOLERANCE:
            raise ValueError("Late event precedes the advanced clock")
        if self._last_frame_time is not None and timestamp_s <= self._last_frame_time + self._GRID_TOLERANCE:
            raise ValueError("Late event: its grid frame was already emitted")
        self._last_event_time = timestamp_s
        pitch, velocity, channel = int(pitch), int(velocity), int(channel)
        if not self.config.midi_low <= pitch < self.config.midi_low + self.config.input_pitches:
            self._ignored_out_of_range += 1
            return
        is_on = event_type == "note_on" and velocity > 0
        self._pending.append((timestamp_s, pitch, channel, is_on))

    def _consume_until(self, timestamp_s):
        while self._pending and self._pending[0][0] <= timestamp_s + self._GRID_TOLERANCE:
            _, pitch, channel, is_on = self._pending.popleft()
            key = (channel, pitch)
            index = pitch - self.config.midi_low
            count = self._counts.get(key, 0)
            if is_on:
                self._counts[key] = count + 1
                self._pitch_counts[index] += 1
            elif count:
                if count == 1:
                    del self._counts[key]
                else:
                    self._counts[key] = count - 1
                self._pitch_counts[index] -= 1
            else:
                self._unmatched_note_off += 1

    def advance_to(self, timestamp_s):
        """Return newly emitted {timestamp_s, frame} records through this time.

        Each frame is an independent JSON-compatible list of binary floats.
        Advancing without events still emits frames and never freezes the clock.
        """
        timestamp_s = _timestamp(timestamp_s)
        if timestamp_s < self._start_time:
            raise ValueError("Cannot advance before start_time_s")
        if self._advanced_to is not None and timestamp_s < self._advanced_to:
            raise ValueError("advance_to timestamps must be nondecreasing")
        frames = []
        while True:
            frame_time = self._start_time + self._next_frame_index / self.config.frame_rate
            if frame_time > timestamp_s + self._GRID_TOLERANCE:
                break
            self._consume_until(frame_time)
            frames.append({
                "timestamp_s": frame_time,
                "frame": [float(count > 0) for count in self._pitch_counts],
            })
            self._last_frame_time = frame_time
            self._next_frame_index += 1
        self._consume_until(timestamp_s)
        self._advanced_to = timestamp_s
        return frames
