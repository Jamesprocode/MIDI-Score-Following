"""The timing network as an observation for the position/tempo grid filter.

Maezawa (2023, sec. 2.1-2.2): the downbeat phase tracker's posterior "gets
mapped to the posterior distribution over the score position p_d(s)", and the
score follower uses p_h(x|s) p_d(s) as the likelihood. ``DownbeatEvidence``
does that mapping for improvement_study/adaptive_grid.follow_positions, through
the same hook its rule-based onset cue (improvement_study/rhythm.py) uses: a
log-potential over (tempo, position) per frame. For a candidate unfolded
position s (quarter-note beats from the first musical bar), its bar phase is
(s mod beats_per_bar) / beats_per_bar, and p_d(s) is the network's probability
of the bar-phase class containing it. The network frame read at a filter frame
at time t is the last one at or before t, so no future input is used.
"""
from __future__ import annotations

import numpy as np
import torch

from yamaha_timing.lakh import SynthesisConfig, rasterize


@torch.no_grad()
def timing_log_probs(model, take, bpm, device="cpu"):
    """Causal whole-take pass (equal to live streaming, with the silent lead-in):
    (beat, bar) log-probabilities per 20 fps frame; frame k is time k / rate.
    ``bpm`` is the BPM condition at every frame (a scalar or one value per frame)."""
    cfg = model.config
    rate, lead = cfg.frame_rate, cfg.context_frames - 1
    end = max(take.span[1], float(take.offsets.max()) if len(take.offsets) else 0.0)
    frames = int(np.ceil(end * rate)) + 1
    offsets = np.minimum(take.offsets, take.onsets + SynthesisConfig().max_note_seconds)
    roll = np.concatenate([np.zeros((lead, cfg.input_pitches), np.uint8),
                           rasterize(take.onsets, offsets, take.pitches, frames, rate)])
    hint = np.full(len(roll), 120.0)
    hint[lead:] = bpm
    valid = np.zeros(len(roll), dtype=bool)
    valid[lead:] = True  # the lead-in never carries a hint, as in training
    out = model(torch.from_numpy(roll).float()[None].to(device),
                bpm_hint=torch.from_numpy(hint).float()[None].to(device),
                hint_valid=torch.from_numpy(valid)[None].to(device))
    return tuple(torch.log_softmax(out[f"{task}_logits"][0, lead:].float(), -1).cpu().numpy()
                 for task in ("beat", "bar"))


class DownbeatEvidence:
    """p_d(s) for the grid filter's rhythm hook (log-potential per tempo and position)."""

    def __init__(self, bar_log_probs, frame_rate, filter_times, beats_per_bar, n_tempi):
        self.log_probs = bar_log_probs
        self.bins = bar_log_probs.shape[1]
        self.frames = np.clip(np.floor(np.asarray(filter_times) * frame_rate + 1e-9).astype(int),
                              0, len(bar_log_probs) - 1)
        self.beats_per_bar = float(beats_per_bar)
        self.n_tempi = n_tempi

    def log_likelihood(self, frame_index, q_grid):
        phase = np.mod(np.asarray(q_grid, dtype=float), self.beats_per_bar) / self.beats_per_bar
        classes = np.minimum((phase * self.bins).astype(int), self.bins - 1)
        row = self.log_probs[self.frames[frame_index], classes]
        return np.broadcast_to(row, (self.n_tempi, len(row)))


class NetworkPhase:
    """Bar-phase log-probabilities for any BPM condition, one frame at a time.

    The backbone does not see the BPM condition (the gate multiplies its output),
    so it runs once per take (causal, with the silent lead-in); each call then
    applies the gate for that frame's hint and the bar head, as the full model
    does. This lets the follower feed its own tempo back as the condition
    (Maezawa 2023, sec. 2.2) without re-running the network. Equal to
    timing_log_probs for a constant hint (tests check this).
    """

    @torch.no_grad()
    def __init__(self, model, take, device="cpu"):
        cfg = model.config
        self.rate, lead = cfg.frame_rate, cfg.context_frames - 1
        end = max(take.span[1], float(take.offsets.max()) if len(take.offsets) else 0.0)
        frames = int(np.ceil(end * self.rate)) + 1
        offsets = np.minimum(take.offsets, take.onsets + SynthesisConfig().max_note_seconds)
        roll = np.concatenate([np.zeros((lead, cfg.input_pitches), np.uint8),
                               rasterize(take.onsets, offsets, take.pitches, frames, self.rate)])
        features = model.backbone(torch.from_numpy(roll).float()[None].transpose(1, 2).to(device))
        self.features = features[0, :, lead:].T.double().cpu().numpy()  # (frames, channels)
        head = model.bar_head
        self.w1, self.b1 = head[1].weight[:, :, 0].double().cpu().numpy(), head[1].bias.double().cpu().numpy()
        self.w2, self.b2 = head[4].weight[:, :, 0].double().cpu().numpy(), head[4].bias.double().cpu().numpy()
        self.proj_w = model.tempo_projection.weight.double().cpu().numpy()
        self.proj_b = model.tempo_projection.bias.double().cpu().numpy()
        self.centers = model.tempo_centers.double().cpu().numpy()

    def frame_at(self, time_s):
        return int(np.clip(np.floor(time_s * self.rate + 1e-9), 0, len(self.features) - 1))

    def bar_log_probs(self, frame, bpm):
        reference = self.centers[-1]
        logits = bpm * (2 * (self.centers - reference)) + (reference ** 2 - self.centers ** 2)
        encoded = np.exp(logits - logits.max())
        encoded /= encoded.sum()
        gate = 1 / (1 + np.exp(-(self.proj_w @ encoded + self.proj_b)))
        hidden = self.w1 @ (self.features[frame] * gate) + self.b1
        hidden = np.where(hidden > 0, hidden, np.expm1(np.minimum(hidden, 0)))  # ELU
        out = self.w2 @ hidden + self.b2
        out -= out.max()
        return out - np.log(np.exp(out).sum())
