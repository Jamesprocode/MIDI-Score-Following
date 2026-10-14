"""Beat and downbeat tracking evaluation for the phase outputs.

The network outputs phase distributions, not beat events. ``decode_events``
turns a phase sequence into event times with a simple causal rule (ours; the
cited trackers decode offline with peak picking or a DBN): the expected phase
of each frame's distribution is taken, an event fires when the phase wraps
from the end of the cycle to its start, and its time is interpolated between
the two frames. After firing, the decoder waits until the phase has visited
the middle of the cycle (0.25-0.75), so jitter around the wrap cannot fire twice.

``tracking_metrics`` scores the decoded events against reference times with
mir_eval's standard beat-tracking measures (F-measure with a +-70 ms window,
Cemgil, Goto, P-score, CMLc, CMLt, AMLc, AMLt, information gain), which skip
the first 5 s of each song as mir_eval does by default.
"""
from __future__ import annotations

import warnings

import numpy as np

METRIC_NAMES = {
    "F-measure": "f_measure", "Cemgil": "cemgil", "Cemgil Best Metric Level": "cemgil_best_level",
    "Goto": "goto", "P-score": "p_score",
    "Correct Metric Level Continuous": "cmlc", "Correct Metric Level Total": "cmlt",
    "Any Metric Level Continuous": "amlc", "Any Metric Level Total": "amlt",
    "Information gain": "information_gain",
}


def expected_phase(probabilities):
    """Circular mean of [frames, bins] phase distributions, in [0, 1)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    angles = 2 * np.pi * (np.arange(probabilities.shape[-1]) + 0.5) / probabilities.shape[-1]
    vector = probabilities @ np.exp(1j * angles)
    return np.mod(np.angle(vector) / (2 * np.pi), 1.0)


def decode_events(phase, frame_rate):
    """Event times (s, frame 0 at time 0) where the phase wraps; causal."""
    times = []
    armed = bool(phase[0] >= 0.25) if len(phase) else False
    for t in range(1, len(phase)):
        before, now = phase[t - 1], phase[t]
        if armed and before - now > 0.5:  # wrapped forward past the cycle start
            fraction = (1.0 - before) / ((1.0 - before) + now)
            times.append((t - 1 + fraction) / frame_rate)
            armed = False
        if 0.25 <= now <= 0.75:  # real progress through the cycle re-arms
            armed = True
    return np.asarray(times)


def tracking_metrics(reference, estimated):
    """mir_eval beat measures with short snake_case names; NaN-free."""
    import mir_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty estimates are scored 0, with a warning
        scores = mir_eval.beat.evaluate(np.sort(np.asarray(reference, dtype=float)),
                                        np.sort(np.asarray(estimated, dtype=float)))
    return {METRIC_NAMES[name]: float(np.nan_to_num(value)) for name, value in scores.items()}
