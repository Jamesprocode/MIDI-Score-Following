"""How a cue's log-likelihood becomes the evidence the HMM adds (then times elapsed seconds and the cue's weight).

"max" (B5 and every run before 11-13): subtract the best place's value, so the best place gets 0 and every other place
a penalty. How big the penalties are is whatever the cue's own numbers give:
  harmony    log of a softmax of z-scored template matches, with a 2% floor
  rhythm     the network's log-probability of the place's bar phase
  melody v2  the mean log-likelihood per recent note
These are on different scales, so equal weights do not mean equal influence (Alex, 11-05).

Per-cue z-score (11-13, James): one mean and one std per cue, pooled over every score the cue gave on the 60 training
songs (every place of the chart at every check where the cue spoke; fit_scaling.py, from the check_cue runs):
  z      (s - max(s) - mean) / std     base: the score minus the moment's best (cue_scores_per_cue_z_test.png)
  zraw   (s - mean) / std              base: the raw score
  zw     (s - mean(s)) / std_within    ours (11-13, from the review): each moment's own mean, and the spread of the
                                       places within a moment, pooled over moments (sqrt of the mean per-check variance)
The follower renormalises after every update, so anything added to every place at one update cancels (the max, the
means): every rule is the cue's evidence divided by its fitted std, a fixed weight 1/std (times the follower's own
weight). They differ only in which std. z's and zraw's std also count how much the cue's level wanders from moment to
moment (a different part for each base), which the follower never sees; zw's counts only the spread across places at
one moment, the part the follower feels, so under zw every cue's evidence has the same typical spread (1) inside the
follower. Ours: the numbers come from check_cue, which gave each cue the annotated tempo as its BPM hint; inside a
follower the cue gets the follower's tempo.

A follower picks one rule per cue (followers.py, "cue@rule"); hmm.Plugged.scaling holds the rule's name or, for a
fitted rule, the function built for that cue (resolve).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path


def subtract_max(log_likelihood):
    return log_likelihood - log_likelihood.max()


class ZScore:
    """(s - max(s) - mean) / std (base "max"), (s - mean) / std (base "raw") or (s - mean(s) - mean) / std (base
    "within", where the fitted mean is 0), with one cue's fitted numbers."""

    def __init__(self, mean, std, base):
        if base not in ("max", "raw", "within"):
            raise ValueError(f"base {base!r}: max, raw or within")
        if not std > 0:
            raise ValueError(f"std {std!r} must be positive")
        self.mean, self.std, self.base = float(mean), float(std), base

    def __call__(self, log_likelihood):
        centre = (log_likelihood.max() if self.base == "max" else log_likelihood.mean() if self.base == "within"
                  else 0.)
        return (log_likelihood - centre - self.mean) / self.std

    def __repr__(self):
        return f"ZScore(mean={self.mean}, std={self.std}, base={self.base!r})"


SCALINGS = {"max": subtract_max}        # rules without fitted numbers
FITTED = {"z": "max", "zraw": "raw", "zw": "within"}  # rules that read fit_scaling.py's file: rule -> base
RULES = (*SCALINGS, *FITTED)


@functools.lru_cache(maxsize=None)
def _read(path):
    return json.loads(Path(path).read_text())


def resolve(rule, cue, params=""):
    """What hmm.Plugged.scaling holds for ``cue`` under ``rule``: the rule's name (SCALINGS), or a ZScore with the
    cue's mean and std from ``params`` (fit_scaling.py's file)."""
    if rule in SCALINGS:
        return rule
    if rule not in FITTED:
        raise ValueError(f"unknown scaling {rule!r}; rules: {list(RULES)}")
    if not params:
        raise ValueError(f"scaling {rule!r} needs the fitted numbers: --scaling-params FILE (fit_scaling.py)")
    numbers = _read(str(Path(params).resolve()))["cues"]
    if cue not in numbers:
        raise ValueError(f"{params} has no numbers for cue {cue!r} (it has {sorted(numbers)})")
    base = FITTED[rule]
    return ZScore(numbers[cue][base]["mean"], numbers[cue][base]["std"], base)


def function(scaling):
    """The function for what hmm.Plugged.scaling holds (a rule name or a built function)."""
    return scaling if callable(scaling) else SCALINGS[scaling]
