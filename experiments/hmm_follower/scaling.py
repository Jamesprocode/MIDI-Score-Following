"""How a cue's log-likelihood becomes the evidence the HMM adds (then times elapsed seconds and the cue's weight).

So far every cue uses "max": subtract the best place's value, so the best place gets 0 and every other place a
penalty. How big the penalties are is whatever the cue's own numbers give:
  harmony    log of a softmax of z-scored template matches, with a 2% floor
  rhythm     the network's log-probability of the place's bar phase
  melody v2  the mean log-likelihood per recent note
These are on different scales, so equal weights do not mean equal influence (Alex, 11-05). Other rules go here, next
to "max", and a follower picks one per cue (hmm.Plugged.scaling).
"""
from __future__ import annotations


def subtract_max(log_likelihood):
    return log_likelihood - log_likelihood.max()


SCALINGS = {"max": subtract_max}
