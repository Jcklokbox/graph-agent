"""REJECTED probe: room-bounded configural gate. D-114: optimism is the room between the law's prediction and a bounded payoff.

The provisional payoff of the unified object is charged to rare onsets, so on
`canc` it rewards the control state that is entered by staying (cue+lamp, onset
probability .97) instead of the decision that can realize the maximum payoff (left
at a cue with the lamp off). It is also blind to whether the law already explains
the payoff, so it is either absent where it is needed or large where nothing is
left to gain.

This agent keeps the marginal statistic exactly, but the optimism it may add on
top of the learned law is bounded by the room that is actually left,

    optimism = plausibility(feature) * (1 - r_predicted(state, feature)),

so a state whose payoff the law already predicts at the maximum gains nothing,
while a decision whose observed configuration predicts a mid payoff keeps an
opening toward a better one. The gate is the activation of the feature the
hypothesis is about, not only its onset, because a payoff left untested at a
configuration is reachable from the decision that manifests that configuration.
"""
from __future__ import annotations
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class RoomProvision(SparseSignedReward):
    def provisional(self, rule):
        bound, var = super().provisional(rule)
        return bound, var

    def provisional_gate(self, b, hist):
        # The hypothesis is about a configuration that can still be *manifested*,
        # i.e. that can turn on. A coordinate that is on whenever its control
        # state persists (the actuator that sustained it) offers no untried
        # configuration to try, so it must not carry the optimism; this is the
        # same stationarity test the marginal statistic applies through `past_on`.
        x = self.input(b, hist)[1:].astype(bool)
        transient = np.tile(self.onset_total <= self.past_on * .5, self.L)
        return self.phase(b, hist) | (x & transient)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = RoomProvision(f.J, f.A, f.D)
