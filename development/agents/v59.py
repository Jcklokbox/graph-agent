"""REJECTED probe: per-state Jeffreys room. D-114: provisional optimism is the room left *in the cell it is attached to*.

Measured defect (AR-6 alt-N). The optimistic payoff of the unified object is

    bound = m + (1-m) * plausibility / sqrt(1+n),    incumbent = max(r, (bound*onset).max)

with `m` the payoff mean of an *outcome control state* and `plausibility` the
transient-on-ness of a feature. On `canc` the only state that is reached with high
probability and carries an onset is cue+lamp (`p(stay|cue)=.97` but
`p(left|cue)=.04`), so the optimism rewards staying; the maximum-reward decision,
left at a cue with the lamp off, keeps its learned mid payoff and is never tested.

At the same time the term claims (1-m)*plausibility of headroom on a feature with
no evidence, and `m` belongs to the state rather than to that feature. An onset
feature may even be one the reward law has already screened and explained (it is
in the slot bank), in which case there is nothing provisional about it at all.

One coherent change. Keep the marginal statistic exactly; let the provisional
headroom of a cell be the remaining spread of the *realized payoff in that cell*,
which is the honest Jeffreys width of the cell's Beta posterior,

    room(c, f) = sqrt( m_cf (1 - m_cf) / (2 + n_cf) ),

computed from sufficient statistics the head already keeps (`on`, `type_count`,
`type_sum`), so

* a cell whose payoff is already pinned near 0 or 1 has almost no room, and an
  onset the law has explained cannot manufacture optimism;
* a cell with a mid, poorly determined payoff has room, which is exactly the
  decision that has not been identified yet;
* the room shrinks with the cell's own exposure, so no exploration rate is added.

The hypothesis is then attached to the activation of the feature it is about,
because the configuration that must be tried is manifested at the decision, not
only at its onset.
"""
from __future__ import annotations
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class RoomProvision(SparseSignedReward):
    def provisional(self, rule):
        n = self.type_count                                   # (C, B)
        s = self.type_sum
        mean = (s + self.center[None, :] / self.B) / (n + 1. / self.B)
        total = n.sum(1, keepdims=True)
        m = (mean * n).sum(1, keepdims=True) / np.maximum(total, 1e-12)
        room = np.sqrt(np.maximum(0., m * (1. - m) / (2. + total)))
        # Same feature plausibility as the marginal statistic: how likely a novel
        # payoff opportunity attaches to that coordinate (a coordinate that is on
        # whenever its state merely persists cannot carry one).
        strength = np.tile(self.onset_total / (.5 + self.past_on), self.L)
        rarity = 1. - (.5 + self.present) / (1. + self.total)
        plaus = strength * rarity
        plaus[self.slots[self.slots >= 0]] = 0.
        upper = room * plaus[None, :]
        upper *= ~rule[1].any(1)[:, None]
        return np.maximum(0., upper), np.broadcast_to(room * room, (self.C, self.F - 1)).copy()

    def provisional_gate(self, b, hist):
        return self.input(b, hist)[1:].astype(bool) | self.phase(b, hist)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = RoomProvision(f.J, f.A, f.D)
