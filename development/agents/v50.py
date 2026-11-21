"""REJECTED probe: configuration cells over all free channels. D-114: provisional payoff indexed by the perceptual configuration (AR-6 alt-N).

Problem this addresses. The optimistic payoff of the unified factor/reward object
is a function of the *control state* only: `incumbent = max(r, (bound*onset).max)`.
Once every state has been sampled, the optimism is monotone in the state mean, so
an action can never look better than the best state mean; a payoff that lives in
an untried *configuration* of a state whose mean is already poor is invisible.
On `canc` that is exactly the maximum-reward action: left at a cue with the
cancellation lamp off. The learned law already screens that parent (the reward
slot bank includes the lamp), so the missing object is not representation but the
domain on which the provisional payoff is counted.

The change (one statistic, one place). The provisional payoff is counted per
(outcome control state, configuration of the channels the control factor does not
own) instead of per outcome state. Its upper value keeps the model's own tempered
form
    local = base + plausibility * (mean - base)
    upper = local + plausibility * (1 - mean) / sqrt(1 + n)
with `mean` and `n` the Jeffreys mean and the opportunity count of the cell, and
`base`/`plausibility` the same state baseline and transient-coordinate strength
the marginal statistic already uses. The opportunity count therefore belongs to
the configuration that must be tried, so the optimism anneals exactly where the
evidence accrues, and no exploration rate is introduced.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from agents.v44 import Agent as Controller, SparseSignedReward


class ConfiguralProvision(SparseSignedReward):
    cell_cap = 512

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cells = OrderedDict()
        self._sig = None
        self._free = np.empty(0, int)

    def _pattern(self, b):
        sig = tuple(int(s) for s in self.slots)
        if sig != self._sig:
            free = np.ones(self.J, bool)
            free[self.slots[self.slots >= 0]] = False
            self._free = np.flatnonzero(free)
            self._sig = sig
        return tuple(np.flatnonzero(b[self._free] > .5).tolist())

    def _observe(self, b, bn, y):
        key = (int(self.code(bn)), self._pattern(b))
        rec = self.cells.get(key)
        if rec is None:
            if len(self.cells) >= self.cell_cap:
                self.cells.popitem(last=False)
            rec = [0., 0.]
            self.cells[key] = rec
        else:
            self.cells.move_to_end(key)
        rec[0] += 1.
        rec[1] += y

    def learn_joint(self, b, hist, a, y, bn):
        super().learn_joint(b, hist, a, y, bn)
        self._observe(b, bn, y)

    def configural_upper(self, rule, b):
        pattern = self._pattern(b)
        n = np.zeros(self.C)
        s = np.zeros(self.C)
        for c in range(self.C):
            rec = self.cells.get((c, pattern))
            if rec is not None:
                n[c], s[c] = rec
        mean = (s + .5) / (1. + n)
        # Same transient-coordinate strength as the marginal statistic: a
        # coordinate that is on whenever the control state merely persists cannot
        # carry a novel payoff opportunity, and the bound must not assert one.
        strength = np.tile(self.onset_total / (.5 + self.past_on), self.L)
        rarity = 1. - (.5 + self.present) / (1. + self.total)
        plaus = strength * rarity
        plaus[self.slots[self.slots >= 0]] = 0.
        # Per-state plausibility of a transient opportunity, not one episode-wide
        # constant: a state that only ever persists carries no novel payoff, so a
        # deterministic safe control transition is not penalised by the bound.
        if (self.slots >= 0).any():
            p = np.max(np.where(rule[1], 0., plaus), axis=1)
        else:
            p = np.zeros(self.C)
        base = rule[5]
        local = base + p * (mean - base)
        upper = local + p * (1. - mean) / np.sqrt(1. + n)
        return upper * ~rule[1].any(1)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = ConfiguralProvision(f.J, f.A, f.D)
