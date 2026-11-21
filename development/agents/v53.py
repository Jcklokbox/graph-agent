"""REJECTED probe: configuration posterior spread. D-114: optimism = plausible posterior variance of the untried configuration.

Same locus as v50 (provisional optimism indexed by the perceptual configuration
rather than the control-state mean), but the magnitude is the posterior variance
of the cell's mean payoff under a Jeffreys prior,

    var = mean(1-mean) / (n + 2),

which is large only while the configuration is untried and anneals with its own
exposure, and the gate is the union of the two statements under which a payoff
can be realized: arrival in an untried control state (onset) or testing an untried
configuration at the state where the law still has spread (feature activation).
No exploration rate, no per-row switch.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from agents.v44 import Agent as Controller, SparseSignedReward


class VarianceProvision(SparseSignedReward):
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

    def learn_joint(self, b, hist, a, y, bn):
        super().learn_joint(b, hist, a, y, bn)
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

    def provisional(self, rule):
        pattern = self._pattern(self._last_b) if self._last_b is not None else ()
        n = np.zeros(self.C)
        s = np.zeros(self.C)
        for c in range(self.C):
            rec = self.cells.get((c, pattern))
            if rec is not None:
                n[c], s[c] = rec
        mean = (s + .5) / (1. + n)
        sd = np.sqrt(mean * (1. - mean) / (2. + n))
        strength = np.tile(self.onset_total / (.5 + self.past_on), self.L)
        rarity = 1. - (.5 + self.present) / (1. + self.total)
        plaus = strength * rarity
        plaus[self.slots[self.slots >= 0]] = 0.
        if (self.slots >= 0).any():
            p = np.max(np.where(rule[1], 0., plaus), axis=1)
        else:
            p = np.zeros(self.C)
        # Same shape and unit as the marginal statistic: (C, F-1) payoff upper
        # values, here driven by the posterior spread of the configuration.
        upper = (p * sd)[:, None] * np.ones((self.C, self.F - 1))
        upper *= ~rule[1].any(1)[:, None]
        return upper, sd * sd

    _last_b = None

    def provisional_gate(self, b, hist):
        self._last_b = b
        return self.phase(b, hist) | self.input(b, hist)[1:].astype(bool)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = VarianceProvision(f.J, f.A, f.D)
