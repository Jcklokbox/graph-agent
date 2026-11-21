"""REJECTED probe: feature-basis spread. D-114: configuration posterior spread carried on the law's own feature basis."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class SpreadProvision(SparseSignedReward):
    cell_cap = 512

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cells = OrderedDict()
        self._sig = None
        self._free = np.empty(0, int)
        self._b = None

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
        pattern = self._pattern(self._b) if self._b is not None else ()
        n = np.zeros(self.C)
        s = np.zeros(self.C)
        for c in range(self.C):
            rec = self.cells.get((c, pattern))
            if rec is not None:
                n[c], s[c] = rec
        mean = (s + .5) / (1. + n)
        sd = np.sqrt(mean * (1. - mean) / (2. + n))
        # Marginal plausibility on the law's own feature basis, exactly as the
        # marginal statistic: how likely a novel payoff opportunity attaches to
        # that feature. The new part is only its scale: the spread of the payoff
        # observed in that configuration, so a configuration whose payoff is
        # already pinned down carries no optimism anywhere.
        strength = np.tile(self.onset_total / (.5 + self.past_on), self.L)
        rarity = 1. - (.5 + self.present) / (1. + self.total)
        plaus = strength * rarity
        plaus[self.slots[self.slots >= 0]] = 0.
        upper = sd[:, None] * plaus[None, :]
        upper *= ~rule[1].any(1)[:, None]
        return np.maximum(0., upper), (sd * sd)[:, None] * np.ones((self.C, self.F - 1))

    def provisional_gate(self, b, hist):
        self._b = b
        return self.phase(b, hist)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = SpreadProvision(f.J, f.A, f.D)
