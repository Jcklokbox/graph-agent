"""REJECTED probe: parent-configuration cells, no marginal term. D-114: provisional payoff counted per configuration of its own payoff parents.

The optimistic payoff of the unified object is a function of the control state, so
once every state is sampled it is monotone in the state mean; a payoff that waits
in an untried configuration of a state whose mean is poor cannot be exposed. This
agent keeps that marginal term and adds the same hypothesis where it can be
tested: the cell is (outcome control state, observed pattern of the channels the
reward law has already screened as payoff parents, i.e. the sources in its own slot
bank). Its upper value is the Jeffreys upper value of the cell's realized payoff,
which anneals with that cell's own exposure, so no exploration rate is introduced
and coordinates that never condition the payoff cannot inflate the bound (a noise
channel is not a payoff parent, so it does not multiply the cells).
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class ParentConfigProvision(SparseSignedReward):
    cell_cap = 256

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cells = OrderedDict()
        self._sig = None
        self._idx = np.empty(0, int)
        self._b = None

    def _pattern(self, b):
        """Observed values of the channels the payoff law has screened as parents."""
        active = self.slots[self.slots >= 0]
        sig = tuple(int(s) for s in active)
        if sig != self._sig:
            src = np.unique(active % self.J) if len(active) else np.empty(0, int)
            self._idx = src.astype(int)
            self._sig = sig
        if len(self._idx) == 0:
            return ()
        return tuple(np.flatnonzero(b[self._idx] > .5).tolist())

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

    def configural_upper(self, rule, b):
        pattern = self._pattern(b)
        n = np.zeros(self.C)
        s = np.zeros(self.C)
        for c in range(self.C):
            rec = self.cells.get((c, pattern))
            if rec is not None:
                n[c], s[c] = rec
        mean = (s + .5) / (1. + n)
        return mean + (.5 / (.5 + n)) * (1. - mean)

    def provisional_gate(self, b, hist):
        return self.phase(b, hist)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = ParentConfigProvision(f.J, f.A, f.D)
