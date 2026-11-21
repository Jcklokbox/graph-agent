"""REJECTED probe: unary-coordinate cells. D-114b: provisional payoff indexed by (outcome state, one coordinate) (AR-6 alt-N).

Same change as D-114, but the cell is (outcome control state, free channel j,
value of j). A full configuration over nuisance channels is never revisited (an
iid noise bit redraws every step), so its opportunity count cannot anneal and the
optimism stays uninformative. A unary cell is revisited whenever that coordinate
takes the same value, so the Jeffreys upper bound falls with real exposure and
the exploratory drive concentrates on coordinates that keep paying off.
"""
from __future__ import annotations

import numpy as np

from agents.v44 import Agent as Controller, SparseSignedReward


class CoordinateProvision(SparseSignedReward):
    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cell = np.zeros((self.C, J, 2, 2))   # [.., value, (n, sum)]

    def learn_joint(self, b, hist, a, y, bn):
        super().learn_joint(b, hist, a, y, bn)
        c, j, bit = int(self.code(bn)), np.arange(self.J), (b > .5).astype(int)
        self.cell[c, j, bit, 0] += 1.
        self.cell[c, j, bit, 1] += y

    def coordinate_upper(self, b):
        n, s = self.cell[..., 0], self.cell[..., 1]
        total = n.sum(-1)[..., None]
        mean = (s + .5) / (1. + total)
        unseen = .5 / (.5 + total)
        upper = mean + unseen * (1. - mean)
        bit = (b > .5).astype(int)
        return upper[np.arange(self.C)[:, None], np.arange(self.J)[None, :], bit].max(1)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = CoordinateProvision(f.J, f.A, f.D)
