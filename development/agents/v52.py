"""REJECTED probe: residual-variance product. D-114: optimism proportional to the payoff variance the law cannot explain.

Diagnosis this answers (AR-6 alt-N, exposed probes). v48's optimistic payoff is an
*onset-frequency* statistic: `incumbent = max(r, (bound*onset).max(1))`. Two
consequences, both measured on `canc`:

* the term is charged to onsets, and the only state reached with high probability
  that carries an onset is cue+lamp (stay preserves location, so `p(stay)` from the
  cue state to cue+lamp is ~0.97 while `p(left)` is ~0.04). The bound is therefore
  a reward for *staying*, and the maximum-reward decision (left at a cue with the
  cancellation lamp off) is scored at exactly 0.5, i.e. below staying;
* the term is blind to whether the current law already predicts the payoff. It
  fires wherever an onset is rare, so it cannot distinguish a state whose payoff
  is already explained from one whose payoff is a mixture.

The change (one statistic, one locus). The provisional payoff becomes

    upper = base + plausibility * sqrt(residual variance)

where the residual variance is the bounded two-moment estimate of the realized
payoff for that (outcome control state, feature on) cell, from counts the head
already keeps plus one new sufficient statistic. It is the part of the outcome
the current law does not explain, so

* it vanishes where the law is exact (a deterministic payoff given the observed
  configuration carries no optimism), and
* it is positive exactly where an observed configuration predicts the payoff
  poorly, which is the state at which the untried configuration must be tested.

The gate follows the same statement: the hypothesis is about observed
configurations, not about onsets, so it enters at the feature activation.
Supplied by `provisional_gate`, which the rollout prefers over the onset mask.
"""
from __future__ import annotations

import numpy as np

from agents.v44 import Agent as Controller, SparseSignedReward


class ResidualVarianceProvision(SparseSignedReward):
    """Provisional payoff = plausible transient feature x unexplained variance."""

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cell_n = np.zeros((self.C, self.F))
        self.cell_s = np.zeros((self.C, self.F))
        self.cell_q = np.zeros((self.C, self.F))

    def _accumulate(self, b, hist, y):
        x = self.input(b, hist)
        self.cell_n += x
        self.cell_s += y * x
        self.cell_q += (y * y) * x

    def learn_joint(self, b, hist, a, y, bn):
        super().learn_joint(b, hist, a, y, bn)
        self._accumulate(b, hist, y)

    def clear_cells(self):
        self.cell_n.fill(0.)
        self.cell_s.fill(0.)
        self.cell_q.fill(0.)

    def provisional(self, rule):
        n = self.cell_n
        mean = (self.cell_s + .5) / (1. + n)
        second = (self.cell_q + 1. / 3) / (1. + n)
        variance = np.maximum(0., second - mean * mean)[:, 1:]   # x[:,0]=1 is the baseline
        # Same transient-coordinate plausibility as the marginal statistic: a
        # coordinate that is on whenever the control state merely persists cannot
        # carry a novel payoff opportunity.
        strength = np.tile(self.onset_total / (.5 + self.past_on), self.L)
        rarity = 1. - (.5 + self.present) / (1. + self.total)
        plaus = strength * rarity
        plaus[self.slots[self.slots >= 0]] = 0.
        upper = plaus * np.sqrt(variance)
        upper *= ~rule[1].any(1)[:, None]
        return np.maximum(0., upper), variance

    def provisional_gate(self, b, hist):
        # Both hypotheses are live and neither is a switch: a payoff can be
        # realized by arriving in an untried control state (onset) or by testing
        # an untried configuration at the state where the law still has variance
        # (activation). The magnitude of either is the same unexplained variance.
        return self.phase(b, hist) | self.input(b, hist)[1:].astype(bool)


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = ResidualVarianceProvision(f.J, f.A, f.D)
