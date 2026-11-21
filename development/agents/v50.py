"""D-114: sensory-qualified configural optimism (fixes D-108 discovery deadlock).

D-108 gated configural refinement by marginal reward parentage (`same`).
Parent admission needs marginal reward evidence, which needs payoff, which
needs configural optimism: deadlock on canc when the first cue is missed
and trap shocks extinguish pooled unary optimism (r[left]~0.05).

Correction (one change): qualification by posterior uncertainty. When no
reward parent is earned (`~rule[1].any(1)`), uncertainty is maximal, so
sensory-associated configural hypotheses are qualified; when a parent is
earned, refinement requires that parent (`same`). This is the v44
qualification `same|~any`, restored with derivation: hypotheses are
proposed by sensory association (`assoc`, already thresholded at
log1p(J)), evaluated by reward. No new flag; D-108 scope condition removed
(L-9). Behaviorally coincides with v44 on probed rows; v48 frozen for
comparison.
"""
from agents.v48 import Agent as Controller, ScopedSignedReward


class SensoryQualifiedReward(ScopedSignedReward):
    context_requires_parent = False


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = SensoryQualifiedReward(f.J, f.A, f.D)
