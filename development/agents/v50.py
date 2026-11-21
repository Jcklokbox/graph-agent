"""D-114: one posterior sd of successor-law epistemic optimism at the root.

Same unified M_factor + controller as v48. The only change is the root
action-values: under-tested successor outcome laws carry a bounded pull
(one posterior sd, branch-mass and transition weighted) instead of tying at
the shared optimism bound and falling back to a uniform RNG lottery. The
passive 3b-1 predictor, isolation checks and scaling are unchanged.
"""
from agents.v48 import Agent as Base, ScopedSignedReward


class Agent(Base):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        self.factor.reward.epistemic_bonus=True
