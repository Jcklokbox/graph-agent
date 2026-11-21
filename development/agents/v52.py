"""D-116: under-tested control-transition optimism + successor-law spread.

Same unified M_factor + controller as v48. Two bounded root-score pulls, added
not multiplied: (1) count-based bonus c/(2+n) on under-tested control
transitions from this state (v51; action-asymmetric, breaks bitwise ties and
spurious pin margins, replenishes on scope revision), (2) one posterior sd of
successor-law epistemic optimism (v50; outcome-atom uncertainty, decays as
laws tighten). Passive 3b-1 predictor unchanged.
"""
from agents.v48 import Agent as Base, ScopedSignedReward


class Agent(Base):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        self.factor.reward.root_transition_bonus=0.3
        self.factor.reward.epistemic_bonus=True
