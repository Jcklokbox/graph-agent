"""D-115: count-based root bonus on under-tested control transitions.

Same unified M_factor + controller as v48. The only change is the root
action-values: an additive bonus of c/(2+n) where n is the number of times
this action was taken from this control state (replenishes when the actuator
scope is revised). Untried actions beat the bitwise tie / spurious pin margin;
the bonus decays below the margin after a handful of probes, so exploitation
is untouched once the position-control channel is identified. Not a semantic
seed, not onset novelty; the passive 3b-1 predictor is unchanged.
"""
from agents.v48 import Agent as Base, ScopedSignedReward


class Agent(Base):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        self.factor.reward.root_transition_bonus=0.3
