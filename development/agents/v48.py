"""D-108: configural exploration refines an earned local event parent."""
from agents.v44 import Agent as Controller, SparseSignedReward


class ScopedSignedReward(SparseSignedReward):
    context_requires_parent=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ScopedSignedReward(f.J,f.A,f.D)
