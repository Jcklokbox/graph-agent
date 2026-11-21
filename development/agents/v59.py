"""AR6-16: share the fixed and optimistic projection, without a policy change."""
from agents.v58 import Agent as Controller
from agents.v57 import MarginalInformation


class SharedProjectedReward(MarginalInformation):
    share_fixed_projection=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=SharedProjectedReward(f.J,f.A,f.D)
