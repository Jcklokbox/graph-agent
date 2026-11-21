"""D-109: credible signed events with cause-scoped refinement."""
from agents.v47 import Agent as Controller, CredibleEventReward


class RefinedCredibleReward(CredibleEventReward):
    context_requires_parent=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=RefinedCredibleReward(f.J,f.A,f.D)
