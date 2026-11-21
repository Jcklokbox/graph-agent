"""AR6-10: unchanged canonical sensory predictor with model-owned control."""
from agents.v21 import Agent as FactorController
from agents.v57 import MarginalInformation


class Agent(FactorController):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,
                         posterior=posterior,event=True,action_roles=False)
        f=self.factor
        f.reward=MarginalInformation(f.J,f.A,f.D)
