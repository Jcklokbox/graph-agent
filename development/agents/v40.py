"""D-96: sensory-informed exploratory probes of a real bounded reward law."""
from agents.v36 import Agent as Controller, ModeEventReward
from models.factor_context_reward import SuccessorOpportunities


class BridgeEventReward(SuccessorOpportunities,ModeEventReward):
    pass


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=BridgeEventReward(f.J,f.A,f.D)
