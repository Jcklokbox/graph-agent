"""D-100: signed likelihood of a supported configural reward factor."""
from agents.v36 import Agent as Controller, ModeEventReward
from models.factor_context_reward import SignedContextualOpportunities


class SignedEventReward(SignedContextualOpportunities,ModeEventReward):
    pass


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=SignedEventReward(f.J,f.A,f.D)
