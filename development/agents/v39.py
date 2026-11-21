"""D-95: sparse configural opportunity, retaining factor-only reward and state."""
from agents.v36 import Agent as Controller, ModeEventReward
from models.factor_context_reward import ContextualOpportunities


class ConfiguralEventReward(ContextualOpportunities,ModeEventReward):
    pass


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ConfiguralEventReward(f.J,f.A,f.D)
