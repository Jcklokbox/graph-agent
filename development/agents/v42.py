"""D-99: retain an opportunity on the sibling of a confirmed main effect."""
from agents.v39 import Agent as Controller, ConfiguralEventReward


class RefiningEventReward(ConfiguralEventReward):
    context_refines_known=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=RefiningEventReward(f.J,f.A,f.D)
