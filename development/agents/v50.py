"""AR6-03: repeatable, reachable decision information (experimental overlay)."""
from agents.v48 import Agent as Controller, ScopedSignedReward
from models.factor_information import RepeatableInformation


class InformativeReward(RepeatableInformation,ScopedSignedReward):
    pass


class Agent(Controller):
    reward_class=InformativeReward

    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=self.reward_class(f.J,f.A,f.D)
