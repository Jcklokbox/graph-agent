"""D-90: recurrent bounded factor-control and reward modes."""
from agents.v34 import Agent as Controller, FrontierEventReward
from models.factor_regime_reward import RecurrentRegimes


class ModeEventReward(RecurrentRegimes,FrontierEventReward):
    pass


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ModeEventReward(f.J,f.A,f.D)
