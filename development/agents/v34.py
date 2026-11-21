"""D-86: posterior transition uncertainty for an actuator row."""
from agents.v33 import Agent as Controller, DwellEventReward


class FrontierEventReward(DwellEventReward):
    motion_optimism=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=FrontierEventReward(f.J,f.A,f.D)
