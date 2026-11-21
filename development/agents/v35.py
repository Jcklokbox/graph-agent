"""D-87: D-81 complete onset planning with protected actuator dwell."""
from agents.v33 import Agent as Controller, DwellEventReward


class FullEventReward(DwellEventReward):
    factored_projection=False
    elapsed_only=False


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=FullEventReward(f.J,f.A,f.D)
