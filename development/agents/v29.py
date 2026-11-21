"""Factor-only retractable actuator roles and transient event hypotheses."""
from agents.v28 import Agent as Controller
from models.factor_reward import RetractableEventReward


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=12,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=RetractableEventReward(f.J,f.A,f.D)
