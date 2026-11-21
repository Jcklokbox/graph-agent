"""Factor-only elapsed-event exploration with support-aware actuator prior."""
from agents.v30 import Agent as Controller
from models.factor_reward import ElapsedEventReward


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ElapsedEventReward(f.J,f.A,f.D)
