"""Bounded controlled event law with provisional onset hypotheses (D-81)."""
from agents.v29 import Agent as Controller
from models.factor_reward import ProvisionalEventReward


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ProvisionalEventReward(f.J,f.A,f.D)
