"""Factor-only controlled reward-type likelihood (D-76)."""
from agents.v21 import Agent as Controller


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=12,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,
                         posterior=posterior,action_roles=True,event=True)
