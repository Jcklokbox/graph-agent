"""Factor-only controller using the shared successor reward hypothesis (D-68)."""
from agents.v21 import Agent as Controller


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=3,posterior=True,action_roles=True):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,
                         posterior=posterior,successor=True,action_roles=action_roles)
