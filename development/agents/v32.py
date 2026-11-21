"""Factor-only controlled event head with exact sparse sensory projection."""
from agents.v31 import Agent as Controller
from models.factor_reward import ElapsedEventReward


class ProjectedEventReward(ElapsedEventReward):
    factored_projection=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ProjectedEventReward(f.J,f.A,f.D)
