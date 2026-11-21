"""D-85: projected controlled-event planning and actuated-dwell roles."""
from agents.v30 import Agent as Controller
from models.factor_reward import ProvisionalEventReward


class DwellEventReward(ProvisionalEventReward):
    factored_projection=True
    elapsed_only=True
    controlled_dwell=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=DwellEventReward(f.J,f.A,f.D)
