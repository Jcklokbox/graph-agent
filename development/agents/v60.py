"""AR7: common-support signed reward selection with canonical sensory inference."""
from agents.v59 import Agent as Controller, SharedProjectedReward
from models.factor_joint_reward import JointEvidence


class JointReward(JointEvidence,SharedProjectedReward):
    pass


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=JointReward(f.J,f.A,f.D)
