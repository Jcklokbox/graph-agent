"""AR7: common-support signed reward selection with canonical sensory inference."""
from agents.factor_controller import FactorController
from models.factor_event_reward import SignedSelection
from models.factor_joint_reward import JointEvidence


class JointReward(JointEvidence, SignedSelection):
    share_fixed_projection = True


class Agent(FactorController):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws,
                         posterior=posterior, event=True, event_reward_factory=JointReward)
