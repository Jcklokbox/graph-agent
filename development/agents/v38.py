"""D-93: factor-only state transition evidence for motor relabeling."""
from agents.v37 import Agent as Controller, SparseModeReward


class RelabelledModeReward(SparseModeReward):
    relabel_actions=True


class Agent(Controller):
    reward_type=RelabelledModeReward
