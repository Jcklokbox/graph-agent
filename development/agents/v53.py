"""AR6-05: soft evidence only in empirically configured onset pairs."""
from agents.v52 import Agent as Controller, SoftEventReward


class PairedEventReward(SoftEventReward):
    scoped_pair_only=True


class Agent(Controller):
    reward_class=PairedEventReward
