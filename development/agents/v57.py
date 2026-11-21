"""AR6-08: coincident atoms pay for a test relative to the conditional incumbent."""
from agents.v56 import Agent as Controller, SignedSelection


class MarginalInformation(SignedSelection):
    intrinsic_opportunity=True


class Agent(Controller):
    reward_class=MarginalInformation
