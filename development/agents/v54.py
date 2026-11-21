"""AR6-06: a learned payoff scope permits tests only of its own refinement."""
from agents.v53 import Agent as Controller, PairedEventReward


class ScopedEventInformation(PairedEventReward):
    restrict_info_parent=True


class Agent(Controller):
    reward_class=ScopedEventInformation
