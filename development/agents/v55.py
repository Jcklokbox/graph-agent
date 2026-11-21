"""AR6-06: avoid counting the local predictive shift twice in test value."""
from agents.v54 import Agent as Controller, ScopedEventInformation


class ConsistentEventInformation(ScopedEventInformation):
    consistent_info_base=True


class Agent(Controller):
    reward_class=ConsistentEventInformation
