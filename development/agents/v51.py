"""AR6-03: once control is witnessed, replace payoff optimism with test value."""
from agents.v50 import Agent as Controller, InformativeReward


class VoIReward(InformativeReward):
    information_mode='replace'


class Agent(Controller):
    reward_class=VoIReward
