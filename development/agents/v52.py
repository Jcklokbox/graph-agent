"""AR6-05: retain soft local evidence consistently with exploratory decisions."""
from agents.v50 import Agent as Controller
from agents.v48 import ScopedSignedReward
from models.factor_information import SoftEventInformation


class SoftEventReward(SoftEventInformation,ScopedSignedReward):
    pass


class Agent(Controller):
    reward_class=SoftEventReward
