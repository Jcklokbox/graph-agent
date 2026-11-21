"""D-102: a minimal unary event factor plus a supported signed refinement."""
from agents.v43 import Agent as Controller, SignedEventReward


class SparseSignedReward(SignedEventReward):
    P=1
    sparse_outcomes=True


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=SparseSignedReward(f.J,f.A,f.D)
