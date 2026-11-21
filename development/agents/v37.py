"""Factor-only mode-aware event controller (D-92)."""
from __future__ import annotations
import numpy as np
from models.factor_event_reward import ModeEventReward
from models.factor import CANONICAL, make_factor


class SparseModeReward(ModeEventReward):
    sparse_outcomes=True


class Agent:
    reward_type=SparseModeReward

    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5):
        self.factor=make_factor(CANONICAL,n_inputs,n_actions,value_index,seed)
        self.factor.attach_reward(action_roles=True, event=True, event_reward_factory=self.reward_type)
        self.draws=draws
        self.pending=None
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,541]))

    def step(self,x):
        if self.pending is not None:
            raise RuntimeError('previous action transition has not been learned')
        f=self.factor;u=self.rng.random((self.draws,f.J+4))
        q=f.two_step_values(x,u)
        peak=q.max();ties=np.flatnonzero(peak-q<=8.*np.spacing(peak))
        if len(ties)>1:
            c=int(f.reward.code(f._bits(x)));n=f.reward.trans_total[ties,c]
            ties=ties[n==n.min()]
        a=int(self.rng.choice(ties));f.predict(x,a)
        self.pending=(x,a)
        return a

    def learn(self,y):
        if self.pending is None:
            raise RuntimeError('action transition missing')
        self.factor.learn(*self.pending,y)
        self.pending=None
