"""Action selection and transition lifecycle for factor-only controllers."""
from __future__ import annotations

import numpy as np
from models.factor import CANONICAL, make_factor


class FactorController:
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=6, posterior=True, successor=False, action_roles=False, hybrid=False, controlled=False, phase=False, exception=False, optimistic=False, event=False, event_reward_factory=None):
        self.factor = make_factor(CANONICAL,n_inputs,n_actions,value_index,seed)
        self.factor.attach_reward(diagonal=posterior,successor=successor,action_roles=action_roles,hybrid=hybrid,controlled=controlled,phase=phase,exception=exception,optimistic=optimistic,event=event,
                                  event_reward_factory=event_reward_factory)
        self.posterior = posterior
        self.rng = np.random.default_rng(np.random.SeedSequence([seed,541]))
        self.draws = draws
        self.pending = None

    def step(self,x):
        if self.pending is not None:
            raise RuntimeError('learn the last transition before stepping')
        u=self.rng.random((self.draws,self.factor.J+3+int(self.factor.reward_control)))
        theta = self.factor.reward.sample(self.rng) if self.posterior else None
        q=self.factor.two_step_values(x,u,theta)
        a=int(self.rng.choice(np.flatnonzero(q==q.max())))
        self.factor.predict(x,a)  # prepare the one real learning cache
        self.pending=(x,a)
        return a

    def learn(self,y):
        if self.pending is None:
            raise RuntimeError('no action transition pending')
        self.factor.learn(*self.pending,y)
        self.pending=None
