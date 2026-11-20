"""Slice-1 candidate (a), prototype: v11 reading a learned latent state.

An online HMM (models/online_hmm.py) runs on the raw channels; the index of its
most probable hidden state is appended to the observation as K one-hot
"latent channels" (cf. orbitofrontal state representations of hidden task
states, Wilson et al. 2014). v11 learns on the augmented vector unchanged, so
latent states can enter its contexts, expectation windows and attention like
any channel. predict_next returns only the real channels.
"""
from __future__ import annotations

import numpy as np

from agents.v11 import PosteriorGraph
from models.online_hmm import OnlineHMM

K_LATENT, LAG = 4, 4


class Agent:
    SOFT = False      # L2: predict sum_z b(z) p_v11(x' | x, z) instead of using argmax z
    CONTEXT_ONLY = False   # L3: latent channels are context only, never targets
    def __init__(self, n_inputs, n_actions, value_index, seed=0, K=None, lag=None, **kw):
        self.n, self.K = n_inputs, K or K_LATENT
        self.h = OnlineHMM(n_inputs, n_actions, K=self.K, lag=lag or LAG, seed=seed,
                           exclude=[value_index])
        self.inner = PosteriorGraph(n_inputs + self.K, n_actions, value_index, seed=seed, **kw)
        self.started = False

    def _onehot(self):
        z = np.zeros(self.K)
        z[int(np.argmax(self.h.b))] = 1.
        return tuple(float(v) for v in z)

    def _aug(self, obs):
        obs = tuple(float(v) for v in obs)
        return obs if len(obs) == self.n + self.K else obs + self._onehot()

    @property
    def _pending(self):
        return self.inner._pending

    @_pending.setter
    def _pending(self, v):
        self.inner._pending = v

    def step(self, obs):
        if not self.started:
            self.h.observe(obs)
            self.started = True
        return self.inner.step(self._aug(obs))

    def learn(self, next_obs):
        action = self.inner._pending[1]
        self.h.update(action, next_obs)
        mask = (True,) * self.n + (not self.CONTEXT_ONLY,) * self.K
        self.inner.learn(tuple(float(v) for v in next_obs) + self._onehot(), observed=mask)

    def predict_next(self, obs, action):
        if not self.SOFT:
            return self.inner.predict_next(self._aug(obs), action)[:self.n]
        base = tuple(float(v) for v in obs)[:self.n]
        b = self.h.b
        out = np.zeros(self.n)
        for z in range(self.K):
            if b[z] > 1e-3:
                oh = tuple(1. if k == z else 0. for k in range(self.K))
                out += b[z] * np.array(self.inner.predict_next(base + oh, action)[:self.n])
        return tuple(out / b[b > 1e-3].sum())
