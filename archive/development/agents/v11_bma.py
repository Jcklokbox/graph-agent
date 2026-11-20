"""Slice-1 candidate A': per-channel Bayesian model averaging of v11 and an
online HMM (models/online_hmm.py).

For every channel i the prediction is  sum_m w_{m,i} p_{m,i}  with
w_{m,i} proportional to exp(-L_{m,i}), where L_{m,i} is model m's discounted
cumulative log loss on channel i (discount GAMMA per step: a switching prior).
This is the posterior over two hypotheses about each channel's generative
process. The models do not share information, and only predictions are
combined: actions are still chosen by v11's planner.
"""
from __future__ import annotations

import numpy as np

from agents.v11 import PosteriorGraph
from models.online_hmm import OnlineHMM

GAMMA, EPS = .995, 1e-4


class Agent:
    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        self.n = n_inputs
        self.inner = PosteriorGraph(n_inputs, n_actions, value_index, seed=seed, **kw)
        self.h = OnlineHMM(n_inputs, n_actions, K=8, lag=4, seed=seed, exclude=[value_index])
        self.loss = np.zeros((2, n_inputs))
        self.started = False

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
        return self.inner.step(obs)

    def _parts(self, obs, action):
        return np.array([self.inner.predict_next(obs, action), self.h.predict(action)])

    def predict_next(self, obs, action):
        P = self._parts(obs, action)
        w = np.exp(-(self.loss - self.loss.min(0)))
        return tuple((w * P).sum(0) / w.sum(0))

    def learn(self, next_obs):
        x, action = self.inner._pending
        P = np.clip(self._parts(x, action), EPS, 1. - EPS)
        y = np.asarray(next_obs, float) > .5
        self.loss = GAMMA * self.loss - np.where(y[None, :], np.log(P), np.log(1. - P))
        self.h.update(action, next_obs)
        self.inner.learn(next_obs)
