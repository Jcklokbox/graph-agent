"""v15 (slice 3, C0): v12 unchanged + the slice-3 decomposition witness as a second
predictor, mixed per channel by posterior weight; planning uses the mixture.

Generative reading: for every channel two hypotheses about its generative process —
(a) v12's context tables (fast association within the observed configuration), (b) a member
of a slowly discovered group of channels driven by an independent hidden process
(models/decompose.py, `decompw`: online HMMs with K in {2,3,4,6} and the window family per
group, Bayesian partition by fixed-horizon Bayes factors). Prediction for channel i:

    p_i = sum_m w_{m,i} p_{m,i},   w_{m,i} prop. to exp(-L_{m,i}),
    L_{m,i} = discounted cumulative log loss (GAMMA per step: switching prior)

as in slice-1 A' (agents/v11_bma.py, S1-2: GC .81 on T1 v1). New against A': (1) the second
model is the decomposition witness instead of one global HMM; (2) the planner sees the
mixture: v12's depth-0 next-observation prediction (the input of the expected continuation)
is replaced by the mixture. The immediate value and v12's model-based value are v12's own; the
value channel is predicted by v12 only. Probes at a hypothetical observation x (T2) condition
the witness on x (Decomposer.peek_all).
"""
from __future__ import annotations

import numpy as np

from agents.v12 import PosteriorGraph12
from models.decompose import Decomposer

GAMMA, EPS = .995, 1e-4


class Agent(PosteriorGraph12):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        super().__init__(n_inputs, n_actions, value_index, seed=seed, **kw)
        self.dec = Decomposer(n_inputs, n_actions, value_index, seed, windows=True)
        self.mloss = np.zeros((2, n_inputs))
        self._now = None                      # binary key of the observation the witness absorbed last
        self._dmemo = {}

    # ------------------------------------------------------------ witness side
    def _dec_pred(self, x, action):
        key = (x, action)
        hit = self._dmemo.get(key)
        if hit is None:
            if self._now is None or self._now == self._key(x):
                hit = self.dec.predict(x, action)
            else:
                hit = self.dec.peek_all(x, action)
            self._dmemo[key] = hit
        return hit

    def _weights(self):
        w = np.exp(-(self.mloss - self.mloss.min(0)))
        w = w / w.sum(0)
        w[:, self.value_index] = (1., 0.)
        return w

    def _predict_next(self, x, action, depth):
        own = PosteriorGraph12._predict_next(self, x, action, depth)
        if depth != 0:
            return own
        w = self._weights()
        mix = w[0] * np.asarray(own) + w[1] * self._dec_pred(x, action)
        return tuple(float(v) for v in mix)

    def _invalidate(self):
        super()._invalidate()
        self._dmemo = {}

    # ------------------------------------------------------------ learning
    def learn(self, next_obs, **kw):
        x, action = self._pending
        own = np.asarray(PosteriorGraph12._predict_next(self, x, action, 0))
        pd = self.dec.predict(x, action)          # also sets the witness's prior for `action`
        y = np.asarray([float(v) for v in next_obs]) > .5
        P = np.clip(np.array([own, pd]), EPS, 1. - EPS)
        self.mloss = GAMMA * self.mloss - np.where(y[None, :], np.log(P), np.log(1. - P))
        self.dec.learn(x, action, y.astype(float))
        super().learn(next_obs, **kw)
        self._now = self._key(next_obs)

    def _key(self, x):
        v = self.value_index
        return tuple(float(u) > .5 for j, u in enumerate(x) if j != v)

    def partition(self):
        return self.dec.partition()
