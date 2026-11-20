"""v17 (slice 3b-0, D-20): v12 unchanged + an anonymous latent-cause predictor per channel,
mixed per channel by posterior weight exactly as in v15 (discounted cumulative log loss);
the planner sees the mixture at depth 0.

Second predictor for channel j: `models/latent_cause_lin.LatentCauseSubAdd` — sticky-CRP hidden
causes (learned hazard) × Bayesian average over parent sets of size <= 2 plus one additive model,
with candidates = every non-value channel at t (j's own included) and the action (one-hot);
200 particles (D-19). The same model for every channel: no role knowledge. This is the
interim agent that tests whether T2 abilities of the witness (D-18: T2 v1 10/10) reach an agent
through the per-channel mixture; its cost is quadratic in the number of channels (all channels
are candidates) — the bounded-candidate version is 3b-1.
"""
from __future__ import annotations

import numpy as np

from agents.v12 import PosteriorGraph12
from models.latent_cause_lin import LatentCauseSubAdd

GAMMA, EPS, PARTICLES = .995, 1e-4, 200


class Agent(PosteriorGraph12):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        super().__init__(n_inputs, n_actions, value_index, seed=seed, **kw)
        self.chans = [j for j in range(n_inputs) if j != value_index]
        m = len(self.chans) + n_actions
        self.lcs = {j: LatentCauseSubAdd(emit=[], cues=list(range(m)), us=-1, hazard=None,
                                         n=PARTICLES, seed=seed + 17 * j) for j in self.chans}
        self.mloss = np.zeros((2, n_inputs))
        self._lmemo = {}

    def _vec(self, x, action):
        v = [float(float(x[j]) > .5) for j in self.chans] + [0.] * self.n_actions
        v[len(self.chans) + action] = 1.
        return v

    def _lc_pred(self, x, action):
        key = (tuple(x), action)
        hit = self._lmemo.get(key)
        if hit is None:
            v = self._vec(x, action)
            hit = np.full(self.n_inputs, .5)
            for j, lc in self.lcs.items():
                hit[j] = lc.predict_us(v)
            self._lmemo[key] = hit
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
        mix = w[0] * np.asarray(own) + w[1] * self._lc_pred(x, action)
        return tuple(float(v) for v in mix)

    def _invalidate(self):
        super()._invalidate()
        self._lmemo = {}

    def learn(self, next_obs, **kw):
        x, action = self._pending
        own = np.asarray(PosteriorGraph12._predict_next(self, x, action, 0))
        pl = self._lc_pred(x, action)
        y = np.asarray([float(v) for v in next_obs]) > .5
        P = np.clip(np.array([own, pl]), EPS, 1. - EPS)
        self.mloss = GAMMA * self.mloss - np.where(y[None, :], np.log(P), np.log(1. - P))
        v = self._vec(x, action)
        for j, lc in self.lcs.items():
            lc.observe(v, bool(y[j]))
        super().learn(next_obs, **kw)
