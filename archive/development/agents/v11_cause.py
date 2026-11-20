"""Slice-1 candidate B: v11 as the emission model of latent causes.

Generative model:  z_t ~ T(z_{t-1}, .)  (sticky, CRP-style, at most K causes);
x_{t+1} ~ p_v11(. | x_t, z_t, a_t), where v11 runs on (x, one-hot z). The cause
is inferred by filtering with v11's predictive likelihood on the real channels;
learning is hard EM: after x_{t+1} is seen, v11 learns with the most probable
cause, and the transition counts are incremented along the hard path. A cause
that has never been used predicts with the uninformed prior .5 (Gershman, Blei
& Niv 2010). Latent channels are context only (never targets of v11).
"""
from __future__ import annotations

import numpy as np

from agents.v11 import PosteriorGraph

K_MAX, KAPPA, ALPHA, ALPHA_NEW, EPS = 4, 10., 1., .5, 1e-4


class Agent:
    inner_cls = PosteriorGraph

    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        self.n, self.K, self.v = n_inputs, K_MAX, value_index
        self.switches = 0
        self.inner = self.inner_cls(n_inputs + K_MAX, n_actions, value_index, seed=seed, **kw)
        self.counts = np.zeros((K_MAX, K_MAX))
        self.used = 1
        self.post = np.eye(K_MAX)[0]
        self.z_prev = None
        self.real = [i for i in range(n_inputs) if i != value_index]
        self.mask = tuple([True] * n_inputs + [False] * K_MAX)
        self.causes_created = 1

    # ------------------------------------------------------------ helpers
    def _T(self):
        K, u = self.K, self.used
        T = np.zeros((K, K))
        for z in range(u):
            row = np.zeros(K)
            row[:u] = self.counts[z, :u] + ALPHA
            row[z] += KAPPA
            if u < K:
                row[u] = ALPHA_NEW
            T[z] = row / row.sum()
        return T

    def _prior(self):
        return self.post @ self._T()

    def _aug(self, obs, z):
        obs = tuple(float(v) for v in obs)[:self.n]
        return obs + tuple(1. if k == z else 0. for k in range(self.K))

    def _cause_preds(self, x, action):
        """Per-cause predictions for causes 0..used (the last one new if used < K)."""
        P = [np.array(self.inner.predict_next(self._aug(x, z), action)[:self.n])
             for z in range(self.used)]
        if self.used < self.K:
            P.append(np.full(self.n, .5))
        return np.array(P)

    @property
    def _pending(self):
        return self.inner._pending

    @_pending.setter
    def _pending(self, v):
        self.inner._pending = v

    # ---------------------------------------------------------- interface
    def step(self, obs):
        z = int(np.argmax(self._prior()))
        return self.inner.step(self._aug(obs, min(z, self.used - 1)))

    def predict_next(self, obs, action):
        pi = self._prior()
        P = self._cause_preds(obs, action)
        w = pi[:len(P)] / pi[:len(P)].sum()
        return tuple(w @ P)

    def learn(self, next_obs):
        xn, action = self.inner._pending
        pi = self._prior()
        P = np.clip(self._cause_preds(xn, action), EPS, 1. - EPS)
        y = np.asarray(next_obs, float) > .5
        ll = np.where(y[None, self.real], np.log(P[:, self.real]),
                      np.log(1. - P[:, self.real])).sum(1)
        logp = np.log(np.maximum(pi[:len(P)], 1e-300)) + ll
        post = np.exp(logp - logp.max())
        post /= post.sum()
        z = int(np.argmax(post))
        if z == self.used:
            self.used += 1
            self.causes_created += 1
        if self.z_prev is not None:
            self.counts[self.z_prev, z] += 1.
            self.switches += int(z != self.z_prev)
        self.z_prev = z
        full = np.zeros(self.K)
        full[:len(post)] = post
        self.post = full
        self.inner._pending = (self._aug(xn, z), action)
        self.inner.learn(self._aug(next_obs, z), observed=self.mask)
