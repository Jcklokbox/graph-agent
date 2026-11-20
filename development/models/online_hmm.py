"""Probe model for slice 1: an online-learned hidden Markov model over binary
channels (not an agent; a predictor for tier T1).

Generative model (one process, K hidden states):
    z_{t+1} ~ T_a[z_t, :]                      a = the action taken at t
    x_{t+1, i} ~ Bernoulli(E[z_{t+1}, i])      channels conditionally independent

Learning is streaming Bayes with conjugate counts (Dirichlet rows for T_a,
Beta cells for E): after each observation the expected sufficient statistics
of one transition are added to the counts. With lag L = 0 the expectations use
the filtered belief (assumed-density filtering); with L > 0 they use the
fixed-lag smoothed posterior of the transition L steps back (a backward pass
over the last L observations with the current parameters), which lets later
evidence (e.g. an outcome after a delay) shape what an earlier state meant.
Counts are discounted by `decay` per step (old evidence about a possibly
changed world fades). Symmetry between states is broken by small random
emission pseudo-counts.

Action dependence: T_a = (n_a + c * Tbar) / (n_a.sum + c), where Tbar is the
action-pooled transition estimate (shrinkage towards "the action does not
matter"). Cost per step O(L K^2 + K m).
"""
from __future__ import annotations

import numpy as np


class OnlineHMM:
    def __init__(self, n_channels, n_actions, K=8, lag=4, decay=1., alpha=.2,
                 sticky=2., shrink=2., seed=0, exclude=()):
        rng = np.random.default_rng(seed)
        self.K, self.m, self.A, self.L = K, n_channels, n_actions, lag
        self.decay, self.alpha, self.sticky, self.shrink = decay, alpha, sticky, shrink
        self.ntr = np.zeros((n_actions, K, K))
        self.non = rng.random((K, n_channels)) * .5 + .25      # symmetry breaking
        self.ntot = np.ones(K)
        self.b = None
        self.buf = []          # (alpha_prev, action, y) of the last L+1 transitions
        self.exclude = list(exclude)

    # ------------------------------------------------------------ parameters
    def T(self, a):
        K = self.K
        pool = self.ntr.sum(0) + self.alpha + self.sticky * np.eye(K)
        pool /= pool.sum(1, keepdims=True)
        n = self.ntr[a]
        return (n + self.shrink * pool) / (n.sum(1, keepdims=True) + self.shrink)

    def E(self):
        return (self.non + .5) / (self.ntot[:, None] + 1.)

    def _lik(self, y, E):
        y = np.asarray(y) > .5
        L = np.where(y[None, :], E, 1. - E)
        if self.exclude:
            L[:, self.exclude] = 1.
        return np.exp(np.log(np.maximum(L, 1e-300)).sum(1))

    # ------------------------------------------------------------- interface
    def observe(self, x0):
        lik = self._lik(x0, self.E())
        self.b = lik / lik.sum()

    def predict(self, a):
        return (self.b @ self.T(a)) @ self.E()

    def update(self, a, y):
        E = self.E()
        prev = self.b
        lik = self._lik(y, E)
        post = (prev @ self.T(a)) * lik
        self.b = post / post.sum()
        self.buf.append((prev, a, np.asarray(y, float)))
        if len(self.buf) > self.L:
            self._learn_oldest(E)

    def _learn_oldest(self, E):
        # backward messages over the buffered transitions with current parameters
        beta = np.ones(self.K)
        Ts = [self.T(a) for _, a, _ in self.buf]
        liks = [self._lik(y, E) for _, _, y in self.buf]
        for k in range(len(self.buf) - 1, 0, -1):
            beta = Ts[k] @ (liks[k] * beta)
            beta /= beta.sum()
        prev, a, y = self.buf.pop(0)
        xi = prev[:, None] * Ts[0] * (liks[0] * beta)[None, :]
        xi /= xi.sum()
        g = xi.sum(0)
        if self.decay < 1.:
            self.ntr *= self.decay
            self.non *= self.decay
            self.ntot *= self.decay
        self.ntr[a] += xi
        self.non += g[:, None] * y[None, :]
        self.ntot += g


class OnlineHMMPredictor:
    """T1 adapter: predict(x, a) / learn(x, a, y) on the shared stream."""
    KW = {}

    def __init__(self, n, A, v, seed=0):
        self.h = OnlineHMM(n, A, seed=seed, exclude=[v], **self.KW)
        self.started = False

    def predict(self, x, a):
        if not self.started:
            self.h.observe(x)
            self.started = True
        return self.h.predict(a)

    def learn(self, x, a, y):
        self.h.update(a, y)


def make_predictor(name, **kw):
    return type(name, (OnlineHMMPredictor,), {"KW": kw})
