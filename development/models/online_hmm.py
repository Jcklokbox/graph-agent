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
        self._eye = sticky * np.eye(K)
        self._Ec = None        # E() cache
        self._Tc = {}          # T(a) cache; cleared whenever ntr changes (only in _learn_oldest)
        self.non = rng.random((K, n_channels)) * .5 + .25      # symmetry breaking
        self.ntot = np.ones(K)
        self.b = None
        self.buf = []          # (alpha_prev, action, y) of the last L+1 transitions
        self.exclude = list(exclude)
        self._lc = {}          # current-emission likelihoods for binary observations

    # ------------------------------------------------------------ parameters
    def T(self, a):
        """Transition matrix for action a (read-only, cached until ntr changes)."""
        t = self._Tc.get(a)
        if t is None:
            pool = self.ntr.sum(0) + self.alpha + self._eye
            pool /= pool.sum(1, keepdims=True)
            n = self.ntr[a]
            t = self._Tc[a] = (n + self.shrink * pool) / (n.sum(1, keepdims=True) + self.shrink)
        return t

    def E(self):
        """Emission matrix (read-only, cached until non/ntot change in _learn_oldest)."""
        e = self._Ec
        if e is None:
            e = self._Ec = (self.non + .5) / (self.ntot[:, None] + 1.)
        return e

    def _lik(self, y, E):
        bits = np.asarray(y) > .5
        key = bits.tobytes()
        cached = self._lc.get(key)
        if cached is not None and E is self._Ec:
            return cached
        return self._lik_many(bits[None], E)[0]

    def _lik_many(self, bits, E):
        # Batch over observations; keep the scalar channel reduction order.
        L = np.where(bits[:, None, :], E, 1. - E)
        if self.exclude:
            L[:, :, self.exclude] = 1.
        ll = np.exp(np.log(np.maximum(L, 1e-300)).sum(-1))
        if E is self._Ec:
            self._lc.update((b.tobytes(), v) for b, v in zip(bits, ll))
        return ll

    # ------------------------------------------------------------- interface
    def observe(self, x0):
        lik = self._lik(x0, self.E())
        self.b = lik / lik.sum()

    def predict(self, a):
        return (self.b @ self.T(a)) @ self.E()

    def update(self, a, y, prior=None, lik=None):
        E = self.E()
        prev = self.b
        if lik is None:
            lik = self._lik(y, E)
        post = (prev @ self.T(a) if prior is None else prior) * lik
        self.b = post / post.sum()
        self.buf.append((prev, a, np.asarray(y, float)))
        if len(self.buf) > self.L:
            self._learn_oldest(E, lik)

    def _learn_oldest(self, E, latest):
        # backward messages over the buffered transitions with current parameters
        beta = np.ones(self.K)
        Ts = [self.T(a) for _, a, _ in self.buf]
        if len(self.buf) > 1:
            bits = np.asarray([y for _, _, y in self.buf[:-1]]) > .5
            liks = self._lik_many(bits, E)
        else:
            liks = [latest]
        for k in range(len(self.buf) - 1, 0, -1):
            beta = Ts[k] @ ((latest if k == len(self.buf) - 1 else liks[k]) * beta)
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
        self._Tc.clear()
        self.non += g[:, None] * y[None, :]
        self.ntot += g
        self._Ec = None
        self._lc.clear()


def update_population(entries, a):
    """Update independent HMMs in equal-shaped batches without cross-model reductions.

    Entries carry the already-scored predictive belief and emission likelihood. Keep
    short/discounted buffers on the general scalar path.
    """
    buckets = {}
    for h, y, prior, lik in entries:
        if len(h.buf) != h.L or h.decay != 1. or h.exclude:
            h.update(a, y, prior=prior, lik=lik)
        else:
            buckets.setdefault((h.K, h.m, h.L), []).append((h, y, prior, lik))
    for (K, _, L), batch in buckets.items():
        if len(batch) == 1:
            h, y, prior, lik = batch[0]
            h.update(a, y, prior=prior, lik=lik)
            continue
        hs, ys, priors, latest = zip(*batch)
        beliefs = np.asarray(priors) * np.asarray(latest)
        beliefs /= beliefs.sum(-1, keepdims=True)
        for h, y, b in zip(hs, ys, beliefs):
            h.buf.append((h.b, a, y))
            h.b = b
        Ts = np.asarray([[h.T(ac) for _, ac, _ in h.buf] for h in hs])
        if L:
            bits = np.asarray([[y for _, _, y in h.buf[:-1]] for h in hs]) > .5
            E = np.asarray([h.E() for h in hs])[:, None]
            v = np.where(bits[:, :, None], E, 1. - E)
            liks = np.exp(np.log(np.maximum(v, 1e-300)).sum(-1))
            beta = (Ts[:, L] @ np.asarray(latest)[..., None])[..., 0]
            beta /= beta.sum(-1, keepdims=True)
            for i in range(L - 1, 0, -1):
                beta = (Ts[:, i] @ (liks[:, i] * beta)[..., None])[..., 0]
                beta /= beta.sum(-1, keepdims=True)
            li = liks[:, 0] * beta
        else:
            li = np.asarray(latest)
        prev, acts, target = zip(*(h.buf.pop(0) for h in hs))
        xi = np.asarray(prev)[:, :, None] * Ts[:, 0] * li[:, None, :]
        xi /= xi.reshape(len(hs), K * K).sum(-1)[:, None, None]
        gamma = xi.sum(1)
        emit = gamma[:, :, None] * np.asarray(target)[:, None, :]
        for h, ac, xx, g, e in zip(hs, acts, xi, gamma, emit):
            h.ntr[ac] += xx
            h.non += e
            h.ntot += g
            h._Ec = None
            h._Tc.clear()
            h._lc.clear()


class PackedHMM(OnlineHMM):
    """Unit-decay, fully observed HMM used by the Decomposer population.

    The same fixed-lag filter as OnlineHMM, with contiguous ring buffers and one
    fused smoothing/count update. Other witness settings use the scalar class.
    """

    def __init__(self, m, actions, K, seed):
        super().__init__(m, actions, K=K, seed=seed)
        self._trans = np.empty_like(self.ntr)
        self._emit = np.empty_like(self.non)
        self._dirty = True
        self._ix = self._age = 0
        cap = self.L + 1
        self._hb = np.empty((cap, K))
        self._ha = np.empty(cap, dtype=np.int64)
        self._hy = np.empty((cap, m))

    def _refresh(self):
        if self._dirty:
            from models.kernels import hmm_params
            hmm_params(self.ntr, self.non, self.ntot, self.alpha, self.sticky, self.shrink,
                       self._trans, self._emit)
            self._dirty = False

    def T(self, a):
        self._refresh()
        return self._trans[a]

    def E(self):
        self._refresh()
        return self._emit

    def forward(self, a, read):
        from models.kernels import hmm_forward
        self._refresh()
        return hmm_forward(self.b, self._trans[a], self._emit, read)

    def score_learn(self, a, y, prior):
        from models.kernels import hmm_step
        self._refresh()
        score = hmm_step(y, a, prior, self.b, self._trans, self._emit, self._hb, self._ha,
                         self._hy, self._ix, self._age, self.ntr, self.non, self.ntot)
        self._age = min(self._age + 1, self.L + 1)
        self._ix = (self._ix + 1) % (self.L + 1)
        self._dirty = self._age > self.L
        return score

    def update(self, a, y, prior=None, lik=None):
        if prior is None:
            prior, _ = self.forward(a, False)
        self.score_learn(a, np.asarray(y, float), prior)


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
