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

try:
    import numba  # optional, for hot-path batching at Np=200

    @numba.njit
    def _njit_lik(y, E, K, m, out):
        # y (m,) 0/1, E (K,m) -> out (K,)
        for k in range(K):
            s = 0.0
            for j in range(m):
                p = E[k, j]
                if p < 1e-300:
                    p = 1e-300
                elif p > 1 - 1e-300:
                    p = 1 - 1e-300
                if y[j] > 0.5:
                    s += np.log(p)
                else:
                    s += np.log(1.0 - p)
            out[k] = np.exp(s)

    @numba.njit
    def _njit_T(ntr, alpha, eye, shrink, K, A, out):
        # ntr (A,K,K), eye (K,K)
        # pool = sum_A ntr + alpha + eye, normalized per row
        # out[a,i,j] = (ntr[a,i,j] + shrink*pool[i,j]) / (row_sum + shrink)
        pool = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                v = alpha + eye[i, j]
                for a in range(A):
                    v += ntr[a, i, j]
                pool[i, j] = v
        for i in range(K):
            rs = 0.0
            for j in range(K):
                rs += pool[i, j]
            # normalize pool row
            for j in range(K):
                pool[i, j] /= rs
        for a in range(A):
            for i in range(K):
                rs = 0.0
                for j in range(K):
                    rs += ntr[a, i, j]
                denom = rs + shrink
                for j in range(K):
                    out[a, i, j] = (ntr[a, i, j] + shrink * pool[i, j]) / denom

    @numba.njit
    def _njit_E(non, ntot, K, m, out):
        for k in range(K):
            d = ntot[k] + 1.0
            for j in range(m):
                out[k, j] = (non[k, j] + 0.5) / d

    @numba.njit
    def _njit_backward(Ts, liks, K, L, beta):
        # Ts (L+1,K,K), liks (L+1,K), beta (K,) in/out
        # backward: for k = L down to 1: beta = Ts[k] @ (liks[k]*beta)
        for kk in range(L, 0, -1):
            # tmp = liks[kk]*beta
            tmp = np.empty(K)
            for i in range(K):
                tmp[i] = liks[kk, i] * beta[i]
            nb = np.empty(K)
            for i in range(K):
                s = 0.0
                for j in range(K):
                    s += Ts[kk, i, j] * tmp[j]
                nb[i] = s
            s = 0.0
            for i in range(K):
                s += nb[i]
            if s == 0:
                s = 1.0
            for i in range(K):
                beta[i] = nb[i] / s

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


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
        # numba buffers
        if HAS_NUMBA:
            self._Tbuf = np.empty((n_actions, K, K))
            self._Ebuf = np.empty((K, n_channels))
            self._likbuf = np.empty(K)

    # ------------------------------------------------------------ parameters
    def T(self, a):
        """Transition matrix for action a (read-only, cached until ntr changes)."""
        t = self._Tc.get(a)
        if t is None:
            if HAS_NUMBA:
                # fast path: compute all A at once via njit, cache all
                _njit_T(self.ntr, self.alpha, self._eye, self.shrink, self.K, self.A, self._Tbuf)
                for aa in range(self.A):
                    self._Tc[aa] = self._Tbuf[aa].copy()
                return self._Tc[a]
            pool = self.ntr.sum(0) + self.alpha + self._eye
            pool /= pool.sum(1, keepdims=True)
            n = self.ntr[a]
            t = self._Tc[a] = (n + self.shrink * pool) / (n.sum(1, keepdims=True) + self.shrink)
        return t

    def E(self):
        """Emission matrix (read-only, cached until non/ntot change in _learn_oldest)."""
        e = self._Ec
        if e is None:
            if HAS_NUMBA:
                _njit_E(self.non, self.ntot, self.K, self.m, self._Ebuf)
                e = self._Ec = self._Ebuf.copy()
            else:
                e = self._Ec = (self.non + .5) / (self.ntot[:, None] + 1.)
        return e

    def _lik(self, y, E):
        y = np.asarray(y) > .5
        if HAS_NUMBA and self.m <= 8 and self.K <= 8:
            # use njit for small K,m (common case)
            y_arr = np.asarray(y, dtype=np.float64).flatten()
            # ensure y_arr length m
            if y_arr.size != self.m:
                # fallback
                L = np.where(y[None, :], E, 1. - E)
                if self.exclude:
                    L[:, self.exclude] = 1.
                return np.exp(np.log(np.maximum(L, 1e-300)).sum(1))
            out = np.empty(self.K)
            _njit_lik(y_arr, E, self.K, self.m, out)
            if self.exclude:
                # fix excluded channels (should be 1): adjust by dividing out their contribution
                # For m<=8 and exclude list small, we can correct: L originally had 1 for excluded,
                # njit included them as per E (which may not be 1). So fallback for exclude.
                # Use numpy path for exclude case
                L = np.where(y[None, :], E, 1. - E)
                L[:, self.exclude] = 1.
                return np.exp(np.log(np.maximum(L, 1e-300)).sum(1))
            return out
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
        if HAS_NUMBA and self.K <= 8 and len(self.buf) <= 6:
            # fast njit path
            Lb = len(self.buf)
            Ts = np.empty((Lb, self.K, self.K))
            liks = np.empty((Lb, self.K))
            for i, (_, aa, yy) in enumerate(self.buf):
                Ts[i] = self.T(aa)
                liks[i] = self._lik(yy, E)
            beta = np.ones(self.K)
            _njit_backward(Ts, liks, self.K, Lb - 1, beta)
            prev, a, y = self.buf.pop(0)
            # xi = prev[:,None] * Ts[0] * (liks[0]*beta)[None,:]
            tmp = liks[0] * beta
            xi = prev[:, None] * Ts[0] * tmp[None, :]
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
            return
        # original Python path
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
        self._Tc.clear()
        self.non += g[:, None] * y[None, :]
        self.ntot += g
        self._Ec = None


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


# ----------------------------------------------------------------------
# Batched HMM for Decomposer singleton groups (equal K, equal m)
# Cost = O(sum_k |P_k|) preserved: each group's HMMs are processed in one
# tensor, reduction order identical to per-HMM loop (bitwise).
class BatchedHMM:
    """Vectorized HMM for B parallel groups with same K and m."""

    def __init__(self, B, m, K, A, lag=4, decay=1., alpha=.2, sticky=2., shrink=2., seeds=None):
        self.B, self.m, self.K, self.A, self.L = B, m, K, A, lag
        self.decay, self.alpha, self.sticky, self.shrink = decay, alpha, sticky, shrink
        self.ntr = np.zeros((B, A, K, K))
        self._eye = sticky * np.eye(K)
        self.non = np.zeros((B, K, m))
        # per-HMM symmetry breaking via seeds
        if seeds is None:
            seeds = [0] * B
        for b, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            self.non[b] = rng.random((K, m)) * .5 + .25
        self.ntot = np.ones((B, K))
        self.b = np.full((B, K), 1.0 / K)  # will be set on observe
        self.buf_prev = []  # list of (B,K) prev beliefs
        self.buf_a = []     # list of (B,) actions (shared across B, but store per step)
        self.buf_y = []     # list of (B,m) observations
        self.started = False
        self._Tc = {}  # cache per action: (B,K,K)
        self._Ec = None

    def _T_all(self):
        # compute T for all A for all B
        # pool = sum_A ntr + alpha + eye, per B
        # Using vectorized numpy, same order as per-HMM loop (sum over A in 0..A-1)
        pool = self.ntr.sum(axis=0)  # actually sum over A? need per B: sum over A dim 1
        # ntr shape (B,A,K,K) -> sum over A => (B,K,K)
        pool = self.ntr.sum(axis=1) + self.alpha + self._eye  # (B,K,K)
        pool = pool / pool.sum(axis=2, keepdims=True)
        # for each a
        out = {}
        for a in range(self.A):
            n = self.ntr[:, a]  # (B,K,K)
            row_sum = n.sum(axis=2, keepdims=True)  # (B,K,1)
            out[a] = (n + self.shrink * pool) / (row_sum + self.shrink)
        self._Tc = out
        self._Ec = (self.non + 0.5) / (self.ntot[:, :, None] + 1.0)
        return out

    def observe(self, xs):
        # xs (B,m)
        if not self.started:
            if not self._Tc:
                self._T_all()
            # compute lik per B
            # E (B,K,m)
            E = self._Ec if self._Ec is not None else (self.non + 0.5) / (self.ntot[:, :, None] + 1.0)
            # lik_{b,k} = prod_j Bernoulli
            lik = np.ones((self.B, self.K))
            for b in range(self.B):
                # for each k, product over m
                # vectorized per b
                # we can do numpy but keep order identical to per-HMM: log sum
                for k in range(self.K):
                    s = 0.0
                    for j in range(self.m):
                        p = E[b, k, j]
                        y = 1 if xs[b, j] > 0.5 else 0
                        if y:
                            s += np.log(max(p, 1e-300))
                        else:
                            s += np.log(max(1 - p, 1e-300))
                    lik[b, k] = np.exp(s)
            self.b = lik / lik.sum(axis=1, keepdims=True)
            self.started = True

    def predict_marg(self, a):
        # return (B,m) predictive marginals: (b @ T_a) @ E
        if not self._Tc:
            self._T_all()
        T = self._Tc[a]  # (B,K,K)
        E = self._Ec  # (B,K,m)
        # b_T = b @ T  -> (B,K)
        b_T = np.einsum("bk,bkj->bj", self.b, T)
        # marg = b_T @ E -> (B,m)
        marg = np.einsum("bk,bkm->bm", b_T, E)
        return marg

    def update(self, a, ys):
        # ys (B,m)
        if not self.started:
            self.observe(ys)
            return
        if not self._Tc:
            self._T_all()
        E = self._Ec
        # lik per B
        lik = np.ones((self.B, self.K))
        for b in range(self.B):
            for k in range(self.K):
                s = 0.0
                for j in range(self.m):
                    p = E[b, k, j]
                    y = 1 if ys[b, j] > 0.5 else 0
                    if y:
                        s += np.log(max(p, 1e-300))
                    else:
                        s += np.log(max(1 - p, 1e-300))
                lik[b, k] = np.exp(s)
        # post = (b @ T) * lik
        T = self._Tc[a]
        b_T = np.einsum("bk,bkj->bj", self.b, T)
        post = b_T * lik
        post = post / post.sum(axis=1, keepdims=True)
        # buffer for learning
        self.buf_prev.append(self.b.copy())
        self.buf_a.append(a)
        self.buf_y.append(ys.copy())
        self.b = post
        if len(self.buf_prev) > self.L:
            self._learn_oldest()

    def _learn_oldest(self):
        Lb = len(self.buf_prev)
        # build Ts and liks for buffered steps
        # Ts: list of (B,K,K) per step, liks: (Lb,B,K)
        Ts = []
        liks = []
        E = self._Ec
        for i in range(Lb):
            a = self.buf_a[i]
            Ts.append(self._Tc[a])
            y = self.buf_y[i]
            lik = np.ones((self.B, self.K))
            for b in range(self.B):
                for k in range(self.K):
                    s = 0.0
                    for j in range(self.m):
                        p = E[b, k, j]
                        yv = 1 if y[b, j] > 0.5 else 0
                        if yv:
                            s += np.log(max(p, 1e-300))
                        else:
                            s += np.log(max(1 - p, 1e-300))
                    lik[b, k] = np.exp(s)
            liks.append(lik)
        # backward beta per B
        beta = np.ones((self.B, self.K))
        for k in range(Lb - 1, 0, -1):
            # beta = Ts[k] @ (liks[k]*beta)
            tmp = liks[k] * beta  # (B,K)
            # bmm: Ts[k] (B,K,K) @ tmp (B,K,1)
            nb = np.einsum("bkj,bj->bk", Ts[k], tmp)
            nb = nb / nb.sum(axis=1, keepdims=True)
            beta = nb
        # oldest
        prev = self.buf_prev.pop(0)
        a = self.buf_a.pop(0)
        y = self.buf_y.pop(0)
        Ts0 = Ts[0]
        lik0 = liks[0]
        # xi = prev[:,None] * Ts0 * (lik0*beta)[None,:] -> (B,K,K)
        tmp = lik0 * beta  # (B,K)
        xi = np.empty((self.B, self.K, self.K))
        for b in range(self.B):
            for i in range(self.K):
                for j in range(self.K):
                    xi[b, i, j] = prev[b, i] * Ts0[b, i, j] * tmp[b, j]
            xi[b] /= xi[b].sum()
        g = xi.sum(axis=1)  # (B,K)
        if self.decay < 1.0:
            self.ntr *= self.decay
            self.non *= self.decay
            self.ntot *= self.decay
        for b in range(self.B):
            self.ntr[b, a] += xi[b]
        self._Tc.clear()
        # non += g[:,None]*y
        for b in range(self.B):
            for k in range(self.K):
                for j in range(self.m):
                    self.non[b, k, j] += g[b, k] * y[b, j]
        self.ntot += g
        self._Ec = None
        self._T_all()  # refresh cache

