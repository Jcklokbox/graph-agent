"""Latent causes with an additive (linear-Gaussian) US model per cause — witness for D-17.

Same sticky-CRP over hidden causes as `LatentCauseSMC` (learned hazard on a grid, optimal
proposal, Rao-Blackwellised), but within each cause the US is predicted additively from the
cue vector (plus a bias) by Bayesian linear regression (a Kalman filter on static weights, as in
Dayan & Kakade 2001 and Gershman, Norman & Niv 2015) instead of a table over configurations.
Additivity gives cue competition (blocking, conditioned inhibition); the number of statistics is
quadratic in the number of cues per cause instead of exponential. Cues and emitted channels are
anonymous lists — the model does not know which channel is a context.
"""
from __future__ import annotations

import numpy as np

from models.latent_cause import LatentCauseSMC


class LatentCauseLin(LatentCauseSMC):
    def __init__(self, emit, cues, us, s0=1., noise=.25, **kw):
        super().__init__(emit, cues, us, **kw)
        N, K, d = self.N, self.K, len(self.cues) + 1
        self.noise = noise
        self.mu = np.zeros((N, K, d))
        self.S = np.broadcast_to(np.eye(d) * s0, (N, K, d, d)).copy()

    def _vec(self, x):
        return np.append((np.asarray(x)[self.cues] > .5).astype(float), 1.)

    def _pred(self, v):
        m = self.mu @ v                                        # (N, K)
        var = np.einsum("nkij,i,j->nk", self.S, v, v) + self.noise
        return m, var

    def predict_us(self, x):
        v = self._vec(x)
        m, _ = self._pred(v)
        joint = self._prior() * self._lik_x(x)
        w = np.exp(self.logw - self.logw.max())[:, None]
        return float((w * joint * np.clip(m, .01, .99)).sum() / (w * joint).sum())

    def observe(self, x, us_next):
        N = self.N
        v = self._vec(x)
        y = 1. if us_next else 0.
        m, var = self._pred(v)
        lik = np.exp(-.5 * (y - m) ** 2 / var) / np.sqrt(var)
        joint = self._prior() * self._lik_x(x) * lik
        tot = joint.sum(1)
        self.logw += np.log(tot)
        z = self._advance(joint / tot[:, None])
        ar = np.arange(N)
        xe = np.asarray(x)[self.emit] > .5
        self.t1[ar, z] += xe
        self.t0[ar, z] += ~xe
        Sz = self.S[ar, z]                                     # (N, d, d)
        g = Sz @ v / var[ar, z][:, None]                       # (N, d)
        self.mu[ar, z] += g * (y - m[ar, z])[:, None]
        self.S[ar, z] = Sz - g[:, :, None] * g[:, None, :] * var[ar, z][:, None, None]
        self._resample()

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z", "mu", "S"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)


class LatentCauseSub(LatentCauseSMC):
    """Latent causes x Bayesian structure averaging (Courville, Daw & Touretzky 2006 style).

    Within each cause the US is predicted by a mixture over small parent sets S (|S| <= smax,
    including the empty set): model S keeps Beta counts over the configurations of S only;
    its weight is its marginal likelihood on the cause's own data (Bayesian Occam: a larger set
    must earn its extra cells). Cue competition arises because a set that predicted well from the
    start out-weighs one that only became predictive later (blocking). Statistics per cause:
    sum_S 2^|S| cells — polynomial in the number of cues for bounded smax.
    """

    def __init__(self, emit, cues, us, smax=2, **kw):
        super().__init__(emit, [], us, **kw)      # no full-configuration table (exponential)
        self.cues = list(cues)
        from itertools import combinations
        m = len(self.cues)
        self.sets = [s for r in range(smax + 1) for s in combinations(range(m), r)]
        self.off = np.cumsum([0] + [2 ** len(s) for s in self.sets])
        N, K, C = self.N, self.K, int(self.off[-1])
        self.q1 = np.zeros((N, K, C))
        self.q0 = np.zeros((N, K, C))
        self.ev = np.zeros((N, K, len(self.sets)))

    def _cells(self, x):
        b = [x[c] > .5 for c in self.cues]
        return np.array([self.off[i] + sum(1 << j for j, k in enumerate(s) if b[k])
                         for i, s in enumerate(self.sets)])

    def _sub_pred(self, cells):
        n1, n0 = self.q1[:, :, cells], self.q0[:, :, cells]          # (N, K, S)
        p = (n1 + self.a) / (n1 + n0 + 2 * self.a)
        lw = self.ev - self.ev.max(2, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(2, keepdims=True)
        return p, (w * p).sum(2)

    def predict_us(self, x):
        _, pu = self._sub_pred(self._cells(x))
        joint = self._prior() * self._lik_x(x)
        w = np.exp(self.logw - self.logw.max())[:, None]
        return float((w * joint * pu).sum() / (w * joint).sum())

    def observe(self, x, us_next):
        N = self.N
        cells = self._cells(x)
        p, pu = self._sub_pred(cells)
        joint = self._prior() * self._lik_x(x) * (pu if us_next else 1. - pu)
        tot = joint.sum(1)
        self.logw += np.log(tot)
        z = self._advance(joint / tot[:, None])
        ar = np.arange(N)
        xe = np.asarray(x)[self.emit] > .5
        self.t1[ar, z] += xe
        self.t0[ar, z] += ~xe
        pz = p[ar, z]                                                # (N, S)
        self.ev[ar, z] += np.log(pz if us_next else 1. - pz)
        if us_next:
            self.q1[ar[:, None], z[:, None], cells[None, :]] += 1.
        else:
            self.q0[ar[:, None], z[:, None], cells[None, :]] += 1.
        self._resample()

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z", "q1", "q0", "ev"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)


class LatentCauseSubAdd(LatentCauseSub):
    """`LatentCauseSub` plus the elemental hypothesis as one more member of the structure
    average: an additive model over all cues (Bayesian linear regression, prediction clipped to
    [.01, .99]); its weight is its marginal likelihood like every other member's."""

    def __init__(self, emit, cues, us, s0=1., noise=.25, **kw):
        super().__init__(emit, cues, us, **kw)
        N, K, d = self.N, self.K, len(self.cues) + 1
        self.noise = noise
        self.mu = np.zeros((N, K, d))
        self.S = np.broadcast_to(np.eye(d) * s0, (N, K, d, d)).copy()
        self.ev = np.zeros((N, K, len(self.sets) + 1))
        self._memo = None
        self._ver, self._wc, self._pc = 0, None, None    # state version; caches of weights, prior
        self.qp = np.full_like(self.q1, .5)               # (q1 + a) / (q1 + q0 + 2a), kept current

    def _vec(self, x):
        return np.append((np.asarray(x)[self.cues] > .5).astype(float), 1.)

    def _all_pred(self, x):
        """Member predictions and their mixture for every particle and every cause slot that
        can have prior mass (slots beyond max(used) + 1 have none and get 0 — exact)."""
        key = np.asarray(x, float).tobytes()
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        kk = min(self.K, int(self.used.max()) + 1)
        cells = self._cells(x)
        v = self._vec(x)
        lin = np.clip(self.mu[:, :kk] @ v, .01, .99)[:, :, None]
        pk = np.concatenate([self.qp[:, :kk, cells], lin], axis=2)
        if self._wc is not None and self._wc[0] == (self._ver, kk):
            w = self._wc[1]
        else:                         # the weights depend on the evidence only, not on x
            ev = self.ev[:, :kk]
            w = np.exp(ev - ev.max(2, keepdims=True))
            w /= w.sum(2, keepdims=True)
            self._wc = ((self._ver, kk), w)
        p = np.zeros((self.N, self.K, pk.shape[2]))
        p[:, :kk] = pk
        pu = np.zeros((self.N, self.K))
        pu[:, :kk] = (w * pk).sum(2)
        out = cells, v, p, pu
        self._memo = (key, out)
        return out

    def _prior_c(self):
        if self._pc is None or self._pc[0] != self._ver:
            self._pc = (self._ver, self._prior())
        return self._pc[1]

    def predict_us(self, x):
        _, _, _, pu = self._all_pred(x)
        joint = self._prior_c() * self._lik_x(x)
        w = np.exp(self.logw - self.logw.max())[:, None]
        return float((w * joint * pu).sum() / (w * joint).sum())

    def observe(self, x, us_next):
        N = self.N
        cells, v, p, pu = self._all_pred(x)
        self._memo = None
        joint = self._prior_c() * self._lik_x(x) * (pu if us_next else 1. - pu)
        self._ver += 1
        tot = joint.sum(1)
        self.logw += np.log(tot)
        z = self._advance(joint / tot[:, None])
        ar = np.arange(N)
        if self.emit:
            xe = np.asarray(x)[self.emit] > .5
            self.t1[ar, z] += xe
            self.t0[ar, z] += ~xe
        pz = p[ar, z]
        self.ev[ar, z] += np.log(pz if us_next else 1. - pz)
        ix = (ar[:, None], z[:, None], cells[None, :])
        if us_next:
            self.q1[ix] += 1.
        else:
            self.q0[ix] += 1.
        n1 = self.q1[ix]
        self.qp[ix] = (n1 + self.a) / (n1 + self.q0[ix] + 2 * self.a)
        y = 1. if us_next else 0.
        Sz = self.S[ar, z]
        var = np.einsum("nij,i,j->n", Sz, v, v) + self.noise
        g = Sz @ v / var[:, None]
        self.mu[ar, z] += g * (y - self.mu[ar, z] @ v)[:, None]
        self.S[ar, z] = Sz - g[:, :, None] * g[:, None, :] * var[:, None, None]
        self._resample()

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z", "q1", "q0", "qp", "ev", "mu", "S"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)


class LatentCauseSubAddM(LatentCauseSubAdd):
    """F-11 (AR-3, development): each cause also models every emitted channel as a Markov
    chain P(x_t | x_{t-1}, cause), with the prior mean of a cause's transition counts pooled
    over the particle's other causes (a new cause is expected to resemble the known ones).
    Phasic onsets are then about equally likely under known and new causes, while a never-seen
    tonic state is unlikely under every known cause (slice3-plan §9, candidate 1)."""

    def __init__(self, emit, cues, us, **kw):
        super().__init__(emit, cues, us, **kw)
        J = len(self.emit)
        self.tm1 = np.zeros((self.N, self.K, J, 2))
        self.tm0 = np.zeros((self.N, self.K, J, 2))
        self.prev = None
        self._x = None

    def _lik_x(self, x):
        if self.prev is None or not self.emit:
            return np.ones((self.N, self.K))
        J = np.arange(len(self.emit))
        pv = (np.asarray(self.prev)[self.emit] > .5).astype(int)
        xe = np.asarray(x)[self.emit] > .5
        n1, n0 = self.tm1[:, :, J, pv], self.tm0[:, :, J, pv]          # (N, K, J)
        k1 = n1.sum(1, keepdims=True) - n1
        k0 = n0.sum(1, keepdims=True) - n0
        mu = (k1 + self.a) / (k1 + k0 + 2 * self.a)
        p_on = (n1 + 2 * self.a * mu) / (n1 + n0 + 2 * self.a)
        return np.prod(np.where(xe[None, None, :], p_on, 1. - p_on), axis=2)

    def _advance(self, probs):
        z = super()._advance(probs)
        if self._x is not None and self.prev is not None and self.emit:
            ar, J = np.arange(self.N), np.arange(len(self.emit))
            pv = (np.asarray(self.prev)[self.emit] > .5).astype(int)
            xe = np.asarray(self._x)[self.emit] > .5
            self.tm1[ar[:, None], z[:, None], J[None, :], pv[None, :]] += xe
            self.tm0[ar[:, None], z[:, None], J[None, :], pv[None, :]] += ~xe
        return z

    def observe(self, x, us_next):
        self._x = x
        super().observe(x, us_next)
        self.prev, self._x = x, None
        self._pc = None

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z", "q1", "q0", "qp", "ev", "mu",
                     "S", "tm1", "tm0"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)


class LatentCauseSubAddP(LatentCauseSubAdd):
    """F-12 (AR-3, development): factored latent state per particle.

    place y_t: sticky CRP (hazard learned on the grid, like the regime's), each place models every
        emitted channel as a Markov chain P(x_t | x_{t-1}, place) with the prior mean pooled over
        the particle's other places (F-11's context model, moved to its own factor);
    regime z_t: the parent's sticky-CRP cause with its US model and no emission.
    Coupling (the only one): on the first visit of a place the regime is redrawn from the CRP by
    popularity (no stickiness); in a known place it persists as usual. So a novel context
    suggests that other contingencies may hold (AAB), while a familiar place change (home ->
    box) does not reset the regime (F-11 diagnosis, 12:15). SMC: y is proposed from the place
    posterior given x, the weight gets the place marginal, then the parent handles z.
    """

    def __init__(self, emit, cues, us, **kw):
        super().__init__([], cues, us, **kw)
        from models.latent_cause import H_GRID
        self.pemit = list(emit)
        N, KP, J = self.N, self.K, len(self.pemit)
        self.hp = np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=N)]
        self.mp = np.zeros((N, KP))
        self.mp[:, 0] = 1.
        self.usedp = np.ones(N, int)
        self.y = np.zeros(N, int)
        self.tm1 = np.zeros((N, KP, J, 2))
        self.tm0 = np.zeros((N, KP, J, 2))
        self.prev = None
        self.fresh = np.zeros(N, bool)      # this step entered a never-visited place
        self._q = None                      # (key, q, Lsum) for predict_us

    # ---------------------------------------------------------------- place factor
    def _place_post(self, x):
        N, KP = self.N, self.K
        live = np.arange(KP)[None, :] < self.usedp[:, None]
        if self.prev is None:
            pr = np.zeros((N, KP))
            pr[:, 0] = 1.
            lik = np.ones((N, KP))
        else:
            denom = (self.mp * live).sum(1) + self.alpha
            pr = (self.hp / denom)[:, None] * self.mp * live
            rows = np.nonzero(self.usedp < KP)[0]
            pr[rows, self.usedp[rows]] += self.hp[rows] * self.alpha / denom[rows]
            pr[np.arange(N), self.y] += 1. - self.hp
            pr /= pr.sum(1, keepdims=True)
            J = np.arange(len(self.pemit))
            pv = (np.asarray(self.prev)[self.pemit] > .5).astype(int)
            xe = np.asarray(x)[self.pemit] > .5
            n1, n0 = self.tm1[:, :, J, pv], self.tm0[:, :, J, pv]
            # population mean over all of the particle's places (empirical Bayes, own counts
            # included): with leave-one-out a single known place gets the flat .5 and explains
            # an unseen transition better than a place nothing is known about
            k1 = n1.sum(1, keepdims=True)
            k0 = n0.sum(1, keepdims=True)
            mu = (k1 + self.a) / (k1 + k0 + 2 * self.a)
            p_on = (n1 + 2 * self.a * mu) / (n1 + n0 + 2 * self.a)
            lik = np.prod(np.where(xe[None, None, :], p_on, 1. - p_on), axis=2)
        j = pr * lik
        L = j.sum(1)
        return j / L[:, None], L

    def _fresh_prior(self):
        """CRP by popularity over the regimes, no stickiness."""
        N, K = self.N, self.K
        live = np.arange(K)[None, :] < self.used[:, None]
        pr = self.m * live
        rows = np.nonzero(self.used < K)[0]
        pr = pr.copy()
        pr[rows, self.used[rows]] += self.alpha
        return pr / pr.sum(1, keepdims=True)

    def _prior_c(self):
        sticky = self._prior()
        if not self.started:
            return sticky
        if self._mode == "observe":
            f = self.fresh[:, None]
        else:                                   # predict: expectation over the place posterior
            q = self._q[1]
            rows = self.usedp < self.K
            f = np.where(rows, q[np.arange(self.N), np.minimum(self.usedp, self.K - 1)], 0.)[:, None]
        return (1. - f) * sticky + f * self._fresh_prior()

    def _lik_x(self, x):
        if self._mode == "observe":
            return np.ones((self.N, self.K))
        return np.broadcast_to(self._q[2][:, None], (self.N, self.K))

    # ------------------------------------------------------------------ interface
    def predict_us(self, x):
        key = (np.asarray(x, float).tobytes(), self._ver)
        if self._q is None or self._q[0] != key:
            q, L = self._place_post(x)
            self._q = (key, q, L)
        self._mode = "predict"
        return super().predict_us(x)

    def observe(self, x, us_next):
        q, L = self._place_post(x)
        N = self.N
        cdf = np.cumsum(q, axis=1)
        y = np.minimum((cdf < self.rng.random(N)[:, None] * cdf[:, -1:]).sum(1), self.K - 1)
        self.fresh = y >= self.usedp
        if self.prev is not None:
            moved = y != self.y
            self.mp[np.arange(N)[moved], y[moved]] += 1.
            ar, J = np.arange(N), np.arange(len(self.pemit))
            pv = (np.asarray(self.prev)[self.pemit] > .5).astype(int)
            xe = np.asarray(x)[self.pemit] > .5
            self.tm1[ar[:, None], y[:, None], J[None, :], pv[None, :]] += xe
            self.tm0[ar[:, None], y[:, None], J[None, :], pv[None, :]] += ~xe
        self.usedp = np.maximum(self.usedp, y + 1)
        self.y = y
        self.logw += np.log(L)
        self._mode = "observe"
        super().observe(x, us_next)
        self.prev = x
        self.fresh[:] = False
        self._q = None

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z", "q1", "q0", "qp", "ev", "mu",
                     "S", "hp", "mp", "usedp", "y", "tm1", "tm0", "fresh"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)
