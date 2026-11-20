"""Ideal observer for the T2 streams: latent contexts with a sticky Chinese-restaurant prior.

Generative model (the one slice 2 claims; decisions W-0):

    z_t      = z_{t-1}                         with prob 1 - h
             ~ existing k  prop. to m_k        with prob h  (m_k = times context k was entered)
               or a new context prop. to alpha
    x_t,j    ~ Bernoulli(theta_{z_t, j})        every emitted (non-US) channel j
    us_{t+1} ~ Bernoulli(phi_{z_t, c(x_t)})    c = configuration of the cue channels at t
    theta, phi ~ Beta(a, a) independently per context

Inference: Rao-Blackwellised particle filter (counts are sufficient statistics, the context
sequence is sampled with the optimal proposal, systematic resampling at ESS < N/2). The hazard
is either fixed or a static parameter drawn per particle from a grid (posterior over the grid
by the particle weights). `exact_predictive` enumerates all context sequences for tiny
streams; tests/test_latent_cause.py checks the filter against it.
"""
from __future__ import annotations

import numpy as np

H_GRID = (.0003, .001, .003, .01, .03)


class LatentCauseSMC:
    def __init__(self, emit, cues, us, hazard=None, alpha=1., a=.5, n=2000, kmax=10, seed=0,
                 away=None, hier=False):
        self.emit, self.cues, self.us = list(emit), list(cues), us
        self.away = away              # channel: steps with it on are not emitted by the context
        self.hier = hier              # Beta prior mean = pooled frequency over the particle's
                                      # other contexts (concentration 2a): a new context is
                                      # expected to resemble the known ones, not to be arbitrary
        self.alpha, self.a, self.N, self.K = alpha, a, n, kmax
        self.rng = np.random.default_rng(seed)
        J, C = len(self.emit), 2 ** len(self.cues)
        self.h = (np.full(n, float(hazard)) if hazard is not None
                  else np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=n)])
        self.t1 = np.zeros((n, kmax, J))
        self.t0 = np.zeros((n, kmax, J))
        self.p1 = np.zeros((n, kmax, C))
        self.p0 = np.zeros((n, kmax, C))
        self.m = np.zeros((n, kmax))
        self.m[:, 0] = 1.
        self.used = np.ones(n, int)
        self.z = np.zeros(n, int)
        self.logw = np.zeros(n)
        self.x = None
        self.started = False          # the first observation belongs to context 0

    # ------------------------------------------------------------------ pieces
    def _cfg(self, x):
        return int(sum(1 << i for i, c in enumerate(self.cues) if x[c] > .5))

    def _prior(self):
        """P(z_t = k | particle) for every slot k, shape (N, K)."""
        N, K = self.N, self.K
        if not self.started:
            pr = np.zeros((N, K))
            pr[:, 0] = 1.
            return pr
        live = np.arange(K)[None, :] < self.used[:, None]
        denom = (self.m * live).sum(1) + self.alpha
        pr = (self.h / denom)[:, None] * self.m * live
        new = self.used < K
        rows = np.nonzero(new)[0]
        pr[rows, self.used[rows]] += self.h[rows] * self.alpha / denom[rows]
        pr[np.arange(N), self.z] += 1. - self.h
        return pr / pr.sum(1, keepdims=True)

    def _mean(self, n1, n0, part="emit"):
        """Prior mean of every context: pooled frequency over the particle's OTHER contexts
        (leave-one-out, so a context's own data are not counted twice)."""
        if not self.hier or (self.hier == "emit" and part == "us"):
            return .5
        k1 = n1.sum(1, keepdims=True) - n1
        k0 = n0.sum(1, keepdims=True) - n0
        return (k1 + self.a) / (k1 + k0 + 2 * self.a)

    def _lik_x(self, x):
        xe = (np.asarray(x)[self.emit] > .5)[None, None, :]
        mu = self._mean(self.t1, self.t0)
        p_on = (self.t1 + 2 * self.a * mu) / (self.t1 + self.t0 + 2 * self.a)
        return np.prod(np.where(xe, p_on, 1. - p_on), axis=2)

    def _p_us(self, c):
        n1, n0 = self.p1[:, :, c], self.p0[:, :, c]
        mu = self._mean(n1, n0, "us")
        return (n1 + 2 * self.a * mu) / (n1 + n0 + 2 * self.a)

    # --------------------------------------------------------------- interface
    def predict_us(self, x):
        """P(US on the next step | history, x now), without learning."""
        joint = self._prior() * self._lik_x(x)          # x now is evidence too
        w = np.exp(self.logw - self.logw.max())[:, None]
        return float((w * joint * self._p_us(self._cfg(x))).sum() / (w * joint).sum())

    def observe(self, x, us_next):
        """Condition on x_t and us_{t+1} and advance the particles."""
        N = self.N
        c = self._cfg(x)
        if self.away is not None and x[self.away] > .5:
            self._advance(self._prior())
            return
        pu = self._p_us(c)
        joint = self._prior() * self._lik_x(x) * (pu if us_next else 1. - pu)
        tot = joint.sum(1)
        self.logw += np.log(tot)
        z = self._advance(joint / tot[:, None])
        xe = np.asarray(x)[self.emit] > .5
        ar = np.arange(N)
        self.t1[ar, z] += xe
        self.t0[ar, z] += ~xe
        if us_next:
            self.p1[ar, z, c] += 1.
        else:
            self.p0[ar, z, c] += 1.
        self._resample()

    def _advance(self, probs):
        """Sample z_t from per-particle probabilities and update the entry counts."""
        N = self.N
        cdf = np.cumsum(probs, axis=1)
        z = np.minimum((cdf < self.rng.random(N)[:, None] * cdf[:, -1:]).sum(1), self.K - 1)
        moved = z != self.z
        self.m[np.arange(N)[moved], z[moved]] += 1.
        grown = z >= self.used
        self.used[grown] = z[grown] + 1
        self.z = z
        self.started = True
        return z

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for name in ("h", "t1", "t0", "p1", "p0", "m", "used", "z"):
            setattr(self, name, getattr(self, name)[idx].copy())
        self.logw = np.zeros(self.N)

    def hazard_posterior(self):
        w = np.exp(self.logw - self.logw.max())
        return {h: float(w[self.h == h].sum() / w.sum()) for h in np.unique(self.h)}

    # agent-style interface for eval.phenomena
    def step(self, x):
        self.x = x
        return 0

    def learn(self, y):
        self.observe(self.x, y[self.us] > .5)


def exact_predictive(stream, emit, cues, us, hazard, alpha=1., a=.5):
    """Exact P(us_{T+1} = 1 | x_1..T, us_2..T) for a tiny stream [(x_t, us_{t+1}), ...] whose
    last us is unknown (ignored), by enumerating all context sequences (canonical labels)."""
    T = len(stream)

    def logp_seq(zs):
        lp, m, used = 0., [1.], 1
        t1, t0, p1, p0 = {}, {}, {}, {}
        for t, (x, u) in enumerate(stream):
            k = zs[t]
            if t > 0:
                prev = zs[t - 1]
                denom = sum(m) + alpha
                pk = hazard * (m[k] / denom if k < used else alpha / denom)
                if k == prev:
                    pk += 1. - hazard
                lp += np.log(pk)
                if k != prev:
                    if k < used:
                        m[k] += 1.
                    else:
                        m.append(1.)
                        used += 1
            cfg = int(sum(1 << i for i, c in enumerate(cues) if x[c] > .5))
            for j in emit:
                on = x[j] > .5
                n1, n0 = t1.get((k, j), 0.), t0.get((k, j), 0.)
                q = (n1 + a) / (n1 + n0 + 2 * a)
                lp += np.log(q if on else 1. - q)
                if on:
                    t1[(k, j)] = n1 + 1.
                else:
                    t0[(k, j)] = n0 + 1.
            n1, n0 = p1.get((k, cfg), 0.), p0.get((k, cfg), 0.)
            q = (n1 + a) / (n1 + n0 + 2 * a)
            if t == T - 1:
                return lp, q
            lp += np.log(q if u else 1. - q)
            if u:
                p1[(k, cfg)] = n1 + 1.
            else:
                p0[(k, cfg)] = n0 + 1.

    def seqs():
        def rec(prefix, used):
            if len(prefix) == T:
                yield prefix
                return
            for k in range(used + 1):
                yield from rec(prefix + [k], max(used, k + 1))
        yield from rec([0], 1)

    num = den = 0.
    for zs in seqs():
        lp, q = logp_seq(zs)
        w = np.exp(lp)
        num += w * q
        den += w
    return num / den

