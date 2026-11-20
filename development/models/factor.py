"""3b-1 predictor (AR-4, research/factor-design.md): one generative model, one shared filter.

Per particle (SMC, N particles shared by all channels), for the transition x_t -> x_{t+1}:
  place  y   sticky CRP (hazard on a grid);
  regime z   shared by all channels; hierarchical CRP indexed by the place (F-13):
             same place -> previous regime with prob 1 - h_z, otherwise a draw;
             place entered (or sticky switch) -> draw  P(z | y) ∝ n[y, z] + beta * G(z),
             G ∝ global entry counts (+ alpha for a new regime); a never-visited place has n = 0;
  channel j  P(x_j,t+1 = 1) = sum_M w_jM(z) p_jM, members M:
             - parent sets S (|S| <= 2) over j's candidates (own lags, action one-hot, k slots =
               (channel, lag) pairs chosen by pairwise sequential evidence), with a two-level
               hierarchical Beta prior: the regime's cell backs off to the SAME cell pooled over
               all data (a new regime is expected to resemble the known ones: AAB), and a pooled
               cell never seen backs off to its sub-configurations (an unseen configuration is
               predicted by its elements, D-32: no flat .5);
             - additive member (Bayesian linear regression over the candidates, clipped);
             - place member: the place's Markov chain P(x_j,t+1 | x_j,t, y), prior mean pooled over
               the particle's places (F-12's context model, now a member: every observation is
               generated once, design fix 13:45).
             Weights w = marginal likelihoods of the members on the regime's own data.
Inference: per particle the joint (y, z) of each transition is proposed from its exact posterior
given x_{t+1} (KP x K cells); the particle weight multiplies the predictive probabilities of
all channels. Pooled tables are identical across particles (all see the same data) and stored
once.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from models.latent_cause import H_GRID

C_REG = 1.           # concentration: regime cell -> pooled cell
C_GLOB = 1.          # concentration: pooled cell -> sub-configurations
SWAP = np.log(3.)    # slot swap margin (v18)
QBINS = np.array([.05, .2, .5, .8, .95])   # bins of the model prediction (conditional screening)
F32 = np.float32


class FactorModel:
    def __init__(self, n, A, v, seed=0, N=200, K=10, k=4, lags=(0,), own_lags=(0,),
                 beta=1., alpha=1., noise=.25, s0=1., process=False, place="member", one_lag=False, cond=False, ctx_only=False):
        self.n, self.A, self.v = n, A, v
        self.ctx_only = ctx_only     # §9.2: the place (view) emits only persistent channels
        self.cond = cond             # r5 deep form: screen against the model's own prediction
        self.one_lag = one_lag       # r5: one slot per source channel (a cue with one learned delay)
        self.view = place == "view"  # r4: context inferred by a separate pathway (two views)
        self.proc = None
        if process:                  # F3: hidden processes (Decomposer groups, windows) as a member
            from models.decompose import Decomposer
            self.proc = Decomposer(n, A, v, seed, windows=True)
        self._pp = None              # process prediction for the last learned transition
        self.N, self.K, self.KP, self.k = N, K, K, k
        self.rng = np.random.default_rng(seed)
        self.ch = [j for j in range(n) if j != v]
        J = self.J = len(self.ch)
        self.lags, self.own_lags = tuple(lags), tuple(own_lags)
        self.D = max(max(self.lags), max(self.own_lags))
        self.m = len(self.own_lags) + A + k                      # candidate positions
        self.sets = [s for r in range(3) for s in combinations(range(self.m), r)]
        self.S = len(self.sets)                                  # empty, m singletons, pairs
        self.off = np.cumsum([0] + [2 ** len(s) for s in self.sets])
        single = {s[0]: i for i, s in enumerate(self.sets) if len(s) == 1}
        pairs = [s for s in self.sets if len(s) == 2]
        self._sa = np.array([single[s[0]] for s in pairs], int)
        self._sb = np.array([single[s[1]] for s in pairs], int)
        self.M = self.S + 1 + (process is not False) + (not self.view)   # + add [+ proc] [+ place]
        C = int(self.off[-1])
        # regime tables, layout (regime, channel, cell / member, particle)
        self.q1 = np.zeros((K, J, C, N), F32)
        self.q0 = np.zeros((K, J, C, N), F32)
        self.ev = np.zeros((K, J, self.M, N), F32)
        # pooled tables (particle-independent)
        self.g1 = np.zeros((J, C))
        self.g0 = np.zeros((J, C))
        self.evg = np.zeros((J, self.S))
        d = self.m + 1
        self.noise, self.s0 = noise, s0
        self.mu = np.zeros((N, K, J, d))
        self.Sg = np.broadcast_to(np.eye(d) * s0, (N, K, J, d, d)).copy()
        # regime factor
        self.hz = np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=N)]
        self.mz = np.zeros((N, K))
        self.npz = np.zeros((N, self.KP, K))
        self.used = np.ones(N, int)
        self.z = np.zeros(N, int)
        self.beta, self.alpha = beta, alpha
        # place factor
        self.hp = np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=N)]
        self.mp = np.zeros((N, self.KP))
        self.usedp = np.ones(N, int)
        self.y = np.zeros(N, int)
        self.tm1 = np.zeros((N, self.KP, J, 2))
        self.tm0 = np.zeros((N, self.KP, J, 2))
        self.mp[:, 0] = 1.
        self.mz[:, 0] = 1.
        self.npz[:, 0, 0] = 1.
        self.logw = np.zeros(N)
        self.started = False
        # slots and screening (deterministic: outside the particles)
        self.slots = [[None] * k for _ in range(J)]              # (channel position, lag)
        L = len(self.lags)
        self.f1 = np.zeros((J, L, J, 2, 2, 2))   # i, lag, j, x_j(t), x_i(t-lag), x_j(t+1)
        self.f0 = np.zeros((J, 2, 2))
        self.sev = np.zeros((J, L, J))
        self._q = None                           # model prediction of the last transition (J,)
        if cond:
            self.fq1 = np.zeros((J, L, J, len(QBINS) + 1, 2, 2))   # i, lag, j, bin(q_j), x_i, x_j'
            self.fq0 = np.zeros((J, len(QBINS) + 1, 2))
            self.cev = np.zeros((J, L, J))
        self.hist = []                           # past observations (bits), newest last
        self.t = 0
        self._memo = None
        self.births = 0

    # ------------------------------------------------------------------ keys
    def _bits(self, x):
        return (np.asarray(x, float)[self.ch] > .5).astype(float)

    def _lag(self, b, d, hist):
        if d == 0:
            return b
        if len(hist) >= d:
            return hist[-d]
        return np.zeros(self.J)

    def _keys(self, b, a, hist):
        U = np.zeros((self.J, self.m))
        for p, d in enumerate(self.own_lags):
            U[:, p] = self._lag(b, d, hist)
        U[:, len(self.own_lags) + a] = 1.
        base = len(self.own_lags) + self.A
        for j in range(self.J):
            for s, sl in enumerate(self.slots[j]):
                if sl is not None:
                    U[j, base + s] = self._lag(b, sl[1], hist)[sl[0]]
        return U

    def _cells(self, U):
        cells = np.empty((self.J, self.S), int)
        for i, s in enumerate(self.sets):
            c = np.full(self.J, self.off[i])
            for bit, p in enumerate(s):
                c += (U[:, p] > .5).astype(int) << bit
            cells[:, i] = c
        return cells

    # ---------------------------------------------------------------- members
    def _pooled(self, cells):
        """Pooled cell predictions (J, S) with back-off to sub-configurations."""
        Jr = np.arange(self.J)[:, None]
        n1, n0 = self.g1[Jr, cells], self.g0[Jr, cells]
        tot = n1 + n0 + C_GLOB
        p = np.empty_like(n1)
        p[:, 0] = (n1[:, 0] + C_GLOB * .5) / tot[:, 0]
        i1 = slice(1, 1 + self.m)
        p[:, i1] = (n1[:, i1] + C_GLOB * p[:, :1]) / tot[:, i1]
        ea, eb = self.evg[:, self._sa], self.evg[:, self._sb]
        mx = np.maximum(ea, eb)
        wa, wb = np.exp(ea - mx), np.exp(eb - mx)
        prior = (wa * p[:, self._sa] + wb * p[:, self._sb]) / (wa + wb)
        i2 = slice(1 + self.m, self.S)
        p[:, i2] = (n1[:, i2] + C_GLOB * prior) / tot[:, i2]
        return p

    def _members(self, b, a, hist, kk, pp=None):
        """Regime-indexed member predictions P (kk, J, S+1, N) [sets + additive], their
        pooled-weight-free mixture parts and the place member's weight; plus keys."""
        U = self._keys(b, a, hist)
        cells = self._cells(U)
        pg = self._pooled(cells).astype(F32)                       # (J, S)
        Jr = np.arange(self.J)[:, None]
        n1 = self.q1[:kk, Jr, cells]                               # (kk, J, S, N)
        n0 = self.q0[:kk, Jr, cells]
        p = (n1 + C_REG * pg[None, :, :, None]) / (n1 + n0 + C_REG)
        v = np.concatenate([U, np.ones((self.J, 1))], 1)           # (J, d)
        lin = np.clip(np.einsum("nkjd,jd->kjn", self.mu[:, :kk], v), .01, .99).astype(F32)
        parts = [p, lin[:, :, None]]
        if self.proc is not None:
            pp = np.full(self.J, .5) if pp is None else pp
            parts.append(np.broadcast_to(np.clip(pp, .001, .999).astype(F32)[None, :, None, None],
                                         (kk, self.J, 1, self.N)))
        P = np.concatenate(parts, 2)                               # (kk, J, M-1, N)
        ev = self.ev[:kk]
        E = np.exp(ev - ev.max(2, keepdims=True))                  # (kk, J, M, N)
        Z = E.sum(2)
        if self.view:
            rest = (E * P).sum(2) / Z
            wpl = np.zeros_like(rest)
        else:
            rest = (E[:, :, :-1] * P).sum(2) / Z                   # (kk, J, N)
            wpl = E[:, :, -1] / Z                                  # (kk, J, N)
        return dict(U=U, cells=cells, pg=pg, P=P, rest=rest, wpl=wpl, v=v)

    def _place_pred(self, b):
        """Place member: P(x_j,t+1 = 1 | x_j,t, y) for every place, (N, KP, J)."""
        Jr = np.arange(self.J)
        pv = (b > .5).astype(int)
        n1, n0 = self.tm1[:, :, Jr, pv], self.tm0[:, :, Jr, pv]
        k1, k0 = n1.sum(1, keepdims=True), n0.sum(1, keepdims=True)
        mu = (k1 + .5) / (k1 + k0 + 1.)
        return (n1 + mu) / (n1 + n0 + 1.)

    # ----------------------------------------------------------------- priors
    def _place_prior(self):
        N, KP = self.N, self.KP
        if not self.started:
            pr = np.zeros((N, KP))
            pr[:, 0] = 1.
            return pr
        live = np.arange(KP)[None, :] < self.usedp[:, None]
        denom = (self.mp * live).sum(1) + self.alpha
        pr = (self.hp / denom)[:, None] * self.mp * live
        rows = np.nonzero(self.usedp < KP)[0]
        pr[rows, self.usedp[rows]] += self.hp[rows] * self.alpha / denom[rows]
        pr[np.arange(N), self.y] += 1. - self.hp
        return pr / pr.sum(1, keepdims=True)

    def _regime_prior(self, kk):
        """P(z | y, previous y, previous z) for every place y: (N, KP, kk)."""
        N, K = self.N, self.K
        ar = np.arange(N)
        if not self.started:
            pr = np.zeros((N, self.KP, kk))
            pr[:, :, 0] = 1.
            return pr
        live = np.arange(K)[None, :] < self.used[:, None]
        G = self.mz * live
        rows = np.nonzero(self.used < K)[0]
        G[rows, self.used[rows]] += self.alpha
        G /= G.sum(1, keepdims=True)
        Dr = self.npz + self.beta * G[:, None, :]
        Dr = (Dr / Dr.sum(2, keepdims=True))[:, :, :kk]
        h = self.hz
        cur = Dr[ar, self.y].copy()                                # draw in the same place
        Dr[ar, self.y] = h[:, None] * cur
        Dr[ar, self.y, self.z] += 1. - h
        return Dr

    def _joint(self, mem, pl, kk):
        """Per-channel mixture for every (particle, place, regime): (N, KP, kk, J)."""
        rest = mem["rest"].transpose(2, 0, 1)                      # (N, kk, J)
        wpl = mem["wpl"].transpose(2, 0, 1)
        return rest[:, None] + wpl[:, None] * pl[:, :, None, :]

    # ------------------------------------------------------------ view mode
    def _place_view(self, b):
        """Place posterior given x_t (emission of x_t by the place's chains): (N, KP), L."""
        pr = self._place_prior()
        if not self.hist:
            return pr, np.ones(self.N)
        pl = self._place_pred(self.hist[-1])                       # P(x_t | x_{t-1}, y)
        e = np.where((b > .5)[None, None, :], pl, 1. - pl)
        if self.ctx_only:              # persistent channels: P(on -> on) > .7 with >= 5 onsets
            on = self.f0[:, 1].sum(1)
            pers = (on >= 5) & ((self.f0[:, 1, 1] + .5) / (on + 1.) > .7)
            e = np.where(pers[None, None, :], e, 1.)
        lik = np.prod(e, axis=2)
        j = pr * lik
        L = j.sum(1)
        return j / L[:, None], L

    def _regime_prior_q(self, kk, q, zd=None):
        """P(z_t) given a place distribution q (N, KP) and a previous-regime distribution zd."""
        N, K = self.N, self.K
        ar = np.arange(N)
        if not self.started:
            pr = np.zeros((N, kk))
            pr[:, 0] = 1.
            return pr
        if zd is None:
            zd = np.zeros((N, kk))
            zd[ar, self.z] = 1.
        live = np.arange(K)[None, :] < self.used[:, None]
        G = self.mz * live
        rows = np.nonzero(self.used < K)[0]
        G[rows, self.used[rows]] += self.alpha
        G /= G.sum(1, keepdims=True)
        Dr = self.npz + self.beta * G[:, None, :]
        Dr = (Dr / Dr.sum(2, keepdims=True))[:, :, :kk]
        pr = np.einsum("np,npk->nk", q, Dr)
        stay = q[ar, self.y][:, None]
        h = self.hz[:, None]
        pr += stay * ((1. - h) * zd - (1. - h) * Dr[ar, self.y])
        return pr

    def _predict_view(self, x, a, hypothetical):
        b = self._bits(x)
        kk = min(self.K, int(self.used.max()) + 1)
        w = np.exp(self.logw - self.logw.max())
        pp = None
        if self.proc is not None:
            pp = (self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a))[self.ch]
        zd = None
        if hypothetical and self.hist:                 # the transition prev -> x as evidence
            prev = self.hist[-1]
            mem0 = self._members(prev, 0, self.hist[:-1], kk, self._pp)
            p0 = mem0["rest"].transpose(2, 0, 1)                   # (N, kk, J)
            lik0 = np.prod(np.where((b > .5)[None, None, :], p0, 1. - p0), axis=2)
            qs = np.zeros((self.N, self.KP))
            qs[np.arange(self.N), self.y] = 1.
            zd = self._regime_prior_q(kk, qs) * lik0
            Lz = zd.sum(1)
            w = w * Lz
            zd = zd / Lz[:, None]
        q, Lp = self._place_view(b)
        pz = self._regime_prior_q(kk, q, zd)                        # (N, kk)
        mem = self._members(b, a, self.hist, kk, pp)
        pu = mem["rest"].transpose(2, 0, 1)                        # (N, kk, J)
        jw = (w * Lp)[:, None] * pz
        out = np.full(self.n, .5)
        out[self.ch] = np.einsum("nk,nkj->j", jw, pu) / jw.sum()
        return out, mem

    def _learn_view(self, x, a, y, mem):
        N, ar = self.N, np.arange(self.N)
        b, bn = self._bits(x), self._bits(y)
        yb = bn > .5
        kk = mem["rest"].shape[0]
        q, Lp = self._place_view(b)
        cdf = np.cumsum(q, 1)
        yt = np.minimum((cdf < self.rng.random(N)[:, None] * cdf[:, -1:]).sum(1), self.KP - 1)
        if self.hist:                                  # place chains learn x_{t-1} -> x_t
            Jr = np.arange(self.J)
            pv = (self.hist[-1] > .5).astype(int)
            xe = b > .5
            self.tm1[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += xe
            self.tm0[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += ~xe
        qs = np.zeros_like(q)
        qs[ar, yt] = 1.
        pz = self._regime_prior_q(kk, qs)
        pu = mem["rest"].transpose(2, 0, 1)
        if self.cond:                  # the model's prediction of this transition (screening)
            ww = np.exp(self.logw - self.logw.max())[:, None] * pz
            self._q = np.einsum("nk,nkj->j", ww, pu) / ww.sum()
        lik = np.prod(np.where(yb[None, None, :], pu, 1. - pu), axis=2)   # (N, kk)
        joint = pz * lik
        tot = joint.sum(1)
        self.logw += np.log(Lp) + np.log(tot)
        cdf = np.cumsum(joint / tot[:, None], 1)
        zt = np.minimum((cdf < self.rng.random(N)[:, None]).sum(1), kk - 1)
        return yt, zt

    # --------------------------------------------------------------- introspection
    def trace(self):
        """Deterministic summary of the posterior state (function of code and seed only)."""
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        S = self.S
        E = np.exp(self.ev[0] - self.ev[0].max(1, keepdims=True))
        E = (E / E.sum(1, keepdims=True) * w[None, None, :]).sum(2)          # (J, M)
        r = lambda v: [round(float(u), 4) for u in v]
        return dict(t=self.t, births=int(self.births),
                    place=r(np.bincount(self.y, weights=w, minlength=self.KP)),
                    regime=r(np.bincount(self.z, weights=w, minlength=self.K)),
                    used=round(float((w * self.used).sum()), 3),
                    usedp=round(float((w * self.usedp).sum()), 3),
                    slots=[[list(s) if s is not None else None for s in sl] for sl in self.slots],
                    w_sets=r(E[:, :S].sum(1)), w_add=r(E[:, S]),
                    w_rest=r(E[:, S + 1:].sum(1)))

    # --------------------------------------------------------------- interface
    def predict(self, x, a, hypothetical=False):
        """Marginals of all n channels at t+1 given x_t = x; value channel .5.
        hypothetical: x has not been learned (a probe) — the transition from the last learned
        observation to x is first used as evidence about (place, regime)."""
        b = self._bits(x)
        key = (b.tobytes(), a, self.t, hypothetical)
        if self._memo is not None and key in self._memo:
            return self._memo[key][0]
        if self.view:
            out, mem = self._predict_view(x, a, hypothetical)
            if not hypothetical:
                self._memo = self._memo or {}
                self._memo[key] = (out, mem)
            return out
        kk = min(self.K, int(self.used.max()) + 1)
        w = np.exp(self.logw - self.logw.max())
        pp = None
        if self.proc is not None:
            pp = (self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a))[self.ch]
        if hypothetical and self.hist:
            prev = self.hist[-1]
            mem0 = self._members(prev, 0, self.hist[:-1], kk, self._pp)
            pj = self._joint(mem0, self._place_pred(prev), kk)
            lik = np.prod(np.where((b > .5)[None, None, None, :], pj, 1. - pj), axis=3)
            post = self._place_prior()[:, :, None] * self._regime_prior(kk) * lik
            wy = post                                              # (N, KP, kk) as next-step prior
            hist = self.hist
        else:
            wy = self._place_prior()[:, :, None] * self._regime_prior(kk)
            hist = self.hist
        mem = self._members(b, a, hist, kk, pp)
        pj = self._joint(mem, self._place_pred(b), kk)             # (N, KP, kk, J)
        jw = w[:, None, None] * wy
        out = np.full(self.n, .5)
        out[self.ch] = np.einsum("npk,npkj->j", jw, pj) / jw.sum()
        if not hypothetical:
            self._memo = self._memo or {}
            self._memo[key] = (out, mem)
        return out

    def learn(self, x, a, y):
        N, ar = self.N, np.arange(self.N)
        b, bn = self._bits(x), self._bits(y)
        yb = bn > .5
        kk = min(self.K, int(self.used.max()) + 1)
        key = (b.tobytes(), a, self.t, False)
        if self._memo is not None and key in self._memo \
                and self._memo[key][1]["rest"].shape[0] == kk:
            mem = self._memo[key][1]
        else:
            pp = self.proc.predict(x, a)[self.ch] if self.proc is not None else None
            mem = self._members(b, a, self.hist, kk, pp)
        if self.proc is not None:
            self._pp = mem["P"][0, :, -1, 0].astype(float)
            self.proc.predict(x, a)          # the chosen action's prior (a planner may have
            self.proc.learn(x, a, y)         # evaluated other actions last)
        pl = self._place_pred(b)                                   # (N, KP, J)
        if self.view:
            yt, zt = self._learn_view(x, a, y, mem)
        else:
            pj = self._joint(mem, pl, kk)
            lik = np.prod(np.where(yb[None, None, None, :], pj, 1. - pj), axis=3)   # (N, KP, kk)
            prior = self._place_prior()[:, :, None] * self._regime_prior(kk)
            if self.cond:
                ww = np.exp(self.logw - self.logw.max())[:, None, None] * prior
                self._q = np.einsum("npk,npkj->j", ww, pj) / ww.sum()
            joint = prior * lik
            tot = joint.sum((1, 2))
            self.logw += np.log(tot)
            flat = (joint / tot[:, None, None]).reshape(N, -1)
            cdf = np.cumsum(flat, 1)
            idx = np.minimum((cdf < self.rng.random(N)[:, None]).sum(1), flat.shape[1] - 1)
            yt, zt = np.divmod(idx, kk)
        # factor bookkeeping
        if self.started:
            moved = yt != self.y
            self.mp[ar[moved], yt[moved]] += 1.
            entered = moved | (zt != self.z)
        else:
            entered = np.zeros(N, bool)
        self.npz[ar[entered], yt[entered], zt[entered]] += 1.
        self.mz[ar[entered], zt[entered]] += 1.
        self.births += int((zt >= self.used).sum())
        self.usedp = np.maximum(self.usedp, yt + 1)
        self.used = np.maximum(self.used, zt + 1)
        self.y, self.z = yt, zt
        self.started = True
        # member evidence in the sampled regime (place member: the sampled place's chain)
        Pz = np.clip(mem["P"][zt, :, :, ar], 1e-6, 1. - 1e-6)    # (N, J, M-1)
        if self.view:
            Pm = Pz
        else:
            plz = np.clip(pl[ar, yt], 1e-6, 1. - 1e-6)[:, :, None]   # (N, J, 1)
            Pm = np.concatenate([Pz, plz.astype(F32)], 2)
        self.ev[zt, :, :, ar] += np.where(yb[None, :, None], np.log(Pm), np.log(1. - Pm))
        cells = mem["cells"]
        for j in range(self.J):
            tq = self.q1 if yb[j] else self.q0
            tq[zt[None, :], j, cells[j][:, None], ar[None, :]] += 1.
        Jr = np.arange(self.J)
        if not self.view:
            pv = (b > .5).astype(int)
            self.tm1[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += yb
            self.tm0[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += ~yb
        v = mem["v"]
        Sz = self.Sg[ar, zt]
        var = np.einsum("njab,ja,jb->nj", Sz, v, v) + self.noise
        g = np.einsum("njab,jb->nja", Sz, v) / var[..., None]
        err = yb[None, :].astype(float) - np.einsum("nja,ja->nj", self.mu[ar, zt], v)
        self.mu[ar, zt] += g * err[..., None]
        self.Sg[ar, zt] = Sz - g[..., :, None] * g[..., None, :] * var[..., None, None]
        # pooled level (same for every particle)
        pg = np.clip(self._pooled(cells), 1e-6, 1. - 1e-6)
        self.evg += np.where(yb[:, None], np.log(pg), np.log(1. - pg))
        tg = np.where(yb[:, None], 1., 0.)
        np.add.at(self.g1, (Jr[:, None], cells), tg)
        np.add.at(self.g0, (Jr[:, None], cells), 1. - tg)
        self._resample()
        # screening and slots (deterministic)
        self._screen(b, bn)
        self.hist = (self.hist + [b])[-max(1, self.D + 1):]
        self.t += 1
        self._reslot()
        self._memo = None

    # ---------------------------------------------------------------- slots
    def _screen(self, b, bn):
        J = np.arange(self.J)
        xo, xn = b.astype(int), bn.astype(int)
        c0 = self.f0[J, xo]
        p0 = (c0[J, xn] + .5) / (c0.sum(1) + 1.)
        I, Jm = np.meshgrid(J, J, indexing="ij")
        for li, d in enumerate(self.lags):
            if d > len(self.hist):
                continue
            xs = self._lag(b, d, self.hist).astype(int)
            c1 = self.f1[I, li, Jm, xo[Jm], xs[I]]
            p1 = (np.take_along_axis(c1, xn[Jm][..., None], -1)[..., 0] + .5) / (c1.sum(-1) + 1.)
            self.sev[:, li, :] += np.log(p1) - np.log(p0)[None, :]
            self.f1[I, li, Jm, xo[Jm], xs[I], xn[Jm]] += 1.
            if self.cond and self._q is not None:
                qb = np.searchsorted(QBINS, self._q)                # (J,)
                d1 = self.fq1[I, li, Jm, qb[Jm], xs[I]]
                q1 = (np.take_along_axis(d1, xn[Jm][..., None], -1)[..., 0] + .5) / (d1.sum(-1) + 1.)
                d0 = self.fq0[J, qb]
                q0 = (d0[J, xn] + .5) / (d0.sum(1) + 1.)
                self.cev[:, li, :] += np.log(q1) - np.log(q0)[None, :]
                self.fq1[I, li, Jm, qb[Jm], xs[I], xn[Jm]] += 1.
        if self.cond and self._q is not None:
            self.fq0[J, np.searchsorted(QBINS, self._q), xn] += 1.
        self.f0[J, xo, xn] += 1.

    def _reslot(self):
        base = len(self.own_lags) + self.A
        L = len(self.lags)
        for j in range(self.J):
            e = (self.cev if self.cond else self.sev)[:, :, j].copy()   # (J, L) candidate scores
            for li, d in enumerate(self.lags):
                if d in self.own_lags:
                    e[j, li] = -np.inf
            sl = self.slots[j]
            for s in sl:
                if s is not None:
                    e[s[0], self.lags.index(s[1])] = -np.inf
            if self.one_lag:          # another lag of a slotted source must beat that slot
                for s in sl:
                    if s is not None:
                        cur = self.sev[s[0], self.lags.index(s[1]), j]
                        row = e[s[0]]
                        row[row <= cur + SWAP] = -np.inf
            bi, bl = divmod(int(np.argmax(e)), L)
            best = e[bi, bl]
            if not np.isfinite(best) or best <= (SWAP if self.cond else 0.):
                continue
            vals = [self.sev[s[0], self.lags.index(s[1]), j] if s is not None else -np.inf
                    for s in sl]
            p = int(np.argmin(vals))
            if self.one_lag:          # the same source competes only with its own slot (lag change)
                same = [q for q, s in enumerate(sl) if s is not None and s[0] == bi]
                if same:
                    p = same[0]
            if sl[p] is None or best > vals[p] + SWAP:
                sl[p] = (bi, self.lags[bl])
                self._reset_pos(j, base + p)

    def _reset_pos(self, j, pos):
        for i, s in enumerate(self.sets):
            if pos in s:
                sub = self.sets.index(tuple(c for c in s if c != pos))
                for t in (self.q1, self.q0):
                    t[:, j, self.off[i]:self.off[i + 1]] = 0.
                self.g1[j, self.off[i]:self.off[i + 1]] = 0.
                self.g0[j, self.off[i]:self.off[i + 1]] = 0.
                self.ev[:, j, i] = self.ev[:, j, sub]
                self.evg[j, i] = self.evg[j, sub]
        self.mu[:, :, j, pos] = 0.
        self.Sg[:, :, j, pos, :] = 0.
        self.Sg[:, :, j, :, pos] = 0.
        self.Sg[:, :, j, pos, pos] = self.s0

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        u = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        for nm in ("q1", "q0", "ev"):
            setattr(self, nm, getattr(self, nm)[..., idx])
        for nm in ("mu", "Sg", "hz", "mz", "npz", "used", "z", "hp", "mp", "usedp", "y", "tm1",
                   "tm0"):
            setattr(self, nm, getattr(self, nm)[idx].copy())
        self.logw = np.zeros(self.N)
