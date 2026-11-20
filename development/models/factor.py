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
RHO_SLOW = .55       # scale separation: lag-1 autocorrelation above which a channel is
                    # 'slow' (place-scale); lag copies / phasic channels stay in the
                    # regime + theta factors (SPEC-3b.2 [1] item 1: no lag-cascade
                    # context fragmentation, US-alone exposure updates regimes)
NWARM = 8            # steps before the slow/fast split is trusted
KAPPA = .02          # Delta-tau diffusion of the regime on place re-entry: keep-mass
                    # (1-lambda) on the last regime seen at that place, lambda = 1-exp(-KAPPA*Delta)
SURP0, SURP_K = .5, 4.   # surprise = per-channel neg-log-lik above SURP0 nats; the sticky
                    # keep-mass of the regime decays exp(-SURP_K * surprise): US-alone
                    # exposure re-assigns the regime from the entry weights (no theta erasing)
F32 = np.float32
CANONICAL = "factor-L4-P-C"    # M_factor: the one configuration used by every suite (SPEC-3b.2)


class FactorModel:
    def __init__(self, n, A, v, seed=0, N=200, K=10, k=4, lags=(0,), own_lags=(0,),
                 beta=1., alpha=1., noise=.25, s0=1., process=False, cond=False):
        self.n, self.A, self.v = n, A, v
        self.cond = cond             # r5 deep form: screen against the model's own prediction
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
        self.M = self.S + 2 + bool(process)                      # + add [+ proc] + place
        C = int(self.off[-1])
        # regime tables, layout (regime, channel, cell / member, particle)
        self.q1 = np.zeros((K, J, C, N), F32)
        self.q0 = np.zeros((K, J, C, N), F32)
        self.ev = np.zeros((K, J, self.M, N), F32)
        self._E = None                           # exp(ev - max) cache, same layout as ev
        # pooled tables (particle-independent)
        self.g1 = np.zeros((J, C))
        self.g0 = np.zeros((J, C))
        self.evg = np.zeros((J, self.S))
        d = self.m + 1
        self.noise, self.s0 = noise, s0
        self.mu = np.zeros((N, K, J, d))
        self.Sg = np.broadcast_to(np.eye(d) * s0, (N, K, J, d, d)).copy()
        # scale separation ([1]): running lag-1 stats decide which channels are slow
        self.sxy = np.zeros(J)
        self.sx = np.zeros(J)
        self.sy = np.zeros(J)
        self.ntj = 0
        self.slow = np.zeros(J, bool)
        # Delta-tau bookkeeping: last-visit step and last regime per (particle, place)
        self.tvis = np.zeros((N, self.KP), int)
        self.zlast = np.zeros((N, self.KP), int)
        self.snll = 0.               # last step's mean per-channel surprise of the BMA
        # pooled member evidence and pooled theta: the hierarchical prior a NEW regime
        # inherits ([1] item 1: no reset to an uninformed prior)
        self.evm = np.zeros((J, self.M))
        self.mug = np.zeros((J, d))
        self.Sgg = np.broadcast_to(np.eye(d) * s0, (J, d, d)).copy()
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

    @staticmethod
    def _flat3(q):
        """(K, J, C, N) -> (K, J*C, N) view (never a copy: writes/reads must hit the table)."""
        assert q.flags.c_contiguous
        return q.reshape(q.shape[0], -1, q.shape[3])

    def _members(self, b, a, hist, kk, pp=None):
        """Regime-indexed member predictions P (kk, J, S+1, N) [sets + additive], their
        pooled-weight-free mixture parts and the place member's weight; plus keys."""
        self._prep_new()
        U = self._keys(b, a, hist)
        cells = self._cells(U)
        pg = self._pooled(cells).astype(F32)                       # (J, S)
        # batched gather of the regime counts: one flat take over (J*C) per table (exact copy)
        J, C, N, S = self.J, self.q1.shape[2], self.N, self.S
        fidx = (np.arange(J)[:, None] * C + cells).ravel()         # (J*S,)
        n1 = np.take(self._flat3(self.q1)[:kk], fidx, axis=1)      # (kk, J*S, N)
        n0 = np.take(self._flat3(self.q0)[:kk], fidx, axis=1)
        M1 = S + 1 + (self.proc is not None)                       # sets + add [+ proc]
        P = np.empty((kk, J, M1, N), F32)
        ps = P[:, :, :S]                                           # sets, written in place
        ps[...] = (n1 + C_REG * pg.reshape(1, J * S, 1)).reshape(kk, J, S, N)
        n1 += n0
        n1 += C_REG
        ps /= n1.reshape(kk, J, S, N)
        v = np.concatenate([U, np.ones((self.J, 1))], 1)           # (J, d)
        P[:, :, S] = np.clip(np.einsum("nkjd,jd->kjn", self.mu[:, :kk], v), .01, .99)
        if self.proc is not None:
            pp = np.full(self.J, .5) if pp is None else pp
            P[:, :, S + 1] = np.clip(pp, .001, .999).astype(F32)[None, :, None]
        if self._E is None:                    # cached exp(ev - max_m ev); updated in learn
            self._E = np.exp(self.ev - self.ev.max(2, keepdims=True))
        E = self._E[:kk]                                           # (kk, J, M, N)
        Z = E.sum(2)
        rest = (E[:, :, :-1] * P).sum(2) / Z                       # (kk, J, N)
        wpl = E[:, :, -1] / Z                                      # (kk, J, N)
        return dict(U=U, cells=cells, pg=pg, P=P, rest=rest, wpl=wpl, v=v)

    def _place_pred(self, b):
        """Place member: P(x_j,t+1 = 1 | x_j,t, y) for every place, (N, KP, J)."""
        Jr = np.arange(self.J)
        pv = (b > .5).astype(int)
        n1, n0 = self.tm1[:, :, Jr, pv], self.tm0[:, :, Jr, pv]
        k1, k0 = n1.sum(1, keepdims=True), n0.sum(1, keepdims=True)
        mu = (k1 + .5) / (k1 + k0 + 1.)
        return (n1 + mu) / (n1 + n0 + 1.)

    # --------------------------------------------------------------- helpers
    def _prep_new(self):
        """The CRP's next regime draws from the base measure: the pooled theta and
        member evidence (a new regime is expected to resemble the pooled data)."""
        live = self.used < self.K
        if live.any():
            ar = np.nonzero(live)[0]
            self.mu[ar, self.used[ar]] = self.mug
            self.Sg[ar, self.used[ar]] = self.Sgg
            self.ev[self.used[ar], :, :, ar] = self.evm
            if self._E is not None:
                evz = self.ev[self.used[ar], :, :, ar]                 # (n, J, M)
                self._E[self.used[ar], :, :, ar] = np.exp(evz - evz.max(2, keepdims=True))

    def _place_marg(self):
        """P(x_j,t+1 = 1 | y): the place's marginal emission of the slow channels, with
        the prior mean pooled over the particle's places. The marginal (not the chain)
        is what identifies a place on entry: the previous observation's x_j,t comes from
        the place being left, so conditioning on it would fragment the context."""
        n1 = self.tm1.sum(3)                                  # (N, KP, J)
        n0 = self.tm0.sum(3)
        k1, k0 = n1.sum(1, keepdims=True), n0.sum(1, keepdims=True)
        mu = (k1 + .5) / (k1 + k0 + 1.)
        return (n1 + mu) / (n1 + n0 + 1.)

    def _lik(self, pj, yb):
        """P(obs | place, regime), scale-separated: slow channels are scored by the
        place's own marginal emission (no lag features, no fast members in the place
        path); fast channels by the (place, regime) mixture."""
        lik = np.ones(pj.shape[:3])
        fa, sl = ~self.slow, self.slow
        if fa.any():
            q = pj[..., fa]
            lik *= np.prod(np.where(yb[None, None, None, fa], q, 1. - q), 3)
        if sl.any():
            q = self._place_marg()[:, :, sl]
            lik *= np.prod(np.where(yb[None, sl], q, 1. - q), 2)[:, :, None]
        return lik

    def _slow_update(self, b, bn):
        """Running lag-1 autocorrelation per channel -> the slow/fast split."""
        self.sxy += b * bn
        self.sx += b
        self.sy += bn
        self.ntj += 1
        n = self.ntj
        mx, my = self.sx / n, self.sy / n
        c = self.sxy / n - mx * my
        vt, vy = mx - mx * mx, my - my * my
        rho = c / np.sqrt(np.maximum(vt * vy, 1e-12))
        self.slow = (n >= NWARM) & (vt > .01) & (vy > .01) & (rho > RHO_SLOW)

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
        """P(z | y, previous y, previous z) for every place y: (N, KP, kk).
        Entering a place after Delta steps away keeps the last regime seen there with
        mass 1 - lambda(Delta) (inter-session gaps update regime weights)."""
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
        lam = 1. - np.exp(-KAPPA * (self.t - self.tvis))               # (N, KP)
        Dr *= lam[:, :, None]
        kp = np.arange(self.KP)
        Dr[ar[:, None], kp[None, :], np.minimum(self.zlast, kk - 1)] += 1. - lam
        h = 1. - (1. - self.hz) * np.exp(-SURP_K * np.maximum(0., self.snll - SURP0))
        cur = Dr[ar, self.y].copy()                                # draw in the same place
        Dr[ar, self.y] = h[:, None] * cur
        Dr[ar, self.y, self.z] += 1. - h
        return Dr

    def _joint(self, mem, pl, kk):
        """Per-channel mixture for every (particle, place, regime): (N, KP, kk, J)."""
        rest = mem["rest"].transpose(2, 0, 1)                      # (N, kk, J)
        wpl = mem["wpl"].transpose(2, 0, 1)
        return rest[:, None] + wpl[:, None] * pl[:, :, None, :]

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
                    w_rest=r(E[:, S + 1:].sum(1)), slow=[int(u) for u in self.slow])

    # --------------------------------------------------------------- interface
    def predict(self, x, a, hypothetical=False):
        """Marginals of all n channels at t+1 given x_t = x; value channel .5.
        hypothetical: x has not been learned (a probe) — the transition from the last learned
        observation to x is first used as evidence about (place, regime)."""
        b = self._bits(x)
        key = (b.tobytes(), a, self.t, hypothetical)
        if self._memo is not None and key in self._memo:
            return self._memo[key][0]
        kk = min(self.K, int(self.used.max()) + 1)
        w = np.exp(self.logw - self.logw.max())
        pp = None
        if self.proc is not None:
            pp = (self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a))[self.ch]
        if hypothetical and self.hist:
            prev = self.hist[-1]
            mem0 = self._members(prev, 0, self.hist[:-1], kk, self._pp)
            pl0 = self._place_pred(prev)
            pj = self._joint(mem0, pl0, kk)
            lik = self._lik(pj, b > .5)
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
        pp = None
        if self._memo is not None and key in self._memo \
                and self._memo[key][1]["rest"].shape[0] == kk:
            mem = self._memo[key][1]
        else:
            pp = self.proc.predict(x, a)[self.ch] if self.proc is not None else None
            mem = self._members(b, a, self.hist, kk, pp)
        if self.proc is not None:
            self._pp = mem["P"][0, :, -1, 0].astype(float)
            if pp is None:                   # (just predicted (x, a) above: same state, skip)
                self.proc.predict(x, a)      # the chosen action's prior (a planner may have
            self.proc.learn(x, a, y)         # evaluated other actions last)
        pl = self._place_pred(b)                                   # (N, KP, J)
        pj = self._joint(mem, pl, kk)
        lik = self._lik(pj, yb)                                   # (N, KP, kk)
        prior = self._place_prior()[:, :, None] * self._regime_prior(kk)
        if self.cond:
            ww = np.exp(self.logw - self.logw.max())[:, None, None] * prior
            self._q = np.einsum("npk,npkj->j", ww, pj) / ww.sum()
        joint = prior * lik
        tot = joint.sum((1, 2))
        self.logw += np.log(tot)
        if self.cond and self._q is not None:      # the BMA's own prediction error
            q = np.clip(np.asarray(self._q, float), 1e-12, 1. - 1e-12)
            self.snll = float(-(yb * np.log(q) + (1 - yb) * np.log(1. - q)).mean())
        else:
            self.snll = -np.log(np.maximum(tot, 1e-300)) / self.J
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
        born = zt >= self.used
        self.births += int(born.sum())
        self.usedp = np.maximum(self.usedp, yt + 1)
        self.used = np.maximum(self.used, zt + 1)
        self.y, self.z = yt, zt
        self.started = True
        self.zlast[ar, yt] = zt
        self.tvis[ar, yt] = self.t + 1
        # member evidence in the sampled regime (place member: the sampled place's chain)
        Pz = np.clip(mem["P"][zt, :, :, ar], 1e-6, 1. - 1e-6)    # (N, J, M-1)
        plz = np.clip(pl[ar, yt], 1e-6, 1. - 1e-6)[:, :, None]       # (N, J, 1)
        Pm = np.concatenate([Pz, plz.astype(F32)], 2)
        self.ev[zt, :, :, ar] += np.where(yb[None, :, None], np.log(Pm), np.log(1. - Pm))
        self.evm += np.where(yb[None, :, None], np.log(Pm), np.log(1. - Pm)).mean(0)
        if self._E is not None:                # only the sampled regime of each particle moved
            evz = self.ev[zt, :, :, ar]                            # (N, J, M)
            self._E[zt, :, :, ar] = np.exp(evz - evz.max(2, keepdims=True))
        cells = mem["cells"]
        # batched count update: indices (zt_n, j, cell_js, n) are distinct (cells of different
        # sets are offset), so one fancy += over all (j, s, n) is exact
        J, C = self.J, self.q1.shape[2]
        for tq, sel in ((self.q1, yb), (self.q0, ~yb)):
            js = np.nonzero(sel)[0]
            if js.size:
                fl = ((zt[None, None, :] * J + js[:, None, None]) * C
                      + cells[js][:, :, None]) * self.N + ar[None, None, :]
                assert tq.flags.c_contiguous
                tq.reshape(-1)[fl.ravel()] += 1.
        Jr = np.arange(self.J)
        pv = (b > .5).astype(int)
        self.tm1[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += yb
        self.tm0[ar[:, None], yt[:, None], Jr[None, :], pv[None, :]] += ~yb
        v = mem["v"]
        Sz = self.Sg[ar, zt]
        var = np.einsum("njab,ja,jb->nj", Sz, v, v) + self.noise
        g = np.einsum("njab,jb->nja", Sz, v) / var[..., None]
        err = yb[None, :].astype(float) - np.einsum("nja,ja->nj", self.mu[ar, zt], v)
        self.mu[ar, zt] += g * err[..., None]
        Sz -= g[..., :, None] * g[..., None, :] * var[..., None, None]   # Sz is a copy (fancy)
        self.Sg[ar, zt] = Sz
        # pooled theta (the hierarchical prior of a new regime; all data, one filter)
        vg = np.einsum("jab,ja,jb->j", self.Sgg, v, v) + self.noise
        gg = np.einsum("jab,jb->ja", self.Sgg, v) / vg[:, None]
        errg = yb.astype(float) - np.einsum("jd,jd->j", self.mug, v)
        self.mug += gg * errg[:, None]
        self.Sgg -= gg[:, :, None] * gg[:, None, :] * vg[:, None, None]
        # pooled level (same for every particle)
        pg = np.clip(self._pooled(cells), 1e-6, 1. - 1e-6)
        self.evg += np.where(yb[:, None], np.log(pg), np.log(1. - pg))
        tg = np.where(yb[:, None], 1., 0.)
        np.add.at(self.g1, (Jr[:, None], cells), tg)
        np.add.at(self.g0, (Jr[:, None], cells), 1. - tg)
        self._resample()
        # screening and slots (deterministic)
        self._screen(b, bn)
        self._slow_update(b, bn)
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
            bi, bl = divmod(int(np.argmax(e)), L)
            best = e[bi, bl]
            if not np.isfinite(best) or best <= (SWAP if self.cond else 0.):
                continue
            vals = [self.sev[s[0], self.lags.index(s[1]), j] if s is not None else -np.inf
                    for s in sl]
            p = int(np.argmin(vals))
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
                self._E = None
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
        for nm in ("q1", "q0"):              # count tables: C order for flat take/scatter
            setattr(self, nm, np.ascontiguousarray(getattr(self, nm)[..., idx]))
        self.ev = self.ev[..., idx]          # layout kept: reductions over ev depend on it
        if self._E is not None:
            self._E = self._E[..., idx]
        for nm in ("mu", "Sg", "hz", "mz", "npz", "used", "z", "hp", "mp", "usedp", "y", "tm1",
                   "tm0", "tvis", "zlast"):
            setattr(self, nm, getattr(self, nm)[idx].copy())
        self.logw = np.zeros(self.N)
