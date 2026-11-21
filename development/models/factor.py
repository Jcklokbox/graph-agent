"""Scale-separated, online factor model.

The place view is a causal hard assignment from persistent observation channels.
All parent lags and fast-channel outcomes are confined to the regime filter.
Each place remembers its last sampled regime; while unobserved its regime prior
relaxes towards the place's historical mixture. The local outcome model uses
hierarchical Beta cells and signed additive weights with a shared Gaussian mean.
An independent process expert is scored prequentially across regimes, not trained
on a particle's sampled allocation.
"""
from __future__ import annotations

from itertools import combinations
import re

import numpy as np

from models.discovery import NeighborBank
from models.latent_cause import H_GRID
from models.kernels import (copy_ancestors, factor_gaussian, factor_read, factor_update,
                            systematic_ancestry)

C_REG = 1.           # concentration: regime cell -> pooled cell
C_GLOB = 1.          # concentration: pooled cell -> sub-configurations
SWAP = np.log(3.)    # slot swap margin (v18)
GAP_SCALE = 64.     # steps for relaxation of an unobserved regime
NEW_ASSOC = .40    # association shrinkage in an as-yet unused regime
FREE_RATE = .12     # mixture mass for an event without an observed parent
EXPERT_SWITCH = .002  # hazard of switching the predictive family
F32 = np.float32
CANONICAL = "factor-L4-P-C"    # M_factor: the one configuration used by every suite (SPEC-3b.2)


def make_factor(name, n, actions, value, seed=0):
    """One configuration; only the dev particle count may differ."""
    match = re.fullmatch(rf"{CANONICAL}(?:-N([1-9][0-9]*))?", name)
    if match is None:
        raise ValueError(f"unknown factor configuration {name!r}; expected {CANONICAL}[-N<count>]")
    return FactorModel(n, actions, value, seed, N=int(match[1] or 200),
                       lags=range(5), own_lags=(0, 1, 2), k=5, process=True, cond=True)


class FactorModel:
    def __init__(self, n, A, v, seed=0, N=200, K=10, k=4, lags=(0,), own_lags=(0,),
                 beta=1., alpha=1., noise=.25, s0=1., process=False, cond=False):
        self.n, self.A, self.v = n, A, v
        self.cond = cond             # r5 deep form: screen against the model's own prediction
        self.proc = None
        if process:                  # F3: hidden processes (Decomposer groups, windows) as a member
            from models.decompose import Decomposer
            self.proc = Decomposer(n, A, v, seed, windows=True)
        self.N, self.K, self.k = N, K, k
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
        self._pair_pos = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self._sa = np.array([single[s[0]] for s in pairs], int)
        self._sb = np.array([single[s[1]] for s in pairs], int)
        self.M = self.S + 1                                       # + additive
        C = int(self.off[-1])
        # Each particle's regime/member bank is contiguous for filtering and resampling.
        self.q1 = np.zeros((N, K, J, C), F32)
        self.q0 = np.zeros_like(self.q1)
        self.ev = np.zeros((N, K, J, self.M), F32)
        self._E = np.ones_like(self.ev)
        # pooled tables (particle-independent)
        self.g1 = np.zeros((J, C))
        self.g0 = np.zeros((J, C))
        self.evg = np.zeros((J, self.S))
        d = self.m + 1
        self.noise, self.s0 = noise, s0
        self.mu = np.zeros((N, K, J, d))
        self.Sg = np.broadcast_to(np.eye(d) * s0, (N, K, J, d, d)).copy()
        self.shared_mu = np.zeros((J, d))
        self.shared_cov = np.broadcast_to(np.eye(d) * s0, (J, d, d)).copy()
        self.rate1 = np.zeros((N, K, J))
        self.rate0 = np.zeros_like(self.rate1)
        self.expert = np.full(J, .5)
        # regime factor
        self.hz = np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=N)]
        self.hz = np.maximum(self.hz, .003)
        self.mz = np.zeros((N, K))
        self.used = np.ones(N, int)
        self.z = np.zeros(N, int)
        self.beta, self.alpha = beta, alpha
        self.mz[:, 0] = 1.
        self.places = {}       # signature -> (last time, regime posterior, historical occupancy)
        self.place = ()
        self.slow = np.zeros(J, bool)
        self._role_log = np.zeros((J, len(H_GRID)))
        self._phasic_log = np.zeros(J)
        self._on_counts = np.zeros((J, 2))
        self.last_bits = np.zeros(J, bool)
        self.slow_counts = np.zeros((J, 2, 2))
        self.logw = np.zeros(N)
        self.started = False
        # slots and screening (deterministic: outside the particles)
        self.slots = [[None] * k for _ in range(J)]              # (channel position, lag)
        L = len(self.lags)
        # Retained parent slots plus a separate, bounded exploration bank.
        self.discovery = NeighborBank(J, 2 * k + 2, seed + 103)
        width = self.discovery.width
        self._q = None
        self.ct = np.zeros((J, width, L))
        self.cx = np.zeros_like(self.ct)
        self.cr = np.zeros_like(self.ct)
        self.cxr = np.zeros_like(self.ct)
        self.cev = np.zeros_like(self.ct)
        if not cond:
            self.f1 = np.zeros((J, width, L, 2, 2, 2))
            self.f0 = np.zeros((J, 2, 2))
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
        bits = U > .5
        cells = np.broadcast_to(self.off[:-1], (self.J, self.S)).copy()
        cells[:, 1:1 + self.m] += bits
        cells[:, 1 + self.m:] += bits[:, self._pair_pos[:, 0]] + 2 * bits[:, self._pair_pos[:, 1]]
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

    def _members(self, b, a, hist, kk):
        U = self._keys(b, a, hist)
        cells = self._cells(U)
        pg = self._pooled(cells).astype(F32)
        v = np.concatenate([U, np.ones((self.J, 1))], 1)
        center = np.einsum("jd,jd->j", self.shared_mu, v)
        q = factor_read(self.q1, self.q0, self._E, self.mu, v, center, cells, pg,
                        kk, ~self.slow, self.used)
        return dict(cells=cells, pg=pg, v=v, center=center, q=q)

    def _signature(self, b):
        return tuple(np.flatnonzero(self.slow & (b > .5)).tolist())

    def _observe_place(self, b):
        """Compare a long-dwell copy process to a fitted Bernoulli continuation.

        Conditional on an on observation, a tonic channel inherits the regime
        lifetime grid; a phasic channel has a Jeffreys-Beta continuation rate.
        Marginal likelihood and sparse prior odds 1:J determine the MAP view;
        a transient run is not an irreversible commitment.
        """
        now = b > .5
        idx = np.flatnonzero(self.last_bits)
        if idx.size:
            on = now[idx]
            grid = np.asarray(H_GRID)[None, :]
            self._role_log[idx] += np.log(np.where(on[:, None], 1. - grid, grid))
            ct = self._on_counts[idx]
            self._phasic_log[idx] += np.log((ct[np.arange(len(idx)), on.astype(int)] + .5)
                                           / (ct.sum(1) + 1.))
            self._on_counts[idx, on.astype(int)] += 1.
            z = self._role_log[idx] - self._phasic_log[idx, None]
            mx = z.max(1)
            odds = mx + np.log(np.exp(z - mx[:, None]).mean(1)) - np.log(self.J)
            self.slow[idx] = odds > 0.
        self.last_bits = now
        return self._signature(b)

    def _regime_prior(self, place, kk):
        ar = np.arange(self.N)
        if not self.started:
            pr = np.zeros((self.N, kk))
            pr[:, 0] = 1.
            return pr
        G = self.mz[:, :kk].copy()
        grow = self.used < kk
        G[ar[grow], self.used[grow]] += self.alpha
        G /= G.sum(1, keepdims=True)
        memory = self.places.get(place)
        if memory is None:
            return G
        last_t, post, counts = memory
        dr = counts[:, :kk] + self.beta * G
        dr /= dr.sum(1, keepdims=True)
        gap = max(0, self.t - last_t - 1)
        retain = (1. - self.hz) * np.exp(-gap / GAP_SCALE)
        dr *= (1. - retain[:, None])
        dr += retain[:, None] * post[:, :kk]
        return dr

    def _process_scope(self):
        """Committed groups get the process expert when no slow scope is identified.

        A merge has paid its joint Bayes factor; mixing the committed group's
        channel marginals back into unrelated regime tables loses that evidence.
        Singleton predictions also retain that expert when their neighbors have
        supplied evidence for a factorized stream.
        """
        return self.proc is not None and not self.slow.any() and sum(
            len(g.chans) > 1 for g in self.proc.groups) >= 2

    def _conditional(self, mem, kk, pp, quiet=False):
        # A small parent-missing component makes a free event evidence about the
        # baseline rate of an existing regime, without discarding the cue model.
        n1, n0 = self.rate1[:, :kk], self.rate0[:, :kk]
        pooled = (n1.sum(1, keepdims=True) + .5) / ((n1 + n0).sum(1, keepdims=True) + 1.)
        rate = (n1 + pooled) / (n1 + n0 + 1.)
        fresh = np.arange(kk)[None, :] >= self.used[:, None]
        free = np.where(fresh, NEW_ASSOC, FREE_RATE if quiet else 0.)[..., None]
        if pp is None:
            return mem["q"] * (1. - free) + free * rate
        local = mem["q"] if not self.slow.any() else mem["q"] * (1. - free) + free * rate
        return self.expert * local + (1. - self.expert) * pp

    @staticmethod
    def _activity(pj, w, b, fast):
        """For a quiet input, separate the shared event clock from its regime-specific marks.

        Silence carries timing but is not evidence for a different contingency.
        The nonempty Bernoulli patterns are normalized by their total mass.
        """
        if not fast.any() or np.any(b[fast]):
            return None
        idle = np.prod(1. - pj[..., fast], axis=2)
        background = (w * idle).sum() / w.sum()
        return background, (1. - background) / np.maximum(1. - idle, 1e-12)

    # --------------------------------------------------------------- introspection
    def trace(self):
        """Deterministic summary of the posterior state (function of code and seed only)."""
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        E = self.ev[np.arange(self.N), self.z]
        E = np.exp(E - E.max(2, keepdims=True))
        E = (E / E.sum(2, keepdims=True) * w[:, None, None]).sum(0)
        return dict(t=self.t, births=self.births, place=list(self.place),
                    slow=np.flatnonzero(self.slow).tolist(),
                    regime=np.bincount(self.z, weights=w, minlength=self.K).tolist(),
                    used=float(w @ self.used), slots=[[list(s) if s is not None else None
                                                       for s in row] for row in self.slots],
                    w_add=E[:, self.S].tolist(), expert=self.expert.tolist())

    # --------------------------------------------------------------- interface
    def predict(self, x, a, hypothetical=False):
        b = self._bits(x)
        key = (b.tobytes(), a)
        if not hypothetical and self._memo is not None and key == self._memo[0]:
            return self._memo[1]
        if self._process_scope():
            out = self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a)
            if not hypothetical:
                self._memo = key, out, None, None
            return out
        kk = min(self.K, int(self.used.max()) + 1)
        hist = self.hist
        pp = None
        if self.proc is not None:
            pp = (self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a))[self.ch]
        mem = self._members(b, a, hist, kk)
        prior = self._regime_prior(self._signature(b), kk)
        out = np.full(self.n, .5)
        w = np.exp(self.logw - self.logw.max())[:, None] * prior
        pj = self._conditional(mem, kk, pp, not b[~self.slow].any())
        activity = self._activity(pj, w, b, ~self.slow)
        w_eval = w if activity is None else w * activity[1]
        out[self.ch] = np.einsum("nk,nkj->j", w_eval, pj) / w.sum()
        if self.slow.any():
            js = np.flatnonzero(self.slow)
            c = self.slow_counts[js, b[js].astype(int)]
            out[np.asarray(self.ch)[js]] = (c[:, 1] + .5) / (c.sum(1) + 1.)
        if not hypothetical:
            self._memo = key, out, mem, pp
        return out

    def learn(self, x, a, y):
        N, ar = self.N, np.arange(self.N)
        b, bn = self._bits(x), self._bits(y)
        yb = bn > .5
        kk = min(self.K, int(self.used.max()) + 1)
        key = (b.tobytes(), a)
        place = self._observe_place(b)
        if self._process_scope():
            if self._memo is None or key != self._memo[0]:
                self.proc.predict(x, a)
            self.proc.learn(x, a, y)
            self._memo = None
            self.hist = (self.hist + [b])[-max(1, self.D + 1):]
            self.place = place
            jr = np.arange(self.J)
            self.slow_counts[jr, b.astype(int), yb.astype(int)] += 1.
            self.t += 1
            return
        if self._memo is not None and key == self._memo[0] and self._memo[2] is not None:
            _, _, mem, pp = self._memo
        else:
            pp = self.proc.predict(x, a)[self.ch] if self.proc is not None else None
            mem = self._members(b, a, self.hist, kk)
        if self.proc is not None:
            # A cached prediction belongs to the most recent actual predict call.
            self.proc.learn(x, a, y)
        jr = np.arange(self.J)
        self.slow_counts[jr, b.astype(int), yb.astype(int)] += 1.
        self.hist = (self.hist + [b])[-max(1, self.D + 1):]
        hist_before = self.hist[:-1]
        self._memo = None
        self.place = place
        if place != self._signature(bn):
            # Exogenous boundary: it is not a regime observation of the old place.
            self.t += 1
            return
        fast = ~self.slow
        prior = self._regime_prior(place, kk)
        w = np.exp(self.logw - self.logw.max())[:, None] * prior
        quiet = not b[fast].any()
        pj = self._conditional(mem, kk, pp, quiet)
        activity = self._activity(pj, w, b, fast)
        w_eval = w if activity is None else w * activity[1]
        raw = np.einsum("nk,nkj->j", w, pj) / w.sum()
        self._q = raw if activity is None else np.einsum("nk,nkj->j", w_eval, pj) / w.sum()
        if pp is not None:
            # Prequential family evidence is shared, not restarted with a regime.
            q_local = (raw - (1. - self.expert) * pp) / self.expert
            py, qy = np.where(yb, pp, 1. - pp), np.where(yb, q_local, 1. - q_local)
            e = self.expert * qy / (self.expert * qy + (1. - self.expert) * py)
            if not self.slow.any() or b[fast].any():
                self.expert[fast] = EXPERT_SWITCH + (1. - 2. * EXPERT_SWITCH) * e[fast]
        # Only the contextual expert provides evidence about the contextual regime.
        # The process expert is updated on the same stream, but does not explain away
        # a change point in the local model.
        pz = self._conditional(mem, kk, None, quiet)
        ll = self._activity(pz, w, b, fast)
        lik = np.prod(np.where(yb[fast], pz[..., fast], 1. - pz[..., fast]), axis=2)
        if ll is not None:
            lik = lik * ll[1] if yb[fast].any() else ll[0]
        joint = prior * lik
        tot = joint.sum(1)
        self.logw += np.log(tot)
        filtered = joint / tot[:, None]
        cdf = np.cumsum(filtered, axis=1)
        zt = np.minimum((cdf < self.rng.random(N)[:, None]).sum(1), kk - 1)
        self.mz[ar, zt] += (zt != self.z) | (self.used <= zt)
        self.births += int((zt >= self.used).sum())
        self.used = np.maximum(self.used, zt + 1)
        self.z = zt
        self.started = True
        if place not in self.places:
            self.places[place] = (self.t, None, np.zeros_like(self.mz))
        counts = self.places[place][2]
        filtered *= np.arange(kk)[None, :] < self.used[:, None]
        filtered /= filtered.sum(1, keepdims=True)
        post = np.zeros_like(self.mz)
        post[:, :kk] = filtered
        # Silence is a clock observation, not an independent assignment. Store
        # the posterior at marks so gap relaxation is applied once to the anchor.
        if b[fast].any() or yb[fast].any() or self.places[place][1] is None:
            counts += post
            self.places[place] = (self.t, post, counts)
        self.rate1[ar, zt] += yb & fast
        self.rate0[ar, zt] += ~yb & fast
        cells = mem["cells"]
        factor_update(self.q1, self.q0, self.ev, self._E, self.mu, mem["v"],
                      mem["center"], cells, mem["pg"], zt, yb, fast)
        factor_gaussian(self.mu, self.Sg, self.shared_mu, self.shared_cov,
                        mem["v"], mem["center"], zt, yb, fast, self.noise)
        pg = np.clip(self._pooled(cells), 1e-6, 1. - 1e-6)
        self.evg += np.log(np.where(yb[:, None], pg, 1. - pg)) * fast[:, None]
        self.g1[jr[:, None], cells] += (yb & fast)[:, None]
        self.g0[jr[:, None], cells] += (~yb & fast)[:, None]
        self._resample()
        self._screen(b, bn, hist_before)
        self.t += 1
        self._reslot()

    # ---------------------------------------------------------------- slots
    def _screen(self, b, bn, hist):
        J = np.arange(self.J)
        xo, xn = b.astype(int), bn.astype(int)
        src = self.discovery.source
        if not self.cond:
            c0 = self.f0[J, xo]
            p0 = (c0[J, xn] + .5) / (c0.sum(1) + 1.)
            I, S = np.ogrid[:self.J, :self.discovery.width]
        residual = xn - self._q
        for li, d in enumerate(self.lags):
            if d > len(hist):
                continue
            xs = self._lag(b, d, hist)[src]
            if self.cond:
                ct = self.ct[:, :, li]
                cx = self.cx[:, :, li]
                cr = self.cr[:, :, li]
                ct += 1.
                cx += xs
                cr += residual[:, None]
                self.cxr[:, :, li] += xs * residual[:, None]
                cov = self.cxr[:, :, li] - cx * cr / ct
                self.cev[:, :, li] = cov ** 2 / (1. + cx * (1. - cx / ct))
            else:
                xs = xs.astype(int)
                c1 = self.f1[I, S, li, xo[:, None], xs]
                p1 = (np.take_along_axis(c1, xn[:, None, None], -1)[..., 0] + .5) / (c1.sum(-1) + 1.)
                self.cev[:, :, li] += np.log(p1) - np.log(p0)[:, None]
                self.f1[I, S, li, xo[:, None], xs, xn[:, None]] += 1.
        if not self.cond:
            self.f0[J, xo, xn] += 1.

    def _reslot(self):
        if not self.discovery.width:
            return
        base = len(self.own_lags) + self.A
        L = len(self.lags)
        for j, sources in enumerate(self.discovery.source):
            scores = self.cev[j]
            e = scores.copy()
            row = {i: q for q, i in enumerate(sources)}
            if j in row:
                for li, d in enumerate(self.lags):
                    if d in self.own_lags:
                        e[row[j], li] = -np.inf
            if self.cond:
                e[self.slow[sources]] = -np.inf
            sl = self.slots[j]
            vals = []
            for slot in sl:
                if slot is None:
                    vals.append(-np.inf)
                    continue
                i, d = slot
                li = self.lags.index(d)
                q = row[i]
                vals.append(scores[q, li])
                if self.cond:
                    e[q] = -np.inf
                else:
                    e[q, li] = -np.inf
            bi, bl = divmod(int(np.argmax(e)), L)
            margin = .25 if self.cond else SWAP
            p = int(np.argmin(vals))
            if e[bi, bl] > margin and e[bi, bl] > vals[p] + margin:
                sl[p] = (int(sources[bi]), self.lags[bl])
                self._reset_pos(j, base + p)
        # A traversal is the exposure budget, not a time-dependent model switch.
        # Keep each challenger for one traversal interval before recycling a column.
        if not self.discovery.complete and self.t % self.J == 0:
            pinned = np.array([[slot[0] if slot else -1 for slot in row] for row in self.slots])
            rr, cc = self.discovery.rotate(pinned)
            for table in (self.ct, self.cx, self.cr, self.cxr, self.cev):
                table[rr, cc] = 0.
            if not self.cond:
                self.f1[rr, cc] = 0.

    def _reset_pos(self, j, pos):
        for i, s in enumerate(self.sets):
            if pos in s:
                sub = self.sets.index(tuple(c for c in s if c != pos))
                for t in (self.q1, self.q0):
                    t[:, :, j, self.off[i]:self.off[i + 1]] = 0.
                self.g1[j, self.off[i]:self.off[i + 1]] = 0.
                self.g0[j, self.off[i]:self.off[i + 1]] = 0.
                self.ev[:, :, j, i] = self.ev[:, :, j, sub]
                self.evg[j, i] = self.evg[j, sub]
        ev = self.ev[:, :, j]
        self._E[:, :, j] = np.exp(ev - ev.max(-1, keepdims=True))
        self.shared_mu[j, pos] = 0.
        self.shared_cov[j, pos, :] = 0.
        self.shared_cov[j, :, pos] = 0.
        self.shared_cov[j, pos, pos] = self.s0
        self.mu[:, :, j, pos] = 0.
        self.Sg[:, :, j, pos, :] = 0.
        self.Sg[:, :, j, :, pos] = 0.
        self.Sg[:, :, j, pos, pos] = self.s0

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        idx = systematic_ancestry(w, self.rng.random())
        live = int(self.used.max())
        # An unused row has never been scored; resetting a parent slot preserves
        # its initial counts, evidence and covariance. Every ancestor beyond the
        # previous maximum therefore has identical contents.
        for bank in (self.q1, self.q0, self.ev, self._E, self.mu, self.Sg,
                     self.mz, self.rate1, self.rate0):
            copy_ancestors(bank.reshape(self.N, self.K, -1), idx, live)
        for nm in ("hz", "used", "z"):
            setattr(self, nm, getattr(self, nm)[idx])
        self.places = {p: (t, z[idx], n[idx]) for p, (t, z, n) in self.places.items()}
        self.logw = np.zeros(self.N)
