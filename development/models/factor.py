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

from collections import OrderedDict
from itertools import combinations
import re

import numpy as np

from models.discovery import NeighborBank
from models.latent_cause import H_GRID
from models.kernels import (choose_slots, copy_ancestor_rows, factor_gaussian, factor_read, factor_update,
                            reset_factor_slot, systematic_ancestry)

C_REG = 1.           # concentration: regime cell -> pooled cell
C_GLOB = 1.          # concentration: pooled cell -> sub-configurations
SWAP = np.log(3.)    # slot swap margin (v18)
NEW_ASSOC = .40    # association shrinkage in an as-yet unused regime
F32 = np.float32
CANONICAL = "factor-L4-P-C"    # M_factor: the one configuration used by every suite (SPEC-3b.2)


def tonic_log_odds(tonic_ll, phasic_ll):
    """Marginal role odds under a shared, latent dwell-rate mixture.

    Before each channel's own role likelihood, only the other channels are
    evidence about the lifetime grid; each integrates an independent binary
    role prior with odds 1:J. The constant null factor cancels.
    """
    lr = tonic_ll - phasic_ll[:, None] - np.log(len(phasic_ll))
    others = np.logaddexp(0., lr)
    # Joint role-on numerator has factors L_T/L_F, role-off replaces that
    # channel's mixture by one. Work is linear in channels times grid size.
    rest = others.sum(0) - others
    return np.logaddexp.reduce(rest + lr, axis=1) - np.logaddexp.reduce(rest, axis=1)


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
        by_set = {parents: i for i, parents in enumerate(self.sets)}
        self._reset_map = []
        for p in range(self.m):
            items = [(i, by_set[tuple(q for q in parents if q != p)])
                     for i, parents in enumerate(self.sets) if p in parents]
            self._reset_map.append(tuple(np.array([it[k] for it in items], int) for k in (0, 1)))
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
        self.cue = np.zeros_like(self.rate1)
        self.expert = np.full(J, .5)
        self._expert_logodds = np.zeros(J)
        # regime factor
        self.hz = np.asarray(H_GRID)[self.rng.integers(len(H_GRID), size=N)]

        self.mz = np.zeros((N, K))
        self.used = np.ones(N, int)
        self.z = np.zeros(N, int)
        self.beta, self.alpha = beta, alpha
        self.mz[:, 0] = 1.
        # Capacity matches the finite latent inventory. A forgotten location is
        # a prior predictive query, not an unbounded copy on every resample.
        self.places = OrderedDict()  # signature -> (last mark, posterior, occupancy)
        self.background = {}        # bounded place -> [observed zero/one counts]
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
        self._slot_src = np.full((J, k), -1, int)
        self._slot_li = np.zeros((J, k), int)
        self._own_flags = np.array([d in self.own_lags for d in self.lags])
        # Own lags can include a delay outside the discovery grid.
        self._key_lags = tuple(dict.fromkeys(self.lags + self.own_lags))
        self._own_li = [self._key_lags.index(d) for d in self.own_lags]
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
        self._zero = np.zeros(J)
        self.t = 0
        self._memo = None
        self.births = 0

    @property
    def slots(self):
        return [[(int(i), self.lags[l]) if i >= 0 else None for i,l in zip(src,lag)]
                for src,lag in zip(self._slot_src,self._slot_li)]

    # ------------------------------------------------------------------ keys
    def _bits(self, x):
        return (np.asarray(x, float)[self.ch] > .5).astype(float)

    def _lag(self, b, d, hist):
        if d == 0:
            return b
        if len(hist) >= d:
            return hist[-d]
        return self._zero

    def _keys(self, b, a, hist):
        U = np.zeros((self.J, self.m))
        lagged = np.asarray([self._lag(b, d, hist) for d in self._key_lags])
        own = len(self.own_lags)
        U[:, :own] = lagged[self._own_li].T
        U[:, own + a] = 1.
        idx = np.maximum(0, self._slot_src)
        U[:, own+self.A:] = lagged[self._slot_li, idx] * ((self._slot_src >= 0) & ~self.slow[idx])
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

    def _support_deficit(self, cells, U):
        """Compare local joint occupancy to its represented current-cue marginal."""
        if not self.slow.any():
            return self._zero
        current = np.zeros((self.J, self.m), bool)
        current[:, :len(self.own_lags)] = np.asarray(self.own_lags) == 0
        current[:, len(self.own_lags) + self.A:] = ((self._slot_src >= 0)
                                                    & (np.asarray(self.lags)[self._slot_li] == 0))
        active = current & (U > .5)
        ij = self._pair_pos
        n = self.g1[np.arange(self.J)[:, None], cells] + self.g0[np.arange(self.J)[:, None], cells]
        lo = np.maximum(active[:, ij[:, 0]] * n[:, self._sa], active[:, ij[:, 1]] * n[:, self._sb])
        hi = n[:, 1 + self.m:]
        den = lo * (1. + hi)
        delta = np.divide(lo-hi, den, out=np.zeros_like(hi), where=den > 0.)
        delta *= current[:, ij[:, 0]] & current[:, ij[:, 1]]
        return np.maximum(0., delta.max(1)) if delta.shape[1] else self._zero

    def _members(self, b, a, hist, kk):
        U = self._keys(b, a, hist)
        cells = self._cells(U)
        pg = self._pooled(cells).astype(F32)
        v = np.concatenate([U, np.ones((self.J, 1))], 1)
        center = np.einsum("jd,jd->j", self.shared_mu, v)
        deficit = self._support_deficit(cells, U)
        q = factor_read(self.q1, self.q0, self._E, self.mu, v, center, cells, pg,
                        kk, ~self.slow, self.used, deficit)
        return dict(cells=cells, pg=pg, v=v, center=center, q=q, deficit=deficit)

    def _signature(self, b):
        return tuple(np.flatnonzero(self.slow & (b > .5)).tolist())

    def _observe_place(self, b):
        """Compare a long-dwell copy process to a fitted Bernoulli continuation.

        A prior on the common tonic dwell rate shares evidence between
        channels, while each phasic channel has its own uniform-Beta
        continuation rate. Role assignments are MAP and can be revised.
        """
        now = b > .5
        idx = np.flatnonzero(self.last_bits)
        if idx.size:
            on = now[idx]
            grid = np.asarray(H_GRID)[None, :]
            self._role_log[idx] += np.log(np.where(on[:, None], 1. - grid, grid))
            ct = self._on_counts[idx]
            self._phasic_log[idx] += np.log((ct[np.arange(len(idx)), on.astype(int)] + 1.)
                                           / (ct.sum(1) + 2.))
            self._on_counts[idx, on.astype(int)] += 1.
            self.slow = tonic_log_odds(self._role_log, self._phasic_log) > 0.
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
        retain = np.exp((gap + 1) * np.log1p(-self.hz))
        dr *= (1. - retain[:, None])
        dr += retain[:, None] * post[:, :kk]
        return dr

    def _cue_prior(self, prior, b, kk):
        """Input-driven transition, not an extra observation likelihood.

        A cue is a sampled token from currently available non-tonic marks.
        Scores marginalize which token was used, then normalize the row once.
        """
        active = (b > .5) & ~self.slow
        if not self.slow.any() or not active.any():
            return prior
        counts = self.cue[:, :kk, :]
        pooled = counts.sum(axis=1) + .5
        pooled /= pooled.sum(axis=1, keepdims=True)
        contained = (counts[:, :, active].sum(axis=2) + pooled[:, None, active].sum(axis=2))
        pcue = contained / (counts.sum(axis=2) + 1.)
        routed = prior * pcue
        return routed / routed.sum(axis=1, keepdims=True)

    def _conditional(self, mem, kk, place, quiet=False):
        """Place background followed by (if observable) cue-excess emission.

        No quiet observation forces an update of a cue contingency. Regime
        rates provide one pseudocount of shared prior evidence for baselines;
        observed place history owns the rest. The no-place case does not
        distinguish literal silence from a delayed or continuously present cue.
        """
        n1, n0 = self.rate1[:, :kk], self.rate0[:, :kk]
        pooled = (n1.sum(1, keepdims=True) + .5) / ((n1 + n0).sum(1, keepdims=True) + 1.)
        rate = (n1 + pooled) / (n1 + n0 + 1.)
        fresh = (np.arange(kk)[None, :] >= self.used[:, None])[..., None]
        if not self.slow.any():
            # No inferred place: preserve conditional lag and process learning.
            return mem["q"] * (1. - NEW_ASSOC * fresh) + NEW_ASSOC * fresh * rate, np.zeros_like(rate)
        bg = self.background.get(place)
        base = rate if bg is None else (bg[1] + rate) / (1. + bg[0] + bg[1])
        if quiet:
            # Regime-specific marks retain a context-dependent memory path;
            # quiet timing is removed from assignment by _activity below.
            return base, base
        q = mem["q"] * (1. - NEW_ASSOC * fresh)
        return base + (1. - base) * q, base

    @staticmethod
    def _activity(pj, w, b, fast):
        """For a quiet input, separate the shared event clock from its regime-specific marks.

        Silence carries timing but is not evidence for a different contingency.
        The nonempty Bernoulli patterns are normalized by their total mass.
        """
        if not fast.any() or np.any(b[fast]):
            return None
        log_idle = np.log1p(-pj[..., fast]).sum(axis=2)
        with np.errstate(divide="ignore"):
            a = np.log(w) + log_idle
        peak = a.max()
        background = peak + np.log(np.exp(a - peak).sum()) - np.log(w.sum())
        return background, -np.expm1(min(0., background)) / np.maximum(-np.expm1(log_idle), 1e-300)

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
        kk = min(self.K, int(self.used.max()) + 1)
        hist = self.hist
        pp = None
        if self.proc is not None:
            pp = (self.proc.peek_all(x, a) if hypothetical else self.proc.predict(x, a))[self.ch]
        quiet = not b[~self.slow].any()
        mem = None if quiet and self.slow.any() else self._members(b, a, hist, kk)
        place = self._signature(b)
        prior = self._cue_prior(self._regime_prior(place, kk), b, kk)
        out = np.full(self.n, .5)
        w = np.exp(self.logw - self.logw.max())[:, None] * prior
        local, background = self._conditional(mem, kk, place, quiet)
        pj = local if pp is None else self.expert * local + (1. - self.expert) * pp
        activity = self._activity(pj, w, b, ~self.slow)
        w_eval = w if activity is None else w * activity[1]
        q = np.einsum("nk,nkj->j", w_eval, pj) / w.sum()
        out[self.ch] = q
        if self.slow.any():
            js = np.flatnonzero(self.slow)
            c = self.slow_counts[js, b[js].astype(int)]
            out[np.asarray(self.ch)[js]] = (c[:, 1] + .5) / (c.sum(1) + 1.)
        if not hypothetical:
            self._memo = key, out, mem, pp, prior, w, local, background, q
        return out

    def learn(self, x, a, y):
        N, ar = self.N, np.arange(self.N)
        b, bn = self._bits(x), self._bits(y)
        yb = bn > .5
        kk = min(self.K, int(self.used.max()) + 1)
        key = (b.tobytes(), a)
        roles = self.slow.copy()
        place = self._observe_place(b)
        changed = self.slow & ~roles
        if changed.any():
            stale = (self._slot_src >= 0) & changed[np.maximum(0,self._slot_src)]
            for j, pos in zip(*np.nonzero(stale)):
                self._slot_src[j, pos] = -1
                self._reset_pos(j, len(self.own_lags) + self.A + pos)
        cached = (self._memo is not None and key == self._memo[0]
                  and np.array_equal(roles, self.slow))
        if cached:
            _, _, mem, pp, prior, w, pz, base, q = self._memo
        else:
            pp = self.proc.predict(x, a)[self.ch] if self.proc is not None else None
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
        quiet = not b[fast].any()
        if not cached:
            mem = None if quiet and self.slow.any() else self._members(b, a, hist_before, kk)
            prior = self._cue_prior(self._regime_prior(place, kk), b, kk)
            w = np.exp(self.logw - self.logw.max())[:, None] * prior
            pz, base = self._conditional(mem, kk, place, quiet)
            pj = pz if pp is None else self.expert * pz + (1. - self.expert) * pp
            activity = self._activity(pj, w, b, fast)
            w_eval = w if activity is None else w * activity[1]
            q = np.einsum("nk,nkj->j", w_eval, pj) / w.sum()
        self._q = q
        if pp is not None:
            # Prequential family evidence is shared, not restarted with a regime.
            pred_local = np.einsum("nk,nkj->j", w, pz) / w.sum()
            py, qy = np.where(yb, pp, 1. - pp), np.where(yb, pred_local, 1. - pred_local)
            if not self.slow.any() or not quiet:
                logbf = np.log(np.maximum(qy, 1e-12)) - np.log(np.maximum(py, 1e-12))
                self._expert_logodds[fast] += logbf[fast]
                self.expert[fast] = np.exp(-np.logaddexp(0., -self._expert_logodds[fast]))
        # Only the contextual expert provides evidence about the contextual regime.
        # The process expert is updated on the same stream, but does not explain away
        # a change point in the local model.
        ll = self._activity(pz, w, b, fast)
        if ll is not None and not yb[fast].any():
            # The common clock observation cancels from every particle update.
            filtered = prior.copy()
        else:
            py = np.where(yb[fast], pz[..., fast], 1. - pz[..., fast])
            with np.errstate(divide="ignore"):
                joint = np.log(prior) + np.log(py).sum(axis=2)
                if ll is not None:
                    joint += np.log(ll[1])
            mid = joint.max(axis=1)
            zq = np.exp(joint - mid[:, None]); denom = zq.sum(axis=1)
            self.logw += mid + np.log(denom)
            filtered = zq / denom[:, None]
        cdf = np.cumsum(filtered, axis=1)
        zt = np.minimum((cdf < self.rng.random(N)[:, None]).sum(1), kk - 1)
        self.mz[ar, zt] += (zt != self.z) | (self.used <= zt)
        self.births += int((zt >= self.used).sum())
        self.used = np.maximum(self.used, zt + 1)
        self.z = zt
        self.started = True
        if place not in self.places:
            if len(self.places) >= self.K:
                forgotten, _ = self.places.popitem(last=False)
                self.background.pop(forgotten, None)
            self.places[place] = (self.t, None, np.zeros_like(self.mz))
        self.places.move_to_end(place)
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
        scoped = self.slow.any()
        if scoped and not quiet:
            self.cue[ar, zt] += (b > .5) & fast
        if quiet or not scoped:
            self.rate1[ar, zt] += yb & fast
            self.rate0[ar, zt] += ~yb & fast
        if quiet and scoped:
            bg = self.background.setdefault(place, np.zeros((2, self.J)))
            bg[0] += ~yb & fast
            bg[1] += yb & fast
        else:
            cells = mem["cells"]
            bs = base[ar, zt]
            # Conditional expected cue count given the observed label and
            # current regime. Fractional credit approximates posterior updates
            # (lost reduced-state dependence is reintroduced in toy checks).
            obs = pz[ar, zt]
            credit = np.where(yb[None, :], np.maximum(0., obs - bs) / np.maximum(1e-12, obs), 1.)
            factor_update(self.q1, self.q0, self.ev, self._E, self.mu, mem["v"],
                          mem["center"], cells, mem["pg"], zt, yb, fast, bs, credit, mem["deficit"])
            factor_gaussian(self.mu, self.Sg, self.shared_mu, self.shared_cov,
                            mem["v"], mem["center"], zt, yb, fast, self.noise, credit)
            pg = np.clip(self._pooled(cells), 1e-6, 1. - 1e-6)
            post_w = np.exp(self.logw - self.logw.max())
            post_w /= post_w.sum()
            w_cue = np.einsum("n,nj->j", post_w, credit) * fast
            bbar = np.einsum("nk,nkj->j", w, base) / w.sum()
            px = bbar[:, None] + (1. - bbar[:, None]) * pg
            self.evg += np.log(np.where(yb[:, None], px, 1. - px)) * fast[:, None]
            self.g1[jr[:, None], cells] += (yb * w_cue)[:, None]
            self.g0[jr[:, None], cells] += (~yb * w_cue)[:, None]
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
        if not self.discovery.width or not self.k:
            return
        base = len(self.own_lags) + self.A
        change = choose_slots(self.cev, self.discovery.source, self._slot_src, self._slot_li,
                              self._own_flags, self.slow, self.cond, .25 if self.cond else SWAP)
        for j in np.flatnonzero(change[:,0] >= 0):
            pos, src, lag = change[j]
            self._slot_src[j, pos] = src
            self._slot_li[j, pos] = lag
            self._reset_pos(j, base + pos)
        # A traversal is the exposure budget, not a time-dependent model switch.
        if not self.discovery.complete and self.t % self.J == 0:
            rr, cc = self.discovery.rotate(self._slot_src)
            for table in (self.ct, self.cx, self.cr, self.cxr, self.cev):
                table[rr, cc] = 0.
            if not self.cond:
                self.f1[rr, cc] = 0.

    def _reset_pos(self, j, pos):
        affected, parents = self._reset_map[pos]
        for m, sub in zip(affected, parents):
            self.g1[j, self.off[m]:self.off[m + 1]] = 0.
            self.g0[j, self.off[m]:self.off[m + 1]] = 0.
            self.evg[j, m] = self.evg[j, sub]
        self.shared_mu[j, pos] = 0.
        self.shared_cov[j, pos, :] = 0.
        self.shared_cov[j, :, pos] = 0.
        self.shared_cov[j, pos, pos] = self.s0
        reset_factor_slot(self.q1, self.q0, self.ev, self.mu, self.Sg,
                          j, pos, self.off, affected, parents, int(self.used.max()), self.s0)
        ev = self.ev[:, :, j]
        self._E[:, :, j] = np.exp(ev - ev.max(-1, keepdims=True))

    def _resample(self):
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        if 1. / (w ** 2).sum() >= self.N / 2:
            return
        idx = systematic_ancestry(w, self.rng.random())
        # Every row outside a particle's own live prefix is in its prior state.
        # Copying the union of ancestor and destination prefixes both keeps all
        # used data and clears any destination regimes the ancestor never saw.
        for bank in (self.q1, self.q0, self.ev, self._E, self.mu, self.Sg,
                     self.mz, self.rate1, self.rate0, self.cue):
            copy_ancestor_rows(bank.reshape(self.N, self.K, -1), idx, self.used)
        for _, post, occupancy in self.places.values():
            copy_ancestor_rows(post.reshape(self.N, self.K, 1), idx, self.used)
            copy_ancestor_rows(occupancy.reshape(self.N, self.K, 1), idx, self.used)
        for nm in ("hz", "used", "z"):
            setattr(self, nm, getattr(self, nm)[idx])
        self.logw = np.zeros(self.N)
