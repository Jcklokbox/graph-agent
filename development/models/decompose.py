"""Feasibility witness for slice 3: learning the decomposition of anonymous channels into
independent hidden processes, online, at a cost that grows with the number of processes.

Not an agent: a T1 predictor (predict(x, a) / learn(x, a, y)), like models/online_hmm.py.

Generative model claimed:
    channels = disjoint groups g; every group is an independent HMM with K_g hidden states
    (K_g unknown, prior uniform on KS); x_{t, i} | s_{g(i), t} ~ Bernoulli(E_g[s, i]).

Inference (approximate, every piece is a Bayesian comparison of predictive likelihoods):
* each group keeps one online HMM per K in KS (models/online_hmm.py, lag 4) and predicts with
  their posterior mixture: w_K prop. to the product of K's one-step predictive likelihoods
  (prequential = marginal likelihood of K);
* the partition starts from singletons; merges are proposed for the group pairs whose
  channels show the largest running |correlation| at lags 0..LAGS (screening only), and
  accepted by a Bayes factor: a fresh merged model and fresh copies of the two separate
  models are run side by side for W steps (both sides pay their learning cost); merge if
  log P(data | merged) - log P(data | separate) > LOG_BF at the horizon W. Early rejection
  when the difference falls below -REJECT after MIN_AGE steps; no early acceptance: the
  prequential evidence of freshly started online models is not a calibrated likelihood
  ratio in the first few hundred steps (measured: independent pairs reach +14 to +19 nats
  transiently and end at -6 to -117 after 600 steps; same-process pairs end at +58 to +78).
  A rejected pair waits COOL steps before it can be proposed again.
`windows=True` (`decompw`, D-4) adds the window family (last 4 observations of the group) to
every group's mixture, for cue -> delayed-outcome structure the online HMM does not learn.
Cost per step: sum over groups of |KS| * O(lag K^2), plus at most M candidates — linear in
the number of groups for a fixed group size.
"""
from __future__ import annotations

import numpy as np

from models.discovery import NeighborBank
from models.kernels import binary_dependency_scores
from models.online_hmm import PackedHMM

KS = (2, 3, 4, 6)
W, MIN_AGE, LOG_BF, REJECT, M, COOL = 500, 200, 3., 10., 6, 1500
GROUP_WIDTH = 6               # hypothesis class: bounded emission width
LAGS = 4                      # screening: max |correlation| over lags 0..LAGS
C_FAST = 9.                   # fast test: merge when the pairwise log BF exceeds this (D-9)
OWN = 3                       # fast test (null 'own'): j's own last OWN values form the key
QBINS = 8                     # fast test (null 'resid', D-10): bins of j's current group predictive
C_CONJ, CONJ_AGE, CONJ_COOL, CONJ_PEN = 2, 2000, 1000, np.log(4.)  # conjunctive candidates (D-13)
CONJ_REJECT = 10.             # drop a candidate early only on strong evidence against it


class Window:
    """Window family: Laplace counts per (last WIN observations of the group, action) with
    back-off to the longest window seen >= MIN times (as eval.predictive.MarkovK, restricted
    to the group's channels); channels conditionally independent given the window. Shares
    the OnlineHMM interface used by GroupModel (observe / predictive / update)."""
    WIN, MIN = 4, 8

    def __init__(self, m, A):
        self.m, self.t, self.hist = m, {}, []
        self._key_cache = {}

    def observe(self, x0):
        self.hist = [tuple(np.asarray(x0) > .5)]
        self._key_cache.clear()

    def _keys(self, a):
        key = self._key_cache.get(a)
        if key is None or key[0] is not self.hist:
            h = self.hist[-self.WIN:]
            key = self._key_cache[a] = (self.hist, [(tuple(h[-j:]), a) for j in range(1, len(h) + 1)])
        return key[1]

    def marg(self, a):
        best = None
        for key in self._keys(a):
            c = self.t.get(key)
            if c is not None and (best is None or c[1] >= self.MIN):
                best = c
        return np.full(self.m, .5) if best is None else (best[0] + 1.) / (best[1] + 2.)

    def update(self, a, y):
        z = np.asarray(y) > .5
        for key in self._keys(a):
            c = self.t.get(key)
            if c is None:
                c = self.t[key] = [np.zeros(self.m), 0.]
            c[0] += z
            c[1] += 1.
        self.hist = (self.hist + [tuple(z)])[-self.WIN:]


class GroupModel:
    """Posterior mixture over model families on a fixed set of channels: online HMMs with
    K in ks, plus (windows=True) the window family."""

    def __init__(self, chans, A, seed, ks=KS, windows=False):
        self.chans = list(chans)
        self.hs = [PackedHMM(len(self.chans), A, K=k, seed=seed + 7 * k) for k in ks]
        if windows:
            self.hs.append(Window(len(self.chans), A))
        self.logw = np.zeros(len(self.hs))
        self.started = False
        self._prior = None

    def start(self, x):
        xs = np.asarray(x, float)[self.chans]
        for h in self.hs:
            h.observe(xs)
        self.started = True

    def predict(self, x, a, read=True):
        if not self.started:
            self.start(x)
        heads = [((p := h.marg(a)), p) if isinstance(h, Window) else h.forward(a, read)
                 for h in self.hs]
        self._prior = [p for p, _ in heads]
        if read:
            w = np.exp(self.logw - self.logw.max())
            w /= w.sum()
            return sum(wk * q for wk, (_, q) in zip(w, heads))

    @staticmethod
    def _lp(p, h, ys):
        if isinstance(h, Window):
            q = np.clip(p, 1e-6, 1. - 1e-6)
            return float(np.sum(np.where(ys > .5, np.log(q), np.log(1. - q))))
        return np.log(max(1e-300, float(p @ h._lik(ys, h.E()))))

    def peek(self, x, a):
        """Predictive marginals for the step after a hypothetical next observation x (not
        learned): every family member conditions on x_g, the weights too."""
        xs = np.asarray(x, float)[self.chans]
        lw, preds = [], []
        for wk, h in zip(self.logw, self.hs):
            if isinstance(h, Window):
                p = h.marg(a)
                q = np.clip(p, 1e-6, 1. - 1e-6)
                lw.append(wk + float(np.sum(np.where(xs > .5, np.log(q), np.log(1. - q)))))
                saved = h.hist
                h.hist = (h.hist + [tuple(xs > .5)])[-h.WIN:]
                preds.append(h.marg(a))
                h.hist = saved
            else:
                prior = h.b @ h.T(a)
                post = prior * h._lik(xs, h.E())
                z = post.sum()
                lw.append(wk + np.log(max(z, 1e-300)))
                preds.append((post / max(z, 1e-300)) @ h.T(a) @ h.E())
        lw = np.asarray(lw)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return sum(wk * p for wk, p in zip(w, preds))

    def learn(self, a, y):
        """Return prequential joint evidence; the candidate updates share the same rules."""
        ys = np.asarray(y, float)[self.chans]
        lp = np.empty(len(self.hs))
        for i, (p, h) in enumerate(zip(self._prior, self.hs)):
            if isinstance(h, Window):
                lp[i] = self._lp(p, h, ys)
                h.update(a, ys)
            else:
                lp[i] = h.score_learn(a, ys, p)
        w = self.logw - self.logw.max()
        mix = np.log(np.exp(w).sum())
        out = float(np.log(np.exp(w + lp).sum()) - mix)
        self.logw = self.logw + lp
        self.logw -= self.logw.max()
        return out


class Conj:
    """Conjunctive explanation of one channel j (D-13): j is emitted by the JOINT hidden state
    of its own group and of one other group (factorial HMM with a shared emission, as
    `World.links`). Prediction  sum_{s,u} b_own'(s) b_other'(u) E[s, u]  with the predicted
    beliefs of the two groups' most probable HMMs (mean-field: the two beliefs are taken as
    independent) and Beta(.5, .5) counts E filled with the posterior responsibilities.
    `adv` = log P(j's data | conj) - log P(j's data | group model) since creation: the log Bayes
    factor of the conjunctive parent set against the group alone."""

    def __init__(self, j, own, other, t):
        self.j, self.own, self.other, self.t0 = j, own, other, t
        self.ho, self.hg = self._best(own), self._best(other)
        self.n1 = np.zeros((self.ho.K, self.hg.K))
        self.n = np.zeros((self.ho.K, self.hg.K))
        self.adv, self.p = 0., .5

    @staticmethod
    def _best(g):
        idx = [i for i, h in enumerate(g.hs) if not isinstance(h, Window)]
        return g.hs[max(idx, key=lambda i: g.logw[i])]

    def predict(self, a):
        if self.ho.b is None or self.hg.b is None:
            self.bo = self.bg = None
            self.p = .5
            return self.p
        self.bo, self.bg = self.ho.b @ self.ho.T(a), self.hg.b @ self.hg.T(a)
        self.E = (self.n1 + .5) / (self.n + 1.)
        self.p = float(self.bo @ self.E @ self.bg)
        return self.p

    def learn(self, yj, p_base):
        p = min(1. - 1e-6, max(1e-6, self.p))
        q = min(1. - 1e-6, max(1e-6, p_base))
        self.adv += (np.log(p) - np.log(q)) if yj else (np.log(1. - p) - np.log(1. - q))
        if self.bo is None:
            return
        r = np.outer(self.bo, self.bg) * (self.E if yj else 1. - self.E)
        r /= r.sum()
        self.n += r
        if yj:
            self.n1 += r


class Decomposer:
    def __init__(self, n, A, v, seed=0, windows=False, fast=False, conj=False):
        """conj: add conjunctive candidates per channel (D-13). fast: False | True (null key = j's own last OWN values, D-9) | "resid" (null key =
        bin of j's current group predictive, D-10)."""
        self.n, self.A, self.seed, self.win, self.fast = n, A, seed, windows, fast
        self.lastp = None
        self.conj = conj
        self.cj = {}                  # channel -> list of Conj
        self.cj_cool = {}             # (channel, other group) -> step until which not retried
        self.chans = [i for i in range(n) if i != v]
        self.groups = [GroupModel([c], A, seed + 13 * c, ks=(2, 3), windows=windows)
                       for c in self.chans]
        self.cand = []            # dicts: g1, g2 (channel tuples), merged, sep1, sep2, d, t
        self.cool = {}
        self.t = 0
        m = len(self.chans)
        self.discovery = None if fast or conj else NeighborBank(m, 2 * GROUP_WIDTH, seed + 113, False)
        if self.discovery is None:
            # Historical witnesses with dense conjunctive/fast experiments.
            self.s1 = np.zeros(m)
            self.s11 = np.zeros((LAGS + 1, m, m))
        else:
            shape = (LAGS + 1, m, self.discovery.width)
            self.pair_s = np.zeros((4, *shape))  # binary n, source sum, target sum, cross
        self.cnt = 0
        self.proposal_count = 0
        self.tested_evidence = {}
        self.hist = []
        self.merges = []                      # (t, channels) log
        if fast:                              # D-9: exact pairwise sequential evidence
            L = LAGS + 1
            K0 = QBINS if fast == "resid" else 2 ** OWN
            self.f1 = np.zeros((m, m, L, K0, 2, 2))   # i, j, lag, own key of j, x_i(t+1-lag), x_j(t+1)
            self.f0 = np.zeros((m, K0, 2))            # j, own key of j, x_j(t+1)
            self.own = np.zeros(m, int)               # j's last OWN values as an integer
            self.ll1 = np.zeros((m, m, L))
            self.ll0 = np.zeros(m)
            self.fh = []                              # last LAGS observations (channel order)

    # ------------------------------------------------------------------ interface
    def predict(self, x, a):
        out = np.full(self.n, .5)
        for g in self.groups:
            out[g.chans] = g.predict(x, a)
        for c in self.cand:
            for k in ("merged", "sep1", "sep2"):
                c[k].predict(x, a, read=False)
        self.base = out.copy()
        if self.conj:
            for j, lst in self.cj.items():
                if lst:
                    lw = np.array([0.] + [c.adv - CONJ_PEN for c in lst])
                    w = np.exp(lw - lw.max())
                    w /= w.sum()
                    out[j] = w[0] * out[j] + sum(wk * c.predict(a) for wk, c in zip(w[1:], lst))
        self.lastp = out
        return out

    def learn(self, x, a, y):
        self.t += 1
        scores = {tuple(g.chans): g.learn(a, y) for g in self.groups}
        done = []
        for c in self.cand:
            jm = c["merged"].learn(a, y)
            js = c["sep1"].learn(a, y) + c["sep2"].learn(a, y)
            if self.discovery is None:
                c["d"] += jm - js
                age = self.t - c["t"]
                if age >= W or (age >= MIN_AGE and c["d"] < -REJECT):
                    done.append(c)
            else:
                c["joint"] += jm
                c["fresh"] += js
                c["warm"] += scores[c["g1"]] + scores[c["g2"]]
                c["d"] = c["joint"] - np.logaddexp(c["fresh"], c["warm"]) + np.log(2.)
                if abs(c["d"]) > c["bar"]:
                    done.append(c)
        for c in done:
            self.cand.remove(c)
            self._decide(c)
        if self.conj:
            yb = np.asarray(y, float) > .5
            for j, lst in self.cj.items():
                for c in lst:
                    c.learn(bool(yb[j]), self.base[j])
        self._stats(y)
        if self.discovery is not None and not self.discovery.complete and self.t % len(self.chans) == 0:
            rr, cc = self.discovery.rotate()
            self.pair_s[:, :, rr, cc] = 0.
        if self.conj and self.t % 50 == 0 and self.t >= 200:
            self._conj_manage()
        if self.fast:
            self._fast(x, y)
        if self.discovery is not None:
            # Work-scheduling cadence from the modeled lag horizon, not an
            # onset or acceptance age. Decisions are scored on every transition.
            if self.t % (LAGS + 1) == 0:
                self._propose(y)
        elif self.t % 50 == 0 and self.t >= 200:
            self._propose(y)

    # ------------------------------------------------------------------ structure
    def _stats(self, y):
        z = np.asarray(y, float)[self.chans]
        if self.discovery is None:
            self.s1 += z
            self.s11[0] += np.outer(z, z)
            for lag, past in enumerate(reversed(self.hist), 1):
                self.s11[lag] += np.outer(past, z)
        elif self.discovery.width:
            src = self.discovery.source
            yy = (z > .5)[:, None]
            samples = [z, *reversed(self.hist)]
            xx = (np.asarray(samples) > .5)[:, src]
            n = len(samples)
            self.pair_s[0, :n] += 1.
            self.pair_s[1, :n] += xx
            self.pair_s[2, :n] += yy
            self.pair_s[3, :n] += xx * yy
        self.hist = (self.hist + [z])[-LAGS:]
        self.cnt += 1

    def _sparse_pairs(self, busy):
        owners = {ch: g for g in self.groups for ch in g.chans}
        order = {g: i for i, g in enumerate(self.groups)}
        groups = {g: tuple(g.chans) for g in self.groups}
        scores = binary_dependency_scores(self.pair_s)
        # One null vs degree × lags possible parent choices per target.
        code = np.log(max(1, self.discovery.width * (LAGS + 1)))
        valid = {g for g in self.groups if not busy.intersection(g.chans)}
        best = {}
        for j, sources in enumerate(self.discovery.source):
            dest = owners[self.chans[j]]
            if dest not in valid:
                continue
            for i, p in enumerate(sources):
                orig = owners[self.chans[p]]
                if orig is dest or orig not in valid or len(orig.chans) + len(dest.chans) > GROUP_WIDTH:
                    continue
                left, right = (orig, dest) if order[orig] < order[dest] else (dest, orig)
                key = groups[left], groups[right]
                prior = self.tested_evidence.get(key, 0.)
                score = scores[j, i]
                if score > code + prior:
                    best[key] = max(best.get(key, -np.inf), score)
        return best

    def _fast(self, x, y):
        """Update the pairwise KT evidence with the transition x -> y and merge the groups of
        any pair whose log Bayes factor exceeds C_FAST."""
        prev = (np.asarray(x, float)[self.chans] > .5).astype(int)
        nxt = (np.asarray(y, float)[self.chans] > .5).astype(int)
        m, L = len(self.chans), LAGS + 1
        # source values per lag: lag 0 = x_i(t+1), lag l = x_i(t+1-l)
        hist = [nxt, prev] + self.fh[::-1]
        src = np.array([hist[l] if l < len(hist) else np.zeros(m, int) for l in range(L)])
        ok = np.array([l < len(hist) for l in range(L)])
        J = np.arange(m)
        if self.fast == "resid":
            q = np.asarray(self.lastp, float)[self.chans]
            own = np.minimum((q * QBINS).astype(int), QBINS - 1)
        else:
            own = self.own = ((self.own << 1) | prev) & (2 ** OWN - 1)
        c0 = self.f0[J, own]                                     # (m, 2)
        p0 = (c0[J, nxt] + .5) / (c0.sum(1) + 1.)
        self.ll0 += np.log(p0)
        self.f0[J, own, nxt] += 1.
        I, Jm, Lg = np.meshgrid(np.arange(m), J, np.arange(L), indexing="ij")
        xs = src[Lg, I]                                           # x_i at lag
        cell = self.f1[I, Jm, Lg, own[Jm], xs]                     # (m, m, L, 2)
        p1 = (np.take_along_axis(cell, nxt[Jm][..., None], -1)[..., 0] + .5) / (cell.sum(-1) + 1.)
        self.ll1 += np.where(ok[None, None, :], np.log(p1), np.log(p0)[None, :, None])
        np.add.at(self.f1, (I, Jm, Lg, own[Jm], xs, nxt[Jm]), ok[None, None, :].astype(float)
                  * np.ones_like(I, float))
        self.fh = (self.fh + [prev])[-LAGS:]
        if self.t < 10:
            return
        a = self.ll1 - self.ll1.max(-1, keepdims=True)
        bf = self.ll1.max(-1) + np.log(np.exp(a).mean(-1)) - self.ll0[None, :]
        np.fill_diagonal(bf, -np.inf)
        while True:
            gid = {c: k for k, g in enumerate(self.groups) for c in g.chans}
            ch = self.chans
            best, pair = C_FAST, None
            for i, j in zip(*np.nonzero(bf > C_FAST)):
                if gid[ch[i]] != gid[ch[j]] and bf[i, j] > best:
                    best, pair = bf[i, j], (gid[ch[i]], gid[ch[j]])
            if pair is None:
                return
            g1, g2 = self.groups[pair[0]], self.groups[pair[1]]
            merged = GroupModel(sorted(g1.chans + g2.chans), self.A, self.seed + 5000 + self.t,
                                windows=self.win)
            merged.start(y)
            self.groups = [g for g in self.groups if g is not g1 and g is not g2] + [merged]
            self.merges.append((self.t, tuple(merged.chans), "fast"))
            self.cand = [c for c in self.cand
                         if not (set(c["g1"] + c["g2"]) & set(merged.chans))]

    def _corr(self):
        n = self.cnt
        mu = self.s1 / n
        var = np.maximum(mu * (1. - mu), 1e-6)
        sd = np.sqrt(var)
        S = np.outer(sd, sd)
        best = np.abs(self.s11[0] / n - np.outer(mu, mu)) / S
        for lag in range(1, LAGS + 1):
            c = np.abs(self.s11[lag] / max(1, n - lag) - np.outer(mu, mu)) / S
            best = np.maximum(best, np.maximum(c, c.T))
        return best

    def _propose(self, y):
        if len(self.cand) >= M:
            return
        busy = {ch for c in self.cand for ch in c["g1"] + c["g2"]}
        if self.discovery is not None:
            options = self._sparse_pairs(busy)
        else:
            C = self._corr()
            idx = {c: i for i, c in enumerate(self.chans)}
            G = [tuple(g.chans) for g in self.groups]
            options = {}
            for i, g1 in enumerate(G):
                for g2 in G[i + 1:]:
                    key = g1, g2
                    if not busy.intersection(g1 + g2) and self.cool.get(key, -1) <= self.t:
                        options[key] = max(C[idx[a], idx[b]] for a in g1 for b in g2)
        # M is fixed. Linear selection beats sorting all sparse proposals.
        while options and len(self.cand) < M:
            g1, g2 = max(options, key=lambda k: (options[k], k))
            proposal_score = options[(g1, g2)]
            options = {k: s for k, s in options.items() if not set(g1 + g2).intersection(k[0] + k[1])}
            sd = self.seed + 1000 + self.t
            c = dict(g1=g1, g2=g2, t=self.t, d=0.,
                     merged=GroupModel(sorted(g1 + g2), self.A, sd, windows=self.win),
                     sep1=GroupModel(g1, self.A, sd + 1, ks=self._ks(g1), windows=self.win),
                     sep2=GroupModel(g2, self.A, sd + 2, ks=self._ks(g2), windows=self.win))
            for k in ("merged", "sep1", "sep2"):
                c[k].start(y)               # belief from the latest observation
            if self.discovery is not None:
                self.proposal_count += 1
                # Structural prior charges locating the edge and its latent
                # dimension. Repeated looks get a summable prior over trials.
                k = self.proposal_count
                edge = np.log1p(self.discovery.width * (LAGS + 1))
                bar = edge + np.log(k) + np.log1p(k) + np.log(len(KS))
                c.update(bar=bar, joint=0., fresh=0., warm=0., tested=proposal_score)
            self.cand.append(c)
            busy |= set(g1) | set(g2)

    def _conj_manage(self):
        """Per channel keep at most C_CONJ conjunctive candidates: drop those whose groups
        were merged away, whose evidence falls below -CONJ_REJECT, or that still lose to the
        group model after CONJ_AGE steps; refill with the
        other groups in order of their largest |correlation| with j (XOR-like links have none,
        so the order only ranks — every group is tried in turn, a dropped pair waits
        CONJ_COOL steps)."""
        C = self._corr()
        idx = {c: i for i, c in enumerate(self.chans)}
        gof = {c: g for g in self.groups for c in g.chans}
        for j in self.chans:
            own = gof[j]
            keep = []
            for c in self.cj.get(j, []):
                alive = c.own is own and any(c.other is g for g in self.groups)
                if not alive:
                    continue                  # structure changed: no evidence against the pair
                if c.adv > -CONJ_REJECT and not (self.t - c.t0 >= CONJ_AGE and c.adv < 0.):
                    keep.append(c)
                else:
                    self.cj_cool[(j, tuple(c.other.chans))] = self.t + CONJ_COOL
            used = {tuple(c.other.chans) for c in keep}
            others = [g for g in self.groups if g is not own and tuple(g.chans) not in used
                      and self.cj_cool.get((j, tuple(g.chans)), -1) <= self.t and g.started]
            others.sort(key=lambda g: -max(C[idx[j], idx[c]] for c in g.chans))
            for g in others[:C_CONJ - len(keep)]:
                keep.append(Conj(j, own, g, self.t))
            self.cj[j] = keep

    @staticmethod
    def _ks(g):
        return (2, 3) if len(g) == 1 else KS

    def _decide(self, c):
        G = [tuple(g.chans) for g in self.groups]
        if c["g1"] not in G or c["g2"] not in G:
            return
        if c["d"] > c.get("bar", LOG_BF):
            self.groups = [g for g in self.groups if tuple(g.chans) not in (c["g1"], c["g2"])]
            self.groups.append(c["merged"])
            self.merges.append((self.t, tuple(sorted(c["g1"] + c["g2"])), "slow"))
        else:
            key = c["g1"], c["g2"]
            if self.discovery is None:
                self.cool[key] = self.t + COOL
            else:
                if len(self.tested_evidence) >= max(1, len(self.chans) * self.discovery.width):
                    self.tested_evidence.pop(next(iter(self.tested_evidence)))
                self.tested_evidence[key] = c["tested"]

    def peek(self, x, a, ch):
        """Probe: P(channel ch on at the step after a hypothetical observation x)."""
        g = next(g for g in self.groups if ch in g.chans)
        return float(g.peek(x, a)[g.chans.index(ch)])

    def peek_all(self, x, a):
        """Predictive marginals of every channel after a hypothetical observation x (not
        learned); the value channel stays .5."""
        out = np.full(self.n, .5)
        for g in self.groups:
            if g.started:
                out[g.chans] = g.peek(x, a)
        return out

    def partition(self):
        return sorted(tuple(g.chans) for g in self.groups)
