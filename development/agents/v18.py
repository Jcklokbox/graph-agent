"""v18 (slice 3b-1 candidates, D-21): v17 with bounded candidate parents per channel.

Channel j's cause predictor (`LatentCauseSubAdd`, 200 particles) sees a local vector
[x_j(t), action one-hot, x_{s1}(t) .. x_{sk}(t)] with k = 4 slots. Slots are filled by cheap
pairwise screening: for every pair (i, j) a sequential KT evidence that x_i(t) informs x_j(t+1)
beyond j's own state (counts over (x_j(t), x_i(t)) against counts over x_j(t) alone; O(n^2)
counters — the Hebbian part), so the expensive structure average runs only over bounded sets:
cost per channel constant, total linear in channels. A slot is (re)assigned when an outside
channel's evidence exceeds the weakest slot's by SWAP nats (and is > 0); the statistics of the
sets that contain the slot are reset to the prior, and each such set inherits the log evidence
of the same set without the slot (it starts level with its subset and must earn its cells).
"""
from __future__ import annotations

import numpy as np

from agents.v17 import Agent as V17
from agents.v12 import PosteriorGraph12
from models.latent_cause_lin import LatentCauseSubAdd

K_SLOTS, SWAP, PARTICLES = 4, np.log(3.), 200


class Agent(V17):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        PosteriorGraph12.__init__(self, n_inputs, n_actions, value_index, seed=seed, **kw)
        self.chans = [j for j in range(n_inputs) if j != value_index]
        n = len(self.chans)
        self.k = min(K_SLOTS, n - 1)
        m = 1 + n_actions + self.k
        self.lcs = {j: LatentCauseSubAdd(emit=[], cues=list(range(m)), us=-1, hazard=None,
                                         n=PARTICLES, seed=seed + 17 * j) for j in self.chans}
        self.slots = {j: [-1] * self.k for j in self.chans}      # indices into self.chans
        self.f1 = np.zeros((n, n, 2, 2, 2))      # i, j, x_j(t), x_i(t), x_j(t+1)
        self.f0 = np.zeros((n, 2, 2))            # j, x_j(t), x_j(t+1)
        self.ev = np.zeros((n, n))               # log BF "i informs j"
        self.mloss = np.zeros((2, n_inputs))
        self._lmemo = {}

    # ------------------------------------------------------------ local vectors
    def _bits(self, x):
        return np.array([float(x[c]) > .5 for c in self.chans], float)

    def _local(self, b, jj, action):
        v = [b[jj]] + [0.] * self.n_actions + [b[s] if s >= 0 else 0. for s in self.slots[self.chans[jj]]]
        v[1 + action] = 1.
        return v

    def _lc_pred(self, x, action):
        key = (tuple(x), action)
        hit = self._lmemo.get(key)
        if hit is None:
            b = self._bits(x)
            hit = np.full(self.n_inputs, .5)
            for jj, j in enumerate(self.chans):
                hit[j] = self.lcs[j].predict_us(self._local(b, jj, action))
            self._lmemo[key] = hit
        return hit

    # ------------------------------------------------------------ screening
    def _screen(self, b, bn):
        n = len(self.chans)
        xs, xo, xn = b.astype(int), b.astype(int), bn.astype(int)
        J = np.arange(n)
        c0 = self.f0[J, xo]                                         # (n, 2)
        p0 = (c0[J, xn] + .5) / (c0.sum(1) + 1.)
        I, Jm = np.meshgrid(J, J, indexing="ij")
        c1 = self.f1[I, Jm, xo[Jm], xs[I]]                          # (n, n, 2)
        p1 = (np.take_along_axis(c1, xn[Jm][..., None], -1)[..., 0] + .5) / (c1.sum(-1) + 1.)
        self.ev += np.log(p1) - np.log(p0)[None, :]
        self.f0[J, xo, xn] += 1.
        self.f1[I, Jm, xo[Jm], xs[I], xn[Jm]] += 1.

    def _reslot(self):
        for jj, j in enumerate(self.chans):
            sl = self.slots[j]
            e = self.ev[:, jj].copy()
            e[jj] = -np.inf
            for s in sl:
                if s >= 0:
                    e[s] = -np.inf
            best = int(np.argmax(e))
            if not np.isfinite(e[best]) or e[best] <= 0.:
                continue
            vals = [self.ev[s, jj] if s >= 0 else -np.inf for s in sl]
            p = int(np.argmin(vals))
            if sl[p] < 0 or e[best] > vals[p] + SWAP:
                sl[p] = best
                self._reset_slot(self.lcs[j], 1 + self.n_actions + p)

    @staticmethod
    def _reset_slot(lc, pos):
        for i, s in enumerate(lc.sets):
            if pos in s:
                lc.q1[:, :, lc.off[i]:lc.off[i + 1]] = 0.
                lc.q0[:, :, lc.off[i]:lc.off[i + 1]] = 0.
                lc.qp[:, :, lc.off[i]:lc.off[i + 1]] = .5
                lc.ev[:, :, i] = lc.ev[:, :, lc.sets.index(tuple(c for c in s if c != pos))]
        lc._memo = None
        lc._ver += 1
        s0 = 1.
        lc.mu[:, :, pos] = 0.
        lc.S[:, :, pos, :] = 0.
        lc.S[:, :, :, pos] = 0.
        lc.S[:, :, pos, pos] = s0

    # ------------------------------------------------------------ learning
    def learn(self, next_obs, **kw):
        x, action = self._pending
        own = np.asarray(PosteriorGraph12._predict_next(self, x, action, 0))
        pl = self._lc_pred(x, action)
        y = np.asarray([float(v) for v in next_obs]) > .5
        P = np.clip(np.array([own, pl]), EPS, 1. - EPS)
        self.mloss = GAMMA * self.mloss - np.where(y[None, :], np.log(P), np.log(1. - P))
        b, bn = self._bits(x), self._bits(next_obs)
        for jj, j in enumerate(self.chans):
            self.lcs[j].observe(self._local(b, jj, action), bool(y[j]))
        self._screen(b, bn)
        self._reslot()
        PosteriorGraph12.learn(self, next_obs, **kw)


from agents.v17 import EPS, GAMMA  # noqa: E402
