"""AR-6 unified candidate: grounded configural splits, selective anchor retention, and baseline-anchored context optimism."""
from __future__ import annotations

import numpy as np

from agents.v47 import Agent as Controller, CredibleEventReward
from models.factor_context_reward import ContextualOpportunities


class GroundedSignedReward(CredibleEventReward):
    """Configural reward head with empirical-variance splits and grounded context priors."""

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.ctx_sq = np.zeros_like(self.ctx_n)

    def _update_context_structure(self, b):
        before = self.anchors.copy()
        revise = super()._update_context_structure(b)
        if revise:
            self.ctx_sq[:, before != self.anchors] = 0.
        return revise

    def _reset_context_counts(self):
        super()._reset_context_counts()
        self.ctx_sq.fill(0.)

    def _pair_record(self, bn, y, x, phase):
        super()._pair_record(bn, y, x, phase)
        c = int(self.code(bn))
        ix = x.reshape(self.L, self.J)
        on = phase.reshape(self.L, self.J)
        for d, j in enumerate(self.anchors):
            if j < 0:
                continue
            for l in np.flatnonzero(on[:, j]):
                self.ctx_sq[c, d, l, self._j, ix[l].astype(int)] += y * y

    def _pair_record_slots(self, bn, y, x, phase, slots):
        c = int(self.code(bn))
        r = min(self.B - 1, int(y * self.B))
        ix = x.reshape(self.L, self.J)
        on = phase.reshape(self.L, self.J)
        for d in slots:
            j = int(self.anchors[d])
            if j < 0:
                continue
            for l in np.flatnonzero(on[:, j]):
                z = ix[l].astype(int)
                self.ctx_n[c, d, l, self._j, z] += 1.
                self.ctx_s[c, d, l, self._j, z] += y
                self.ctx_sq[c, d, l, self._j, z] += y * y
                self.ctx_atoms[c, d, l, self._j, z, r] += 1.
                self.ctx_amount[c, d, l, self._j, z, r] += y

    def learn_joint(self, b, hist, a, y, bn):
        old = self.slots.copy()
        modes = self.switches
        super(ContextualOpportunities, self).learn_joint(b, hist, a, y, bn)
        before = self.anchors.copy()
        change = self._update_context_structure(b)
        if modes != self.switches or (old != self.slots).any():
            self._reset_context_counts()
            for bx, ax, rx, by, x, phase in self.replay:
                self._pair_record(by, rx, x, phase)
        elif change:
            changed = np.flatnonzero(before != self.anchors)
            if changed.size and self.replay:
                for bx, ax, rx, by, x, phase in self.replay:
                    self._pair_record_slots(by, rx, x, phase, changed)
        elif self.replay:
            _, _, rx, by, x, phase = self.replay[-1]
            self._pair_record(by, rx, x, phase)

    def signed_rules(self, rule):
        """Select signed splits using empirical residual variance without unary-parent blindness."""
        result = []
        baseline = (self.context_x + .5) / (1. + self.context_t)
        possible = np.ones(self.J, bool)
        possible[self.slots[self.slots >= 0]] = False
        for d, j in enumerate(self.anchors):
            if j < 0:
                continue
            rate = (self.pair_on[d] + .5) / (1. + self.anchor_n[d])
            is_pair = possible & (rate > baseline) & (rate < 1. - .5 / (1. + self.anchor_n[d]))
            is_pair[j] = False
            e = np.where(is_pair, self.assoc[d], -np.inf)
            m = int(e.argmax())
            if e[m] <= np.log1p(self.J):
                continue
            for l in range(self.L):
                parent = ((rule[0] == l * self.J + j) & rule[1]).any(1)
                n = self.ctx_n[:, d, l, m]
                s = self.ctx_s[:, d, l, m]
                sq = self.ctx_sq[:, d, l, m]
                N = n.sum(1)
                mu = np.divide(s, n, out=np.zeros_like(s), where=n > 0.)
                gain = np.prod(n, axis=1) / np.maximum(1., N) * np.square(mu[:, 0] - mu[:, 1])
                ss_w = np.maximum(0., sq - np.divide(np.square(s), n, out=np.zeros_like(s), where=n > 0.)).sum(1)
                var = (.25 + ss_w) / (1. + N)
                cost = np.where(parent, np.log1p(N), np.log1p(N) + np.log1p(self.F - 1))
                support = (n > 0).all(1) & (gain > var * cost)
                if support.any():
                    result.append((j, m, l, support, gain, d))
        return result

    def pair_context(self, rule, x, phase, shift=False):
        """Baseline-anchored, empirical-Bernstein optimism for onset configurations."""
        out = np.zeros(self.C)
        f = np.asarray(x).reshape(self.L, self.J)
        on = np.asarray(phase).reshape(self.L, self.J)
        allowed = np.ones(self.J, bool)
        allowed[self.slots[self.slots >= 0]] = False
        base_rate = (self.context_x + .5) / (1. + self.context_t)
        base = rule[5]
        for d, j in enumerate(self.anchors):
            if j < 0:
                continue
            am = (self.pair_on[d] + .5) / (1. + self.anchor_n[d])
            good = allowed & (am > base_rate) & (am < 1. - .5 / (1. + self.anchor_n[d]))
            good[j] = False
            score = np.where(good, self.assoc[d], -np.inf)
            m = int(score.argmax())
            if score[m] <= np.log1p(self.J):
                continue
            rarity = (1. - base_rate[j]) * self.onset_total[j] / (.5 + self.past_on[j])
            for l in np.flatnonzero(on[:, j]):
                if shift and l == 0:
                    continue
                c = int(f[l, m])
                n = self.ctx_n[:, d, l, m, c]
                s = self.ctx_s[:, d, l, m, c]
                sq = self.ctx_sq[:, d, l, m, c]
                total = self.ctx_n[:, d, l, m].sum(1)
                prior = (self.ctx_s[:, d, l, m].sum(1) + base) / (1. + total)
                u = (s + prior) / (1. + n)
                mu_emp = np.divide(s, n, out=np.zeros_like(s), where=n > 0.)
                v_emp = np.maximum(0., np.divide(sq, n, out=np.zeros_like(sq), where=n > 0.) - mu_emp * mu_emp)
                weight = rarity * (am[m] - base_rate[m]) / (
                    max(am[m], 1e-12) if c == 0 else max(1. - base_rate[m], 1e-12)
                )
                u = np.minimum(1., u + weight * (np.sqrt(v_emp / (1. + n)) + (1. - u) / (1. + n)))
                same = ((rule[0] == l * self.J + j) & rule[1]).any(1)
                out = np.maximum(out, u * (same | ~rule[1].any(1)))
        return out


class Agent(Controller):
    reward_cls = GroundedSignedReward

    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed=seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = self.reward_cls(f.J, f.A, f.D)
