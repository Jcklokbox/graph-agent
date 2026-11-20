"""v13 (slice 2 prototype): v12 with one inference over contexts.

Replaces v11/v12's global change detector, regime store and reinstatement
(research/decisions.md, S3-0). Contexts are whole models (every learned table).

* Birth by prequential evidence (Page CUSUM). A challenger "a new context
  began at onset" is always open; its onset restarts whenever its evidence
  falls to 0. For a
  cell it predicts with the current estimate as a prior of m = prior_strength
  pseudo-counts, updated only by the data since onset; on channels where the
  contexts agree it predicts like the current model, so evidence accrues only
  where the data since onset systematically disagree. When the posterior
  P(change) = sigmoid(logit h + S) exceeds 1/2 the
  current model is archived and the new context starts as a copy in which the
  cells with evidence of change (and the factor / coarse cells of the same
  action and target) are tempered to <= m pseudo-counts (means kept =
  generalisation), plus the challenger's data; cells on which the known
  contexts disagree (split Bayes factor > 0) start from their usage-weighted
  mean with m pseudo-counts; structure that is the same everywhere keeps its
  confidence.
* Belief over contexts with a sticky prior: stay with 1 - h, jump with h to a
  context in proportion to the time spent in it (h = global_hazard); updated
  each step by the exact-cell likelihood of every context (outlier-robust with
  eps = global_eps). The most probable context is the active model.
* predict_next is the belief-weighted mixture over contexts while the belief
  is not degenerate (max < MIX_BELOW).
"""
from __future__ import annotations

from math import exp, log, pi, sqrt

from agents.v11 import split_log_bf
from agents.v12 import PosteriorGraph12

MIX_BELOW, MAX_CONTEXTS = .99, 5
VAR_MIN = .05 ** 2


class PosteriorGraph13(PosteriorGraph12):
    CELL_HAZARD = None                    # None: keep v12's per-cell change points

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.CELL_HAZARD is not None:
            self.change_hazard = self.CELL_HAZARD
        self._ctx_models = [self.m]       # all contexts; self.m is one of them
        self._belief = [1.]
        self._usage = [1.]
        self._ch = None                   # challenger: dict(cells, S, age)
        self.births = 0
        self.context_switches = 0

    # ------------------------------------------------------------ helpers
    def _cur(self) -> int:
        return next(i for i, M in enumerate(self._ctx_models) if M is self.m)

    @staticmethod
    def _cell_density(cell, y):
        """Exact-cell predictive; continuous cells with a variance floor (observation
        resolution VAR_MIN), else a constant channel's density grows without bound with
        the amount of data and the context with more data wins every step."""
        if cell is None or cell.n < 1e-9:
            return .5 if y in (0., 1.) else 1.
        if cell.kind == "binary" and y in (0., 1.):
            return max(1e-12, cell.density(y))
        var = max(VAR_MIN, (cell.m2 + 1. / 12.) / (cell.n + 1.) * (1. + 1. / (cell.n + 1.)))
        return max(1e-12, exp(-.5 * (y - cell.mean) ** 2 / var) / sqrt(2. * pi * var))

    def _context_logliks(self, key, action, y, mask):
        """Per-context log likelihood of one transition, only on channels whose exact cells
        every context has seen and the contexts disagree on (split Bayes factor > 0):
        structure shared by all contexts
        is one pooled cell and carries no evidence about which context is active (without
        this the context with more data wins every predictable step)."""
        e = self.global_eps
        out = [0.] * len(self._ctx_models)
        for t, yt in enumerate(y):
            if not mask[t]:
                continue
            cells = [M.ctx.get((key, action, t)) for M in self._ctx_models]
            if any(c is None or c.n < 1. for c in cells):
                continue        # no basis for comparison: a context that never saw this
                                # situation would generalise to it like the others
            if yt in (0., 1.) and all(c.kind == "binary" for c in cells) and \
                    split_log_bf([(c.k, c.n) for c in cells]) <= 0.:
                continue
            for i, c in enumerate(cells):
                d = self._cell_density(c, yt)
                out[i] += log((1. - e) * d + e * (.5 if yt in (0., 1.) else 1.))
        return out

    def _loglik(self, M, key, action, y, mask) -> float:
        e = self.global_eps
        ll = 0.
        for t, yt in enumerate(y):
            if mask[t]:
                d = self._cell_density(M.ctx.get((key, action, t)), yt)
                ll += log((1. - e) * d + e * (.5 if yt in (0., 1.) else 1.))
        return ll

    @staticmethod
    def _snapshot(cell):
        """Pre-onset state of one cell: (n, total, mean, predictive variance)."""
        if cell is None or cell.n < 1e-9:
            return 0., 0., .5, 1. / 12.
        mu0 = cell.p if cell.kind == "binary" else cell.mean
        var = (cell.m2 + 1. / 12.) / (cell.n + 1.) * (1. + 1. / (cell.n + 1.))
        return cell.n, cell.total, mu0, var

    def _dens(self, mu, var, y):
        if y in (0., 1.):
            return max(1e-12, mu if y == 1. else 1. - mu)
        var = max(VAR_MIN, var)
        return max(1e-12, exp(-.5 * (y - mu) ** 2 / var) / sqrt(2. * pi * var))

    def _ch_lr(self, st, y):
        """log p(y | change at onset) - log p(y | no change) for one cell, both with the
        pre-onset estimate mu0 as prior centre of m pseudo-counts (same prior family):
            no change: (s_pre + s + m mu0) / (n_pre + n + m)
            change:    (s + m mu0) / (n + m)"""
        n, s, n0, s0, mu0, var = st
        m = self.prior_strength
        p1 = (s + m * mu0) / (n + m)
        p0 = (s0 + s + m * mu0) / (n0 + n + m)
        return log(self._dens(p1, var, y)) - log(self._dens(p0, var, y))

    # ------------------------------------------------------------ the inference
    def _global_change_step(self, active, action: int, y, mask) -> None:
        key = tuple(active)
        self.last_global_change = 0.
        self._belief_step(key, action, y, mask)
        self._challenger_step(key, action, y, mask)
        cur = self._cur()
        self._usage[cur] += 1.

    def _belief_step(self, key, action, y, mask) -> None:
        K = len(self._ctx_models)
        if K == 1:
            return
        h = self.global_hazard
        tot = sum(self._usage)
        prior = [(1. - h) * b + h * u / tot for b, u in zip(self._belief, self._usage)]
        ll = self._context_logliks(key, action, y, mask)
        top = max(ll)
        post = [p * exp(l - top) for p, l in zip(prior, ll)]
        z = sum(post)
        self._belief = [p / z for p in post]
        best = max(range(K), key=lambda i: self._belief[i])
        if self._ctx_models[best] is not self.m:
            self.m = self._ctx_models[best]
            self._regimes = [M for M in self._ctx_models if M is not self.m]
            self._invalidate()
            self._ch = None
            self.context_switches += 1
            self.regime_switches += 1

    def _challenger_step(self, key, action, y, mask) -> None:
        M = self.m
        ch = self._ch
        if ch is None:
            ch = self._ch = dict(cells={}, S=0., age=0)
        lr = 0.
        for t, yt in enumerate(y):
            if not mask[t]:
                continue
            ck = (key, action, t)
            cell = M.ctx.get(ck)
            st = ch["cells"].get(ck)
            if st is None:
                st = (0., 0.) + self._snapshot(cell) + (0.,)
            d = self._ch_lr(st[:6], yt)
            lr += d
            ch["cells"][ck] = (st[0] + 1., st[1] + yt) + st[2:6] + (st[6] + d,)
        ch["S"] += lr
        ch["age"] += 1
        h = self.global_hazard
        if ch["S"] >= log((1. - h) / h):                 # posterior P(change) > 1/2
            self._birth(ch)
        elif ch["S"] <= 0.:                              # Page CUSUM: restart the onset
            self._ch = None

    def _birth(self, ch) -> None:
        m = self.prior_strength
        new = self.m.copy()
        changed = {ck for ck, st in ch["cells"].items() if st[6] > 0.}
        # every table entry of a changed (context, action, target) restarts from its mean
        # with m pseudo-counts; cells without evidence of change keep their confidence
        tk = {(ck[0], ck[2]) for ck in changed}            # all actions, pooled, windows
        others = [(M, u) for M, u in zip(self._ctx_models, self._usage) if M is not self.m]
        for key, c in new.ctx.items():
            if (key[0], key[2]) in tk:
                if c.n > m:
                    c.decay(m / c.n)
                continue
            if not others or c.kind != "binary":
                continue
            # structure on which the known contexts disagree: the new context starts from
            # their usage-weighted mean with m pseudo-counts (hierarchical prior)
            cells = [(c, self._usage[self._cur()])] + [(M.ctx.get(key), u) for M, u in others]
            cells = [(x, u) for x, u in cells if x is not None and x.kind == "binary" and x.n > 0.]
            if len(cells) > 1 and split_log_bf([(x.k, x.n) for x, _ in cells]) > 0.:
                w = sum(u for _, u in cells)
                p = sum(u * x.p for x, u in cells) / w
                c.n, c.k = m, m * p
        targets = {ck[2] for ck in changed}
        for table in (new.fac, new.coarse):
            for key, c in table.items():
                if key[2] in targets and c.n > m:
                    c.decay(m / c.n)
        for ck, (n, s, *_rest) in ch["cells"].items():
            cell = new.ctx.get(ck)
            if ck in changed and cell is not None and cell.kind == "binary":
                if s > 0.:
                    cell.update(1., s)
                if n - s > 0.:
                    cell.update(0., n - s)
        self._ctx_models.append(new)
        self._usage.append(1.)
        self._belief = [.0] * (len(self._ctx_models) - 1) + [1.]
        if len(self._ctx_models) > MAX_CONTEXTS:
            i = min(range(len(self._ctx_models) - 1), key=lambda j: self._usage[j])
            for lst in (self._ctx_models, self._usage, self._belief):
                lst.pop(i)
        self._regimes = [M for M in self._ctx_models if M is not new]
        self.m = new
        self._invalidate()
        self._ch = None
        self.births += 1

    # ------------------------------------------------------------ predictions
    def predict_next(self, obs, action: int, *, depth: int = 0):
        if len(self._ctx_models) == 1 or max(self._belief) >= MIX_BELOW:
            return super().predict_next(obs, action, depth=depth)
        keep = self.m
        out = [0.] * self.n_inputs
        try:
            for M, b in zip(self._ctx_models, self._belief):
                if b < 1e-3:
                    continue
                self.m = M
                p = super().predict_next(obs, action, depth=depth)
                out = [o + b * q for o, q in zip(out, p)]
        finally:
            self.m = keep
            self._invalidate()
        z = sum(b for b in self._belief if b >= 1e-3)
        return tuple(o / z for o in out)


Agent = PosteriorGraph13
