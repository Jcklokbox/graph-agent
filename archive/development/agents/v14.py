"""v14 (slice 2 redo): v13 brought closer to the T2 reference model `lc-f-E` (decisions N-2).

Three changes to v13, each a property of the reference observer that passes every
context row of T2 v1 (decisions W-1):

1. Learned hazard. The rate of context changes is not fixed: a posterior over the grid
   H_GRID from the change events seen so far (births and context switches),
       w(h) ~ h^c (1 - h)^(t - c),
   its mean is the hazard of the belief prior and of the birth threshold.
2. Contexts share features, not contingencies. The challenger "a new context began at
   onset" lets every cell either be unchanged (it predicts as the current model) or
   changed (a fresh estimate from the data since onset: KT for binary cells; the unchanged
   hypothesis uses the same KT prior on all data, so a cell first seen after the onset
   carries no evidence), with prior probability RHO of a change per cell:
       S = sum_cells log(RHO exp(D_c) + 1 - RHO),
       D_c = log p(cell data since onset | changed) - log p(... | unchanged).
   Cells that never disagree cost at most log(1 - RHO) and carry no evidence for a birth;
   a cell seen for the first time carries none. The onset restarts (Page) when S <= 0 and no
   cell favours a change (D_c <= 0 for all), so the per-cell costs of an ITI do not erase
   the evidence of the trial before it. At a birth the new context copies the current
   one; cells whose posterior P(changed) > 1/2 restart from the challenger's data alone
   (binary; continuous cells are tempered as in v13); every table entry of a changed
   (context, target) — all actions, pooled, windows — and the factor / coarse cells of the
   targets restart too, so the old contingency is not inherited through generalisation.
   v13's usage-weighted start of disagreeing cells is removed.
3. (unchanged from v13) belief over contexts with a sticky prior, evidence only from cells
   every context has seen and the contexts disagree on; mixture predictions.
"""
from __future__ import annotations

from math import exp, log

from agents.v13 import PosteriorGraph13

H_GRID = (.0003, .001, .003, .01, .03)
RHO = .1                      # prior probability that a cell differs in a new context


def _lse2(a: float, b: float) -> float:
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))


class PosteriorGraph14(PosteriorGraph13):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._hz_c, self._hz_t = 0, 0

    # ------------------------------------------------------------ hazard
    def _h_mean(self) -> float:
        c, t = self._hz_c, self._hz_t
        lw = [c * log(h) + (t - c) * log(1. - h) for h in H_GRID]
        top = max(lw)
        w = [exp(x - top) for x in lw]
        return sum(wi * h for wi, h in zip(w, H_GRID)) / sum(w)

    def _global_change_step(self, active, action: int, y, mask) -> None:
        self.global_hazard = self._h_mean()
        b0, s0 = self.births, self.context_switches
        super()._global_change_step(active, action, y, mask)
        self._hz_t += 1
        self._hz_c += (self.births - b0) + (self.context_switches - s0)

    # ------------------------------------------------------------ challenger
    def _changed_lp(self, st, y):
        """log p(y | cell changed at onset): KT on the data since onset (binary)."""
        n, s, n0, s0, mu0, var = st
        if y in (0., 1.):
            p = (s + .5) / (n + 1.)
            return log(max(1e-12, p if y == 1. else 1. - p))
        m = self.prior_strength
        return log(self._dens((s + m * mu0) / (n + m), var, y))

    def _unchanged_lp(self, st, y):
        """Same prior family as the changed hypothesis (KT for binary), so a cell without
        pre-onset data carries no evidence either way."""
        n, s, n0, s0, mu0, var = st
        if y in (0., 1.):
            p = (s0 + s + .5) / (n0 + n + 1.)
            return log(max(1e-12, p if y == 1. else 1. - p))
        m = self.prior_strength
        return log(self._dens((s0 + s + m * mu0) / (n0 + n + m), var, y))

    def _challenger_step(self, key, action, y, mask) -> None:
        M = self.m
        ch = self._ch
        if ch is None:
            ch = self._ch = dict(cells={}, S=0., age=0)
        lr1, lr0 = log(RHO), log(1. - RHO)
        for t, yt in enumerate(y):
            if not mask[t]:
                continue
            ck = (key, action, t)
            st = ch["cells"].get(ck)
            if st is None:
                st = (0., 0.) + self._snapshot(M.ctx.get(ck)) + (0.,)
            D_old = st[6]
            D = D_old + self._changed_lp(st[:6], yt) - self._unchanged_lp(st[:6], yt)
            ch["S"] += _lse2(lr1 + D, lr0) - _lse2(lr1 + D_old, lr0)
            ch["cells"][ck] = (st[0] + 1., st[1] + yt) + st[2:6] + (D,)
        ch["age"] += 1
        h = self.global_hazard
        if ch["S"] >= log((1. - h) / h):                 # posterior P(change) > 1/2
            self._birth(ch)
        elif ch["S"] <= 0. and not any(st[6] > 0. for st in ch["cells"].values()):
            self._ch = None                              # restart the onset only when no
                                                         # cell favours a change any more

    def _birth(self, ch) -> None:
        m = self.prior_strength
        new = self.m.copy()
        cut = log((1. - RHO) / RHO)                      # P(changed | data) > 1/2
        changed = {ck for ck, st in ch["cells"].items() if st[6] > cut}
        targets = {ck[2] for ck in changed}
        tk = {(ck[0], ck[2]) for ck in changed}          # all actions, pooled, windows
        for key, c in new.ctx.items():
            if (key[0], key[2]) in tk:
                c.decay(0. if c.kind == "binary" else min(1., m / max(c.n, 1e-9)))
        for ck in changed:
            c = new.ctx.get(ck)
            if c is None or c.kind != "binary":
                continue
            n, s = ch["cells"][ck][:2]
            if s > 0.:
                c.update(1., s)
            if n - s > 0.:
                c.update(0., n - s)
        for table in (new.fac, new.coarse):
            for key, c in table.items():
                if key[2] in targets:
                    c.decay(0. if c.kind == "binary" else min(1., m / max(c.n, 1e-9)))
        self._ctx_models.append(new)
        self._usage.append(1.)
        self._belief = [.0] * (len(self._ctx_models) - 1) + [1.]
        if len(self._ctx_models) > 5:
            i = min(range(len(self._ctx_models) - 1), key=lambda j: self._usage[j])
            for lst in (self._ctx_models, self._usage, self._belief):
                lst.pop(i)
        self._regimes = [M for M in self._ctx_models if M is not new]
        self.m = new
        self._invalidate()
        self._ch = None
        self.births += 1


Agent = PosteriorGraph14
