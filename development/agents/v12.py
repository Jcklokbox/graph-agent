"""v12: v11 with a calibrated predictive core and value-of-information exploration.

Step 2 (research/decisions.md, S2-0 ... S2-5). Two changes to v11, everything
else (factor level, timing lift, windows, attention, change points, regimes,
planner, event bonus) is v11 unchanged.

1. Predictive core. v11 shrank each context level towards the one above and
   so counted the same observations at every level (two reinforced trials of a
   novel cue -> .94). v12 mixes the levels exactly along the current path
   (context-tree weighting, models/context_tree.py):

       path   = root expert (v11's factor prior incl. the model-based value)
                -> attended context -> exact context -> exact + open windows
       p_w(s) = beta(s) p_e(s) + (1 - beta(s)) p_w(child),  beta = sigmoid(eta(s))
       eta(s) += log p_e(s)(y) - log p_w(child)(y)

   Each node's own estimator p_e generalises from its parent without double
   counting: a Bayesian mixture of the independent estimator (KT) and a Beta
   prior with m = prior_strength pseudo-counts centred at the parent's estimate
   *without the node's own data*; the weight of the two (zeta) is learned per
   level and target. The action-pooled path (controllability) is mixed the same
   way. eta and zeta are part of the regime model and discounted with the cells.

2. Exploration. The calibrated value sd does not vanish in fragmented contexts
   (they hold few data), so v11's bonus explore * sd kept exploring solved
   tasks. The bonus is the value of the information instead: the expected
   improvement of the decision from resolving the action's value,
       d = mu_a - max_{b != a} mu_b,  VOI = sd phi(d/sd) - |d| Phi(-|d|/sd),
   combined with the event bonus as in v11 (weight `explore`).

Known quirk kept for bit-exactness with the screened prototype: the level key
of the pooled window node is its window set, not its target (S2-5 note).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, pi, sqrt

from agents.v11 import Cell, Model, PosteriorGraph, apply_lift

KT = .5


@dataclass
class Model12(Model):
    eta: dict = field(default_factory=dict)      # node id -> log odds "leaf vs refine";
                                                 # ("zeta", level, key) -> log odds "pooled vs KT"

    def copy(self) -> "Model12":
        base = Model.copy(Model(**{k: v for k, v in vars(self).items() if k != "eta"}))
        out = Model12(**vars(base))
        out.eta = dict(self.eta)
        return out


def _dens(p: float, cell: Cell | None, y: float, binary: bool) -> float:
    """Sequential predictive density of y for a node with mean p."""
    if binary:
        return p if y >= .5 else 1. - p
    if cell is None or cell.n <= 0. or cell.kind != "continuous":
        var = 1. / 12.
    else:
        var = (cell.m2 + 1. / 12.) / (cell.n + 1.) * (1. + 1. / (cell.n + 1.))
    return exp(-.5 * (y - p) ** 2 / var) / sqrt(2. * pi * var)


def _sig(z: float) -> float:
    return 1. / (1. + exp(-z)) if z > -50. else 0.


def _voi(d: float, sd: float) -> float:
    if sd <= 1e-9:
        return 0.
    z = abs(d) / sd
    return sd * exp(-.5 * z * z) / sqrt(2. * pi) - abs(d) * .5 * (1. - erf(z / sqrt(2.)))


class PosteriorGraph12(PosteriorGraph):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.m = Model12()

    # ------------------------------------------------------------ the path
    def _root(self, x, act, action, target, depth):
        exc, n_min = self._excitation(act, action, target)
        prior = min(1., max(0., exc * (1. - self._inhibition(act, target))))
        sd2 = max(prior * (1. - prior), .25 / (1. + n_min)) / (2. * n_min / (1. + n_min) + 1.)
        if target == self.value_index and x is not None:
            mb = self._model_based_value(x, action, depth, act)
            if mb is not None:
                p_mb, sd_mb, _ = mb
                w1, w2 = 1. / max(sd2, 1e-6), 1. / max(sd_mb ** 2, 1e-6)
                prior = (w1 * prior + w2 * p_mb) / (w1 + w2)
                sd2 = 1. / (w1 + w2)
        return prior, sd2

    def _paths(self, act, action, target, depth):
        """Node ids and cells below the root: the action path and the pooled path."""
        M = self.m
        ak = self._att(act)
        oc = self._open_cues(act, target, depth)
        nodes = []
        if ak and ak != act:
            ck = (ak, action, target)
            nodes.append((("c",) + ck, M.coarse.get(ck)))
        xk = (act, action, target)
        nodes.append((("x",) + xk, M.ctx.get(xk)))
        nodes.append((("m",) + xk + (oc,), M.ctx.get(xk + (oc,))))
        pk = (act, -1, target)
        pooled = [(("px",) + pk, M.ctx.get(pk)), (("pm",) + pk + (oc,), M.ctx.get(pk + (oc,)))]
        return nodes, pooled

    def _node_parts(self, root_p, nodes):
        """Per node: (p_pooled, p_KT, level key); prior centre = parent without the child."""
        m = self.prior_strength
        cells = [c for _, c in nodes]
        out, q = [], root_p
        for i, (nid, c) in enumerate(nodes):
            k = c.total if (c is not None and c.n > 0) else 0.
            n = c.n if c is not None else 0.
            p_pool = (k + m * q) / (n + m)
            p_kt = (k + KT) / (n + 2. * KT) if (c is None or c.kind != "continuous") \
                else (k + .5) / (n + 1.)
            out.append((p_pool, p_kt, (nid[0], nid[-2] if nid[0] == "m" else nid[-1])))
            nxt = cells[i + 1] if i + 1 < len(cells) else None
            if nxt is not None and nxt.n > 0:
                q = (max(0., k - nxt.total) + m * q) / (max(0., n - nxt.n) + m)
            else:
                q = p_pool
        return out

    def _node_means(self, root_p, nodes):
        eta = self.m.eta
        return [_sig(eta.get(("zeta",) + lk, 0.)) * pp + (1. - _sig(eta.get(("zeta",) + lk, 0.))) * pk
                for pp, pk, lk in self._node_parts(root_p, nodes)]

    def _mix(self, root_p, root_v, nodes, root_id):
        """Mixture over depths: (p_w per level from the root, node means, variance, ids)."""
        eta = self.m.eta
        ps = [root_p] + self._node_means(root_p, nodes)
        D = len(ps) - 1
        pw = [0.] * (D + 1)
        pw[D] = ps[D]
        betas = [1.] * (D + 1)
        ids = [None] + [nid for nid, _ in nodes]
        for d in range(D - 1, -1, -1):
            b = _sig(eta.get(ids[d] if d > 0 else root_id, 0.))
            betas[d] = b
            pw[d] = b * ps[d] + (1. - b) * pw[d + 1]
        w, rest = [], 1.
        for d in range(D + 1):
            w.append(rest * betas[d])
            rest *= 1. - betas[d]
        m = self.prior_strength
        vs = [root_v] + [p * (1. - p) / ((c.n if c is not None else 0.) + m + 1.)
                         for p, (_, c) in zip(ps[1:], nodes)]
        mean = pw[0]
        var = sum(w[d] * (vs[d] + (ps[d] - mean) ** 2) for d in range(D + 1))
        return pw, ps, var, ids

    # ------------------------------------------------------------ predict
    def _predict_uncached(self, x, action: int, target: int, depth: int,
                          act: tuple[int, ...]) -> tuple[float, float]:
        root_p, root_v = self._root(x, act, action, target, depth)
        nodes, pooled = self._paths(act, action, target, depth)
        pw, _, var, _ = self._mix(root_p, root_v, nodes, ("r", action, target))
        p, sd = pw[0], sqrt(max(0., var))
        mem = nodes[-1][1]
        n_obs = mem.n if mem is not None else 0.
        ps = self._p_uncontrollable(target)
        if ps > 1e-3:
            pwp, _, varp, _ = self._mix(root_p, root_v, pooled, ("rp", target))
            pm = pooled[-1][1]
            if pm is not None:
                n_obs = max(n_obs, ps * pm.n)
            p = ps * pwp[0] + (1. - ps) * p
            sd = ps * sqrt(max(0., varp)) + (1. - ps) * sd
        h = self._timing_lift(act, action, target, depth)
        m0 = self.prior_strength
        if h != 0. and n_obs > 0:
            h *= m0 / (m0 + n_obs)
        return min(1., max(0., apply_lift(p, h))), sd

    # ------------------------------------------------------------ learn
    def _update_eta(self, x, act, action, target, outcome):
        binary = outcome in (0., 1.)
        root_p, root_v = self._root(x, act, action, target, 0)
        nodes, pooled = self._paths(act, action, target, 0)
        eta = self.m.eta
        for path in (nodes, pooled):                     # concentration (zeta)
            for (pp, pk, lk), (_, c) in zip(self._node_parts(root_p, path), path):
                key = ("zeta",) + lk
                d = log(max(1e-12, _dens(pp, c, outcome, binary))) - \
                    log(max(1e-12, _dens(pk, c, outcome, binary)))
                eta[key] = max(-50., min(50., eta.get(key, 0.) + d))
        root_p, root_v = self._root(x, act, action, target, 0)
        for path, root_id in ((nodes, ("r", action, target)), (pooled, ("rp", target))):
            pw, ps, _, ids = self._mix(root_p, root_v, path, root_id)   # depths (eta)
            cells = [None] + [c for _, c in path]
            for d in range(len(ps) - 1):
                nid = ids[d] if d > 0 else root_id
                le = log(max(1e-12, _dens(ps[d], cells[d], outcome, binary)))
                lc = log(max(1e-12, _dens(pw[d + 1], cells[d + 1], outcome, binary)))
                eta[nid] = max(-50., min(50., eta.get(nid, 0.) + le - lc))

    def _learn_target(self, active, full_key, att_key, action: int, target: int,
                      outcome: float) -> None:
        x = self._pending[0] if self._pending is not None else None
        self._update_eta(x, tuple(active), action, target, outcome)
        super()._learn_target(active, full_key, att_key, action, target, outcome)

    def _discount_all(self, lam: float) -> None:
        super()._discount_all(lam)
        eta = self.m.eta
        for k in eta:
            eta[k] *= lam

    # ------------------------------------------------------------ act
    def _q_parts(self, obs, action):
        v = self.value_index
        immediate, sd_now = self._predict(obs, action, v, 0)
        nxt = self._predict_next(obs, action, 0)
        nxt = nxt[:v] + (min(nxt[v], .5),) + nxt[v + 1:]
        mu = immediate + self.gamma * self._expected_continuation(nxt)
        e = 0.
        if self.event_explore > 0.:
            e = self.event_explore * self._event_bonus(obs, action, 0) * (1. - immediate)
        return mu, sd_now, e

    def _q(self, obs, action):
        """mu_a + sqrt((explore VOI_a)^2 + event bonus^2); all actions evaluated once per step."""
        if action == 0 or getattr(self, "_voi_obs", None) is not obs:
            self._voi_obs = obs
            self._voi_parts = [self._q_parts(obs, a) for a in range(self.n_actions)]
        parts = self._voi_parts
        mu, sd, e = parts[action]
        other = max(p[0] for j, p in enumerate(parts) if j != action)
        bonus = self.explore * _voi(mu - other, sd)
        return mu + sqrt(bonus * bonus + e * e)


Agent = PosteriorGraph12
