"""PosteriorGraph: an online, sparse, hierarchical posterior-predictive agent.

Standard library only. See research/model.md for the mathematical model and
research/results.md for measured behaviour and failures. Summary:

* Data.   Observation x in [0,1]^n (one channel carries the value w) and one of
  A actions. The value channel is re-coded around a reference point. A channel
  is *active* when above 1/2.
* Cells.  Every learned statistic is one conjugate cell (Beta-Bernoulli counts
  for 0/1 data, Gaussian moments otherwise) with Bayesian change-point
  forgetting. There is no gradient learning and no iterative optimisation.
* Relevance.  Structure decisions are Bayes factors between "the rate depends
  on a split" and "one shared rate": attention (does channel i predict channel
  j?), controllability (does the action matter?), timing relevance (does a
  context channel modulate a timed effect?).
* Prediction.  P(y_j | x, a) is a chain of contexts, each cell shrunk toward
  the prediction of the level above:
      factor prior (cue lifts over a baseline, inhibition, reward model)
        -> attended context -> exact context -> exact context + open windows
  mixed with the action-pooled cells by P(uncontrollable), plus the timing
  lift of open expectation windows (cue, age) from hazard cells.
* Change.  Per-cell CUSUM change points; a joint, outlier-robust global change
  point archives the current model as a latent regime, and stored regimes are
  reinstated when they predict surprising transitions better (savings).
* Control.  Two-step Bellman backup with an expectation over the predicted next
  observation; one optimism budget per transition (posterior sd and the bonus
  of an open, untested expectation add as variances).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, lgamma, log, pi, sqrt
from typing import Iterable

# =============================================================== statistics
@dataclass
class Cell:
    """Discounted posterior of one outcome stream.

    Binary data: Beta(1 + k, 1 + n - k). Fractional data: Welford moments with
    one Uniform(0,1) pseudo-observation in the predictive variance. Weights
    allow fractional evidence (salience-weighted timing exposures)."""
    kind: str | None = None
    n: float = 0.
    k: float = 0.
    mean: float = 0.
    m2: float = 0.
    surprise: float = 0.   # CUSUM log evidence for an unabsorbed change

    def update(self, y: float, w: float = 1.) -> None:
        if not 0 <= y <= 1:
            raise ValueError("outcome must be in [0,1]")
        if w <= 0.:
            return
        if self.kind is None:
            self.kind = "binary" if y in (0., 1.) else "continuous"
        elif self.kind == "binary" and y not in (0., 1.):
            self.kind = "continuous"   # never encode a fraction as Bernoulli counts
        self.n += w
        if self.kind == "binary":
            self.k += w * float(y == 1.)
        delta = y - self.mean
        self.mean += w * delta / self.n
        self.m2 += w * delta * (y - self.mean)

    def decay(self, lam: float) -> None:
        self.n *= lam
        self.k *= lam
        self.m2 *= lam

    def affine(self, a: float, b: float) -> None:
        """Re-express stored outcomes y as a*y + b (exact for the moments)."""
        if self.kind is None:
            return
        self.mean = a * self.mean + b
        self.m2 *= a * a
        if self.kind == "binary" and not (a == 1. and b == 0.):
            self.kind = "continuous"

    @property
    def total(self) -> float:
        return self.k if self.kind == "binary" else self.mean * self.n

    @property
    def p(self) -> float:
        """Predictive mean."""
        if self.kind == "binary":
            return (1. + self.k) / (2. + self.n)
        return self.mean if self.n > 0 else .5

    def density(self, y: float) -> float:
        if self.kind == "binary" and y in (0., 1.):
            return self.p if y == 1. else 1. - self.p
        var = (self.m2 + 1. / 12.) / (self.n + 1.) * (1. + 1. / (self.n + 1.))
        return exp(-.5 * (y - self.mean) ** 2 / var) / sqrt(2. * pi * var)

    def change_point(self, y: float, hazard: float) -> None:
        """Bayesian change point before absorbing y: c = sigmoid(logit h + S),
        S the accumulated (CUSUM) log likelihood ratio of a fresh prior
        (1/2 for 0/1 data, uniform density otherwise) against this cell; old
        evidence is down-weighted by 1 - c and the absorbed part is spent."""
        if self.n <= 0 or hazard <= 0.:
            return
        p0 = .5 if (self.kind == "binary" and y in (0., 1.)) else 1.
        llr = log(p0) - log(max(1e-12, self.density(y)))
        s_t = min(50., self.surprise + llr)
        c = 1. / (1. + exp(-(log(hazard / (1. - hazard)) + s_t)))
        self.decay(1. - c)
        self.surprise = max(0., s_t) * (1. - c)

    def change_point_memoryless(self, event: bool, hazard: float) -> None:
        """The same rule without accumulated evidence (count cells)."""
        r = self.n
        if r <= 0 or hazard <= 0.:
            return
        p = (self.k + 1.) / (r + 2.)
        like = p if event else 1. - p
        c = hazard * .5 / (hazard * .5 + (1. - hazard) * like)
        self.k *= 1. - c
        self.n *= 1. - c


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def split_log_bf(groups) -> float:
    """log Bayes factor of 'each group has its own rate' against 'one shared
    rate' for binary counts [(k, n), ...] under uniform priors."""
    lbf, K, N = 0., 0., 0.
    for k, n in groups:
        lbf += log_beta(1. + k, 1. + n - k)
        K += k
        N += n
    return lbf - log_beta(1. + K, 1. + N - K)


def lift(p: float, base: float) -> float:
    """Signed relative effect of a rate p over a baseline: the fraction of the
    remaining headroom gained (p >= base) or of the baseline removed."""
    if p >= base:
        return (p - base) / max(1e-9, 1. - base)
    return (p - base) / max(base, 1e-9)


def apply_lift(p: float, s: float) -> float:
    """Inverse of lift: p with a relative effect s in [-1, 1] applied."""
    if s > 0.:
        return 1. - (1. - p) * (1. - s)
    if s < 0.:
        return p * (1. + s)
    return p


def shrink(cell: Cell, prior: float, m: float) -> tuple[float, float]:
    """Posterior mean of a cell with `m` pseudo-observations at `prior`, and
    the within-cell variance (data moments plus the pseudo-observations at
    Bernoulli spread around the prior)."""
    p = (cell.total + m * prior) / (cell.n + m)
    if cell.kind == "binary":
        within = cell.k * (1. - p) ** 2 + (cell.n - cell.k) * p ** 2
    else:
        within = cell.m2 + cell.n * (cell.mean - p) ** 2
    return p, (within + m * prior * (1. - prior)) / (cell.n + m)


@dataclass
class Window:
    """An open expectation window: `age` steps since an onset of `cue`."""
    cue: int
    target: int
    age: int = 0
    salience: float = 1.   # 1 - predicted probability of the anchoring onset


@dataclass
class Model:
    """Every learned table of one latent regime (archived/restored as a whole)."""
    fac: dict = field(default_factory=dict)          # ((), a, j) root / ((i,), a, j) cue present
    absent: dict = field(default_factory=dict)       # ((i,), a, j) cue absent
    ctx: dict = field(default_factory=dict)          # (S, a, j) and (S, a, j, windows); a = -1 pooled
    coarse: dict = field(default_factory=dict)       # (attended part of S, a, j)
    kids: dict = field(default_factory=dict)         # coarse key -> exact contexts below it
    tau_dev: dict = field(default_factory=dict)      # (coarse key, S) -> (weight, sq. deviation)
    tau_sum: dict = field(default_factory=dict)      # j -> [sum weights, sum deviations]
    reward: dict = field(default_factory=dict)       # active set of y without w -> value cell
    coarse_reward: dict = field(default_factory=dict)
    reward_kids: dict = field(default_factory=dict)
    cancel: dict = field(default_factory=dict)       # (i, j) -> [cell | i present, cell | i absent]
    hazard: dict = field(default_factory=dict)       # (cue, j, lag, a) -> cell
    hazard_index: dict = field(default_factory=dict)  # (cue, j) -> keys
    hazard_ctx: dict = field(default_factory=dict)   # (cue, j, lag, a, S) -> cell
    hctx_index: dict = field(default_factory=dict)   # (cue, j, lag, a) -> keys
    event_root: dict = field(default_factory=dict)   # (j, a) -> cell
    ctrl_lbf: dict = field(default_factory=dict)     # (attended context, j, windows) -> log BF
    ctrl_total: dict = field(default_factory=dict)   # j -> summed log BF

    def copy(self) -> "Model":
        def cp(v):
            if isinstance(v, Cell):
                return Cell(v.kind, v.n, v.k, v.mean, v.m2, v.surprise)
            if isinstance(v, list):
                return [cp(u) for u in v]
            if isinstance(v, set):
                return set(v)
            return v
        return Model(**{name: {k: cp(v) for k, v in table.items()}
                        for name, table in vars(self).items()})

    def cell_tables(self):
        return (self.fac, self.absent, self.ctx, self.coarse, self.reward,
                self.coarse_reward, self.hazard, self.event_root, self.hazard_ctx)


class _LCG:
    def __init__(self, seed: int):
        self.state = int(seed) & 0x7fffffff

    def random(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7fffffff
        return self.state / 0x80000000


# ==================================================================== agent
class PosteriorGraph:
    def __init__(self, n_inputs: int, n_actions: int, value_index: int, *,
                 seed: int = 0, gamma: float = .9, alpha: float = .01,
                 explore: float = 2., event_explore: float = .5, max_age: int = 12,
                 change_hazard: float = .01, global_hazard: float = .001,
                 rollout_branch: int = 4, value_scale: float = .5,
                 tie_seed: int | None = None) -> None:
        if n_inputs < 1 or n_actions < 2 or not 0 <= value_index < n_inputs:
            raise ValueError("invalid interface dimensions")
        if not 0 <= gamma <= 1 or not 0 < alpha < 1:
            raise ValueError("invalid gamma or alpha")
        self.n_inputs, self.n_actions, self.value_index = n_inputs, n_actions, value_index
        self.gamma = gamma
        self.explore, self.event_explore = explore, float(event_explore)
        self.max_age = max_age
        self.change_hazard, self.global_hazard = change_hazard, global_hazard
        self.rollout_branch, self.rollout_eps = int(rollout_branch), .05
        self.value_scale = float(value_scale)
        self.prior_strength = 2.     # pseudo-observations of a prior level
        # inhibition (cancellation contrast) and blocking thresholds
        self.excite_threshold, self.credit_threshold = .65, .5
        self.block_strength, self.block_error = .6, .25
        # global change point and latent regimes
        self.global_eps = .05        # outlier rate: one step adds <= log(1/eps)
        self.regime_store = .25      # archive the model once P(change) exceeds this
        self.regime_decay, self.regime_bf, self.regime_max = .8, 3., 4
        self.tie_rng = _LCG(seed if tie_seed is None else tie_seed)

        self.m = Model()
        self._regimes: list[Model] = []
        self._reg_evidence: list[float] = []
        self.regime_switches = 0
        self.last_global_change = 0.
        self._g_surprise = self._g_applied = self._g_pending = 0.
        # value frame: z = 1/2 + (w - ref) / S
        self._vlo = self._vhi = self._vref = None
        self._vcounts: dict[float, int] = {}
        # attention: Granger dependence tables per (action, lag)
        self.attended: set[int] = set()
        self._dep: dict = {}
        self._dep_hist: list[list[int]] = []
        n_tests = n_inputs * (n_inputs - 1) * n_actions * max_age
        self._dep_threshold = log(max(2, n_tests) / alpha)
        # coarse-context membership of exact contexts (controllability)
        self._all_fk: set = set()
        self._fk_by_ak: dict = {}
        self._ctrl_c_alen = 0
        # expectation windows
        self.windows: list[Window] = []
        self._time = 0
        self._rise: dict[int, int] = {}
        self._runlen: dict[int, float] = {}
        self._onset_salience: dict[int, float] = {}
        self._pending: tuple[tuple[float, ...], int] | None = None
        self._memo: dict = {}
        self._cache: dict = {}

    # ============================================================ value frame
    def _value_frame(self) -> tuple[float, float]:
        """(reference, scale): reference = median experienced value, scale =
        max(2 * largest deviation from it, value_scale)."""
        ref = self._vref
        return ref, max(2. * max(self._vhi - ref, ref - self._vlo), self.value_scale)

    def _median_value(self) -> float:
        cnt = self._vcounts
        half, acc = sum(cnt.values()) / 2., 0.
        for w in sorted(cnt):
            acc += cnt[w]
            if acc >= half:
                return w
        return w

    def _see_value(self, w: float) -> None:
        key = round(w, 4)
        self._vcounts[key] = self._vcounts.get(key, 0) + 1
        if self._vlo is None:
            self._vlo = self._vhi = self._vref = key
            return
        ref, S = self._value_frame()
        self._vlo, self._vhi = min(self._vlo, w), max(self._vhi, w)
        self._vref = self._median_value()
        ref2, S2 = self._value_frame()
        if ref2 == ref and S2 == S:
            return
        a = round(S / S2, 12)          # z' = a z + b, exactly
        b = round(.5 + (ref - ref2) / S2 - .5 * a, 12)
        for model in [self.m] + self._regimes:
            self._affine_model(model, a, b)
        self._invalidate()

    def _affine_model(self, M: Model, a: float, b: float) -> None:
        v = self.value_index
        for table in (M.fac, M.absent, M.ctx, M.coarse):
            for key, cell in table.items():
                if key[2] == v:
                    cell.affine(a, b)
        for table in (M.reward, M.coarse_reward):
            for cell in table.values():
                cell.affine(a, b)
        if v in M.tau_sum:
            M.tau_sum[v][1] *= a * a
            for key, (wk, d) in M.tau_dev.items():
                if key[0][2] == v:
                    M.tau_dev[key] = (wk, d * a * a)

    def _normalise(self, x: tuple[float, ...]) -> tuple[float, ...]:
        v = self.value_index
        self._see_value(x[v])
        ref, S = self._value_frame()
        z = .5 + (x[v] - ref) / S
        return x[:v] + (min(1., max(0., round(z, 9))),) + x[v + 1:]

    @staticmethod
    def _active(obs: tuple[float, ...]) -> list[int]:
        return [i for i, u in enumerate(obs) if u > .5 + 1e-9]

    def _att(self, act: tuple[int, ...]) -> tuple[int, ...]:
        """Attended part of an active set (the key of the coarse level)."""
        A = self.attended
        return tuple(i for i in act if i in A)

    # ============================================================== attention
    def _update_attention(self, active: list[int], y: tuple[float, ...], action: int) -> None:
        """Granger-style evidence cue(t+1-k) -> target(t+1), k = 1..max_age.

        Within each (action at t, lag k) and each stratum of the target's own
        state j(t), a split Bayes factor compares "j(t+1) depends on
        i(t+1-k)" with "it does not"; the strata add. Conditioning on j(t)
        removes spurious dependence between sticky processes, on the action
        the dependence created by the policy; lags > 1 let delayed causes
        earn attention. A channel is attended once it is part of such a
        dependence (either role) above the threshold log(#tests / alpha).
        Attention only defines the coarse level; exact cells see everything."""
        n = self.n_inputs
        cues = [i for i in active if i < n]
        prev = set(cues)
        on = [y[j] > .5 for j in range(n)]
        hist = self._dep_hist
        hist.insert(0, cues)
        del hist[self.max_age:]
        for k, old in enumerate(hist, 1):
            self._dep_count((action, k), old, prev, on)
        if len(self.attended) == n:
            return
        for k in range(1, len(hist) + 1):
            self._dep_test((action, k))

    def _dep_count(self, a, cues, prev, on) -> None:
        n = self.n_inputs
        st = self._dep.get(a)
        if st is None:
            st = self._dep[a] = ([[0.] * n, [0.] * n], [[0.] * n, [0.] * n], {})
        N, Y, C = st
        for j in range(n):
            sj = int(j in prev)
            N[sj][j] += 1.
            if on[j]:
                Y[sj][j] += 1.
        for i in cues:
            for j in range(n):
                if j != i:
                    cell = C.get((i, j))
                    if cell is None:
                        cell = C[(i, j)] = [0., 0., 0., 0.]   # n1|s=0, n11|s=0, n1|s=1, n11|s=1
                    sj = 2 * int(j in prev)
                    cell[sj] += 1.
                    if on[j]:
                        cell[sj + 1] += 1.

    def _dep_test(self, a) -> None:
        A = self.attended
        N, Y, C = self._dep[a]
        for (i, j), cell in C.items():
            if i in A and j in A:
                continue
            lbf = 0.
            for sj in (0, 1):
                Ns, y1 = N[sj][j], Y[sj][j]
                n1, c = cell[2 * sj], cell[2 * sj + 1]
                if n1 < 1. or n1 >= Ns or y1 < 1. or y1 >= Ns:
                    continue
                lbf += split_log_bf(((c, n1), (y1 - c, Ns - n1)))
            if lbf > self._dep_threshold:
                A.add(i)
                A.add(j)

    # ========================================================== factor prior
    def _base(self, action: int, target: int) -> float:
        root = self.m.fac.get(((), action, target))
        return root.p if root is not None and root.n > 0 else .5

    def _shrunk(self, cell: Cell | None, prior: float) -> float:
        if cell is None or cell.n <= 0:
            return prior
        m = self.prior_strength
        return min(1., max(0., (cell.total + m * prior) / (cell.n + m)))

    def _cue_lift(self, i: int, action: int, target: int, base: float):
        """None if cue i has no cell in this slice, else (n, lift of
        'cue present' over 'cue absent', cue-absent rate)."""
        key = ((i,), action, target)
        f = self.m.fac.get(key)
        if f is None:
            return None
        if f.n == 0:
            return (0., 0., 0.)
        m = self.prior_strength
        p_i = min(1., max(0., (max(0., f.total) + m * base) / (f.n + m)))
        b_i = self._shrunk(self.m.absent.get(key), base)
        return (f.n, lift(p_i, b_i), b_i)

    def _excitation(self, act: tuple[int, ...], action: int, target: int) -> tuple[float, float]:
        """Factor prior from single cues. Returns (mean, support n).

        Enabling cues combine as a noisy-OR of their lifts over the shared
        background (mean cue-absent rate); without one, suppressive lifts
        scale the background down. A cue that merely co-occurs with the
        baseline has equal present/absent rates and no lift. Support n is the
        least-sampled active cue (an untried cue keeps the prior weak)."""
        ckey = ("exc", act, action, target)
        hit = self._cache.get(ckey)
        if hit is not None:
            return hit
        base = self._base(action, target)
        terms = self._cache.setdefault(("cues", action, target), {})
        n_min, missing, lifts, backs = None, False, [], []
        for i in act:
            if i not in terms:
                terms[i] = self._cue_lift(i, action, target, base)
            c = terms[i]
            if c is None:
                missing = True
                continue
            if n_min is None or c[0] < n_min:
                n_min = c[0]
            if c[0] == 0:
                continue
            lifts.append(c[1])
            backs.append(c[2])
        if missing or n_min is None:
            n_min = 0.
        if lifts:
            b = sum(backs) / len(backs)
            pos = [s for s in lifts if s > 0.]
            if pos:
                q = 1. - b
                for s in pos:
                    q *= 1. - s
                hit = (min(1., max(0., 1. - q)), n_min)
            else:
                gate = 1.
                for s in lifts:
                    gate *= 1. + s
                hit = (min(1., max(0., b * gate)), n_min)
        else:
            hit = (base, 0.)
        self._cache[ckey] = hit
        return hit

    def _cancel_contrast(self, cue: int, target: int) -> float:
        rec = self.m.cancel.get((cue, target))
        if rec is None or rec[0].n < 3. or rec[1].n < 3.:
            return 0.
        return rec[0].p - rec[1].p

    def _inhibition(self, act: tuple[int, ...], target: int) -> float:
        """P(an expected outcome is cancelled): noisy-OR of the credible
        cancellation contrasts of the active cues."""
        ckey = ("inh", target)
        ds = self._cache.get(ckey)
        if ds is None:
            ds = self._cache[ckey] = {}
            for (i, t) in self.m.cancel:
                if t == target:
                    d = self._cancel_contrast(i, t)
                    if d > 0:
                        ds[i] = d
        survival = 1.
        if ds:
            for i in act:
                d = ds.get(i)
                if d is not None:
                    survival *= 1. - d
        return 1. - survival

    # ================================================================ timing
    def _windows_for(self, target: int) -> list[Window]:
        ckey = ("win", target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = self._cache[ckey] = [w for w in self.windows if w.target == target]
        return hit

    def _event_base(self, target: int, action: int) -> float:
        ev = self.m.event_root.get((target, action))
        return ev.p if ev is not None else .5

    def _credible_lag(self, cell: Cell, target: int, action: int) -> bool:
        if cell.n < 2.:
            return False
        rate = cell.p
        return abs(rate - self._event_base(target, action)) > 2. * sqrt(
            rate * (1. - rate) / (cell.n + 3.))

    def _lag_informative(self, cue: int, target: int, lag: int) -> bool:
        """Credibly non-baseline timing of cue -> target at this lag (any action)."""
        ckey = ("linf", cue, target, lag)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = any(c is not None and self._credible_lag(c, target, a)
                      for a in range(self.n_actions)
                      for c in (self.m.hazard.get((cue, target, lag, a)),))
            self._cache[ckey] = hit
        return hit

    def _window_informative(self, cue: int, target: int) -> bool:
        """Effect of the cue on the target located at some lag."""
        ckey = ("inf", cue, target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = any(self._credible_lag(self.m.hazard[key], target, key[3])
                      for key in self.m.hazard_index.get((cue, target), ()))
            self._cache[ckey] = hit
        return hit

    def _open_cues(self, active, target: int, depth: int = 0) -> tuple:
        """Belief-state key: informative open windows (cue, age + depth)."""
        ckey = ("open", tuple(active), target, depth)
        hit = self._cache.get(ckey)
        if hit is None:
            act = set(active)
            hit = tuple(sorted({(w.cue, w.age + depth) for w in self._windows_for(target)
                                if w.cue not in act
                                and self._lag_informative(w.cue, target, w.age + 1 + depth)}))
            self._cache[ckey] = hit
        return hit

    def _timing_split(self, gkey):
        """Context channels that modulate one generic timing cell, and its
        exact-context children grouped by the remaining (reduced) context."""
        ck = ("hsel", gkey)
        hit = self._cache.get(ck)
        if hit is not None:
            return hit
        M = self.m
        kids = [(key[4], c) for key in M.hctx_index.get(gkey, ())
                for c in (M.hazard_ctx[key],) if c.n > 1e-6]
        irr = set()
        if len(kids) > 1:
            n_tot = sum(c.n for _, c in kids)
            e_tot = sum(c.k for _, c in kids)
            p = (e_tot + .5) / (n_tot + 1.)
            q = p * (1. - p)
            thr = max(3.84, log(max(1., n_tot)))
            chans = set()
            for cx, _ in kids:
                chans.update(cx)
            for j in chans:
                if self._p_uncontrollable(j) < .5:
                    continue                  # controllable channels keep their split
                e1 = n1 = 0.
                for cx, c in kids:
                    if j in cx:
                        e1 += c.k
                        n1 += c.n
                n0 = n_tot - n1
                if n1 < 1. or n0 < 1.:
                    continue                  # untested on one side
                e0 = e_tot - e1
                if ((e1 - n1 * p) ** 2 / n1 + (e0 - n0 * p) ** 2 / n0) / q <= thr:
                    irr.add(j)
        agg: dict = {}
        for cx, c in kids:
            red = tuple(i for i in cx if i not in irr)
            r = agg.get(red)
            if r is None:
                agg[red] = [c.k, c.n]
            else:
                r[0] += c.k
                r[1] += c.n
        hit = self._cache[ck] = (irr, agg)
        return hit

    def _reduced_count(self, gkey, ctx) -> float:
        irr, agg = self._timing_split(gkey)
        r = agg.get(tuple(i for i in ctx if i not in irr))
        return r[1] if r is not None else 0.

    def _timing_posterior(self, cue: int, target: int, lag: int, action: int,
                          ctx: tuple[int, ...]) -> tuple[float, float]:
        """Beta (a, b) of a timing cell: exact context -> reduced context ->
        generic (cue, target, lag, action) cell, each level shrunk toward the
        one above with prior_strength pseudo-observations."""
        tk = ("tp", cue, target, lag, action, ctx)
        hit = self._cache.get(tk)
        if hit is not None:
            return hit
        gkey = (cue, target, lag, action)
        g = self.m.hazard.get(gkey)
        g_rate = g.p if g is not None else .5
        irr, agg = self._timing_split(gkey)
        red = tuple(i for i in ctx if i not in irr) if irr else ctx
        r = agg.get(red)
        re_, rn = (r[0], r[1]) if r is not None else (0., 0.)
        c = self.m.hazard_ctx.get(gkey + (ctx,))
        ce, cr = (c.k, c.n) if c is not None else (0., 0.)
        m = self.prior_strength
        r_rate = (re_ + m * g_rate) / (rn + m)
        hit = self._cache[tk] = (1. + ce + m * r_rate, 1. + max(0., cr - ce) + m * (1. - r_rate))
        return hit

    def _timing_lift(self, act: tuple[int, ...], action: int, target: int, depth: int) -> float:
        """Strongest credible lift of an open window's timing cell over the
        event baseline (windows of cues active now are the factors' job)."""
        wins = self._windows_for(target)
        if not wins:
            return 0.
        ckey = ("hz", target, action, depth)
        cand = self._cache.get(ckey)
        if cand is None:
            cand = []
            for w in wins:
                lag = w.age + 1 + depth
                rec = self.m.hazard.get((w.cue, target, lag, action))
                if rec is not None and rec.n >= 2.:
                    cand.append((w.cue, lag))
            self._cache[ckey] = cand
        if not cand:
            return 0.
        base = self._event_base(target, action)
        h = 0.
        active = set(act)
        for cue, lag in cand:
            if cue in active:
                continue
            a, b = self._timing_posterior(cue, target, lag, action, act)
            rate = a / (a + b)
            if abs(rate - base) <= 2. * sqrt(a * b / ((a + b) ** 2 * (a + b + 1.))):
                continue
            s = lift(rate, base)
            h = s if abs(s) > abs(h) else h
        return min(1., max(-1., h))

    def _event_bonus(self, x: tuple[float, ...], action: int, depth: int) -> float:
        """Novelty of an open, unexplained expectation on the value channel:
        max over open windows whose effect is not yet located at any lag of
        salience / (1 + n), n = tests of (cue, lag, action, reduced context)."""
        v = self.value_index
        act = tuple(self._active(x))
        active = set(act)
        out = 0.
        for w in self._windows_for(v):
            if w.cue in active or self._window_informative(w.cue, v):
                continue
            r = self._reduced_count((w.cue, v, w.age + 1 + depth, action), act)
            out = max(out, w.salience / (1. + r))
        return out

    # =========================================================== prediction
    def predict(self, obs: Iterable[float], action: int, target: int, *, depth: int = 0) -> float:
        x = tuple(float(v) for v in obs)
        self._validate_obs(x)
        if not 0 <= action < self.n_actions or not 0 <= target < self.n_inputs:
            raise ValueError("action/target out of range")
        self._invalidate()   # public entry: tables may have been edited directly
        return self._predict(x, action, target, depth)[0]

    def predict_next(self, obs: Iterable[float], action: int, *, depth: int = 0) -> tuple[float, ...]:
        x = tuple(float(v) for v in obs)
        self._invalidate()
        return self._predict_next(x, action, depth)

    def _predict_next(self, x, action: int, depth: int) -> tuple[float, ...]:
        return tuple(self._predict(x, action, j, depth)[0] for j in range(self.n_inputs))

    def _predict(self, x, action: int, target: int, depth: int,
                 act: tuple[int, ...] | None = None) -> tuple[float, float]:
        """(mean, sd) of target after (x, action); memoised per active set."""
        if act is None:
            act = tuple(self._active(x))
        mkey = (act, action, target, depth)
        hit = self._memo.get(mkey)
        if hit is None:
            hit = self._memo[mkey] = self._predict_uncached(x, action, target, depth, act)
        return hit

    def _predict_uncached(self, x, action: int, target: int, depth: int,
                          act: tuple[int, ...]) -> tuple[float, float]:
        M, m0 = self.m, self.prior_strength
        # ---- level 0: factor prior (cue lifts x inhibition, reward model)
        exc, n_min = self._excitation(act, action, target)
        prior = min(1., max(0., exc * (1. - self._inhibition(act, target))))
        m = m0 * n_min / (1. + n_min)
        sd_prior = sqrt(max(prior * (1. - prior), .25 / (1. + n_min)) / (m + 1.))
        if target == self.value_index:
            mb = self._model_based_value(x, action, depth, act)
            if mb is not None:
                p_mb, sd_mb, n_mb = mb
                w1, w2 = 1. / max(sd_prior ** 2, 1e-6), 1. / max(sd_mb ** 2, 1e-6)
                prior = (w1 * prior + w2 * p_mb) / (w1 + w2)
                sd_prior = sqrt(1. / (w1 + w2))
                m = max(m, m0 * n_mb / (1. + n_mb))
        # ---- level 1: attended context, partial pooling with learned spread
        ak = self._att(act)
        if ak and ak != act:
            ck = (ak, action, target)
            cc = M.coarse.get(ck)
            if cc is not None and cc.n > 0:
                pc, w_obs = shrink(cc, prior, m)
                sdc2 = max(0., w_obs) / (cc.n + m + 1.)
                # spread of exact contexts around their parents over all
                # parents of this target (learned irrelevance generalises),
                # with one pseudo-child at maximal spread
                W, D = M.tau_sum.get(target, (0., 0.))
                dev, k = (.25 + D) / (1. + W), 1.
                for kid in M.kids.get(ck, ()):
                    c = M.ctx.get((kid, action, target))
                    if c is not None and c.n > 0.:
                        wk = min(1., c.n)
                        dev += wk * ((c.total + pc) / (c.n + 1.) - pc) ** 2
                        k += wk
                tau2 = dev / k
                prior, sd_prior = pc, sqrt(sdc2 + tau2)
                m = min(50., max(w_obs, 1e-3) / tau2)
        # ---- level 2: exact context; level 3: + open windows (belief state)
        cell = M.ctx.get((act, action, target))
        if cell is not None and cell.n > 0:
            p, v = shrink(cell, prior, m)
            sd = sqrt(max(0., v) / (cell.n + m + 1.))
        else:
            p, sd = prior, sd_prior
        oc = self._open_cues(act, target, depth)
        mem = M.ctx.get((act, action, target, oc))
        if mem is not None and mem.n > 0:
            p, v = shrink(mem, p, m0)
            sd = min(sd, sqrt(max(0., v) / (mem.n + m0 + 1.)))
        # ---- controllability: mix in the action-pooled cells
        n_obs = mem.n if mem is not None else 0.
        ps = self._p_uncontrollable(target)
        if ps > 1e-3:
            pool = M.ctx.get((act, -1, target))
            if pool is not None and pool.n > 0 and pool.kind == "binary":
                p_pool = (pool.k + m0 * p) / (pool.n + m0)
                sd_pool = sqrt(p_pool * (1. - p_pool) / (pool.n + m0 + 1.))
                pm = M.ctx.get((act, -1, target, oc))
                if pm is not None and pm.n > 0 and pm.kind == "binary":
                    p_pool = (pm.k + m0 * p_pool) / (pm.n + m0)
                    sd_pool = min(sd_pool, sqrt(p_pool * (1. - p_pool) / (pm.n + m0 + 1.)))
                    n_obs = max(n_obs, ps * pm.n)
                p = ps * p_pool + (1. - ps) * p
                sd = ps * sd_pool + (1. - ps) * sd
        # ---- timing lift, a prior that the belief-state cell overrides
        h = self._timing_lift(act, action, target, depth)
        if h != 0. and n_obs > 0:
            h *= m0 / (m0 + n_obs)
        return min(1., max(0., apply_lift(p, h))), sd

    def _model_based_value(self, x, action: int, depth: int,
                           act: tuple[int, ...]) -> tuple[float, float, float]:
        """E[w'] through the predicted next observation and the reward model.
        A predicted state never observed has the uninformed prior; one whose
        attended part is known is partially pooled over its siblings."""
        v, M, A = self.value_index, self.m, self.attended
        yhat = [self._predict(x, action, j, depth, act)[0] if j != v else 0.
                for j in range(self.n_inputs)]
        key = tuple(j for j in range(self.n_inputs) if j != v and yhat[j] > .5 + 1e-9)
        cell = M.reward.get(key)
        q = 1.     # probability of the modal next state (attended channels)
        for j in (A if A else range(self.n_inputs)):
            if j != v:
                q *= max(yhat[j], 1. - yhat[j])
        if (cell is None or cell.n < 1.) and A:
            ckey = tuple(j for j in key if j in A)
            cc = M.coarse_reward.get(ckey)
            if cc is not None and cc.n >= 1.:
                mc = cc.mean
                dev, k = 1. / 12., 1.
                for kid in M.reward_kids.get(ckey, ()):
                    c = M.reward.get(kid)
                    if c is not None and c.n > 0.:
                        wk = min(1., c.n)
                        dev += wk * ((c.mean * c.n + mc) / (c.n + 1.) - mc) ** 2
                        k += wk
                within = (cc.m2 + 1. / 12.) / (cc.n + 1.)
                return mc, sqrt(within + dev / k + (1. - q) * .25), 0.
        if cell is None or cell.n < 1.:
            return .5, sqrt(1. / 12. + (1. - q) * .25), 0.
        if cell.kind == "binary":
            mean = cell.p
            within = mean * (1. - mean)
        else:
            mean = cell.mean
            within = (cell.m2 + 1. / 12.) / (cell.n + 1.)
        return mean, sqrt(within * (1. + 1. / (cell.n + 1.)) + (1. - q) * .25), cell.n

    # ====================================================== controllability
    def _p_uncontrollable(self, target: int) -> float:
        t = self.m.ctrl_total.get(target)
        if t is None:
            return 0.
        return 1. / (1. + exp(min(50., max(-50., t))))

    def _ckey(self, fk):
        """Controllability context: the attended part of an active set (the
        whole set before anything is attended)."""
        return self._att(fk) if self.attended else fk

    def _rebuild_ctrl_c(self) -> None:
        """Attention changed, hence the coarse partition: recompute every
        coarse controllability BF from the exact belief-state cells."""
        M = self.m
        self._ctrl_c_alen = len(self.attended)
        M.ctrl_lbf.clear()
        M.ctrl_total.clear()
        by_ak: dict = {}
        self._fk_by_ak = by_ak
        for fk in self._all_fk:
            by_ak.setdefault(self._ckey(fk), set()).add(fk)
        todo = set()
        for k in M.ctx:
            if len(k) == 4 and k[1] >= 0:
                todo.add((self._ckey(k[0]), k[2], k[3]))
        for ak, target, oc in todo:
            self._update_controllability(ak, target, oc)

    def _update_controllability(self, key, target: int, oc) -> None:
        """Split BF 'rate depends on the action' within one attended context
        and set of open windows, from the summed exact belief-state cells:
        noise channels do not split the comparison, and the open windows
        remove the confound of a policy that acts on memory."""
        M = self.m
        groups, binary = [], True
        kids = self._fk_by_ak.get(key, ())
        for a in range(self.n_actions):
            n = pos = 0.
            for fk in kids:
                c = M.ctx.get((fk, a, target, oc))
                if c is not None and c.n > 0:
                    if c.kind != "binary":
                        binary = False
                        break
                    n += c.n
                    pos += c.k
            if not binary:
                break
            if n > 0:
                groups.append((pos, n))
        lbf = 50. if not binary else 0. if len(groups) < 2 else split_log_bf(groups)
        ck = (key, target, oc)
        old = M.ctrl_lbf.get(ck, 0.)
        M.ctrl_lbf[ck] = lbf
        M.ctrl_total[target] = M.ctrl_total.get(target, 0.) + lbf - old

    # ======================================================= change / regimes
    @staticmethod
    def _transition_loglik(ctx: dict, key, action: int, y, mask) -> float:
        """log p(y | exact context, action) under one model (fresh prior for
        channels it has not seen in this context)."""
        ll = 0.
        for target, outcome in enumerate(y):
            if not mask[target]:
                continue
            cell = ctx.get((key, action, target))
            if cell is None or cell.n < 1.:
                ll += log(.5) if outcome in (0., 1.) else 0.
            else:
                ll += log(max(1e-12, cell.density(outcome)))
        return ll

    def _regime_step(self, key, action: int, y, mask, surprised: bool) -> bool:
        """Switch to a stored model whose recent predictive evidence beats the
        current one by regime_bf nats (memory regime_decay per step); only
        surprising transitions are evaluated (retrieval by prediction error)."""
        if not self._regimes:
            return False
        if not surprised:
            self._reg_evidence = [d * self.regime_decay for d in self._reg_evidence]
            return False
        cur = self._transition_loglik(self.m.ctx, key, action, y, mask)
        best, best_i = self.regime_bf, -1
        for i, snap in enumerate(self._regimes):
            d = self._reg_evidence[i] * self.regime_decay + (
                self._transition_loglik(snap.ctx, key, action, y, mask) - cur)
            self._reg_evidence[i] = d
            if d > best:
                best, best_i = d, i
        if best_i < 0:
            return False
        snap = self._regimes.pop(best_i)
        self._regimes.append(self.m)
        self._reg_evidence = [0.] * len(self._regimes)
        self.m = snap
        self._invalidate()
        self._g_surprise = self._g_applied = 0.
        self.regime_switches += 1
        return True

    def _new_regime(self) -> None:
        self._regimes.append(self.m.copy())
        self._reg_evidence.append(0.)
        if len(self._regimes) > self.regime_max:
            self._regimes.pop(0)
            self._reg_evidence.pop(0)

    def _global_change_step(self, active, action: int, y, mask) -> None:
        """Environment-level change point shared by every cell.

        A stored regime that explains recent surprising transitions better is
        reinstated. Otherwise the evidence for a new change point is the
        joint log likelihood ratio fresh prior vs exact-context cells, made
        outlier-robust (one step adds at most log 1/eps) and accumulated over
        a run of surprise (CUSUM); c = P(change) with prior hazard H. Memory
        is discounted to 1 - c once per run, after archiving the model when c
        first exceeds regime_store; quiet periods apply only the prior hazard."""
        H = self.global_hazard
        if H <= 0.:
            self.last_global_change = 0.
            return
        key = tuple(active)
        log_lr = 0.
        for target, outcome in enumerate(y):
            if not mask[target]:
                continue
            cell = self.m.ctx.get((key, action, target))
            if cell is None or cell.n < 1.:
                continue
            p0 = .5 if (cell.kind == "binary" and outcome in (0., 1.)) else 1.
            log_lr += log(p0) - log(max(1e-12, cell.density(outcome)))
        if self._regime_step(key, action, y, mask, log_lr > 0.):
            self.last_global_change = 0.
            return
        e = self.global_eps
        L = min(50., log_lr)
        lr = -log((1. - e) * exp(-L) + e) if L > -50. else L
        s_t = min(50., self._g_surprise + lr)
        c = 1. / (1. + exp(-(log(H / (1. - H)) + s_t)))
        self.last_global_change = c
        if s_t > 0.:
            self._g_surprise = s_t
            if c > self.regime_store and self._g_applied <= self.regime_store:
                self._new_regime()
            if c > self._g_applied:
                lam = (1. - c) / (1. - self._g_applied)
                self._g_applied = c
                self._discount_all(lam * exp(-self._g_pending))
                self._g_pending = 0.
        else:
            self._g_surprise = self._g_applied = 0.
            self._g_pending += -log(1. - c)
            if self._g_pending > .01:
                self._discount_all(exp(-self._g_pending))
                self._g_pending = 0.

    def _discount_all(self, lam: float) -> None:
        self._cache = {}
        for table in self.m.cell_tables():
            for c in table.values():
                c.decay(lam)
        for pair in self.m.cancel.values():
            for c in pair:
                c.decay(lam)

    def _invalidate(self) -> None:
        self._memo = {}
        self._cache = {}

    def _absorb(self, cell: Cell, y: float) -> None:
        cell.change_point(y, self.change_hazard)
        cell.update(y)

    def _absorb_count(self, cell: Cell, event: bool, w: float = 1.) -> None:
        cell.change_point_memoryless(event, self.change_hazard)
        cell.update(float(event), w)

    # ================================================================ control
    def _best_next(self, s: tuple[float, ...]) -> float:
        ke, best = self.event_explore, 0.
        for a in range(self.n_actions):
            p, _ = self._predict(s, a, self.value_index, 1)
            if ke > 0.:
                p += ke * self._event_bonus(s, a, 1) * (1. - p)
            best = max(best, p)
        return best

    def _expected_continuation(self, nxt: tuple[float, ...]) -> float:
        """E over the next observation of max_a' value: the most uncertain
        (attended) channels, at most rollout_branch, are enumerated as
        independent Bernoullis; the rest take their modal value."""
        v, A = self.value_index, self.attended
        unc = [(min(p, 1. - p), j) for j, p in enumerate(nxt)
               if j != v and min(p, 1. - p) > self.rollout_eps and (not A or j in A)]
        unc.sort(key=lambda u: -u[0])
        U = [j for _, j in unc[:self.rollout_branch]]
        base = [(1. if p > .5 else 0.) if j != v else p for j, p in enumerate(nxt)]
        total = 0.
        for bits in range(1 << len(U)):
            s, w = list(base), 1.
            for i, j in enumerate(U):
                on = (bits >> i) & 1
                s[j] = float(on)
                w *= nxt[j] if on else 1. - nxt[j]
            if w >= 1e-6:
                total += w * self._best_next(tuple(s))
        return total

    def _q(self, obs: tuple[float, ...], action: int) -> float:
        """Q = E[w'] + sqrt((explore sd)^2 + event bonus^2) + gamma E_x'[max_a' ...].

        Optimism at the root only; the imagined successor never activates the
        value channel (its predicted mean is the reward term, not a cue)."""
        v = self.value_index
        immediate, sd_now = self._predict(obs, action, v, 0)
        nxt = self._predict_next(obs, action, 0)
        nxt = nxt[:v] + (min(nxt[v], .5),) + nxt[v + 1:]
        continuation = self._expected_continuation(nxt)
        bonus = self.explore * sd_now
        if self.event_explore > 0.:
            e = self.event_explore * self._event_bonus(obs, action, 0) * (1. - immediate)
            bonus = sqrt(bonus * bonus + e * e)
        return immediate + bonus + self.gamma * continuation

    def step(self, obs: Iterable[float], *, tie_noise: Iterable[float] | None = None) -> int:
        x = tuple(float(v) for v in obs)
        self._validate_obs(x)
        if self._pending is not None:
            raise RuntimeError("learn(next_obs) must follow each step(obs)")
        x = self._normalise(x)
        if tie_noise is None:
            noise = [self.tie_rng.random() for _ in range(self.n_actions)]
        else:
            noise = list(tie_noise)
            if len(noise) != self.n_actions:
                raise ValueError("tie_noise must have one value per action")
        self._invalidate()
        scores = [self._q(x, a) for a in range(self.n_actions)]
        best = max(scores)
        tied = [a for a, q in enumerate(scores) if abs(q - best) < 1e-12]
        action = max(tied, key=lambda a: noise[a])
        self._pending = (x, action)
        return action

    def _validate_obs(self, x: tuple[float, ...]) -> None:
        if len(x) != self.n_inputs or any(not 0 <= v <= 1 for v in x):
            raise ValueError("observation width/range mismatch")

    # ================================================================ learning
    def learn(self, next_obs: Iterable[float], *, observed: Iterable[bool] | None = None) -> None:
        if self._pending is None:
            raise RuntimeError("step(obs) must precede learn(next_obs)")
        x, action = self._pending
        y = tuple(float(v) for v in next_obs)
        self._validate_obs(y)
        y = self._normalise(y)
        mask = tuple(bool(v) for v in observed) if observed is not None else (True,) * self.n_inputs
        if len(mask) != self.n_inputs:
            raise ValueError("observed mask width mismatch")
        active = self._active(x)
        # salience of each onset = 1 - its prediction before learning
        self._onset_salience = {j: 1. - self._predict(x, action, j, 0)[0]
                                for j in range(self.n_inputs)
                                if mask[j] and y[j] > .5 and x[j] <= .5}
        self._global_change_step(active, action, y, mask)
        full_key = tuple(active)
        att_key = self._att(full_key)
        self._all_fk.add(full_key)
        if len(self.attended) != self._ctrl_c_alen:
            self._rebuild_ctrl_c()
        self._fk_by_ak.setdefault(self._ckey(full_key), set()).add(full_key)
        for target, outcome in enumerate(y):
            if mask[target]:
                self._learn_target(active, full_key, att_key, action, target, outcome)
        v = self.value_index
        if mask[v]:
            M = self.m
            rkey = tuple(j for j in range(self.n_inputs) if j != v and y[j] > .5)
            self._absorb(M.reward.setdefault(rkey, Cell()), y[v])
            ckey = self._att(rkey)
            self._absorb(M.coarse_reward.setdefault(ckey, Cell()), y[v])
            M.reward_kids.setdefault(ckey, set()).add(rkey)
        self._update_timing(y, action, set(active), mask)
        self._invalidate()
        self._update_attention(active, y, action)
        self._invalidate()
        self._pending = None

    def _learn_target(self, active, full_key, att_key, action: int, target: int,
                      outcome: float) -> None:
        M = self.m
        root = M.fac.setdefault(((), action, target), Cell())
        # context hierarchy (exact, action-pooled, attended, + open windows)
        ctx_key = (full_key, action, target)
        self._absorb(M.ctx.setdefault(ctx_key, Cell()), outcome)
        self._absorb(M.ctx.setdefault((full_key, -1, target), Cell()), outcome)
        if att_key:
            ck = (att_key, action, target)
            cc = M.coarse.setdefault(ck, Cell())
            self._absorb(cc, outcome)
            M.kids.setdefault(ck, set()).add(full_key)
            if att_key != full_key:
                self._track_spread(ck, full_key, target, cc, M.ctx[ctx_key])
        oc = self._open_cues(active, target)
        self._absorb(M.ctx.setdefault(ctx_key + (oc,), Cell()), outcome)
        self._absorb(M.ctx.setdefault((full_key, -1, target, oc), Cell()), outcome)
        self._update_controllability(self._ckey(full_key), target, oc)
        # factor level: cancellation contrast (inhibition), blocking, cue cells
        fac = M.fac
        known = [fac[k] for i in active
                 if (k := ((i,), action, target)) in fac and fac[k].n >= 2.]
        excitors = {i for i in active
                    if ((i,), action, target) in fac and fac[((i,), action, target)].n >= 2.
                    and fac[((i,), action, target)].p >= self.excite_threshold}
        expected = max((c.p for c in known), default=0.)
        # 2x2 cancellation contrast: a miss of an expected outcome is credited
        # against every cue except the excitors, a delivery as evidence
        # against cancellation, only with a non-excitor cue present to blame.
        credit_cancel = (outcome == 0. and expected >= self.credit_threshold
                         and bool(set(active) - excitors))
        if credit_cancel or (outcome == 1. and expected >= self.credit_threshold):
            for cue in range(self.n_inputs):
                if credit_cancel and cue in excitors:
                    continue
                rec = M.cancel.setdefault((cue, target), [Cell(), Cell()])
                self._absorb_count(rec[0] if cue in active else rec[1], credit_cancel)
        strong = self._strong_explainers(active, action, target, outcome)
        self._absorb(root, outcome)
        active_set = set(active)
        for cue in active:
            if strong - {cue}:
                continue
            self._absorb(fac.setdefault(((cue,), action, target), Cell()), outcome)
        for cue in range(self.n_inputs):
            if cue in active_set:
                continue
            akey = ((cue,), action, target)
            if akey in fac or akey in M.absent:
                self._absorb(M.absent.setdefault(akey, Cell()), outcome)

    def _strong_explainers(self, active, action: int, target: int, outcome: float) -> set[int]:
        """Kamin blocking: cues with a strong lift whose prediction already
        matches the outcome; the other cues are then not credited."""
        base = self._base(action, target)
        out = set()
        for j in active:
            f = self.m.fac.get(((j,), action, target))
            if f is None or f.n < 3.:
                continue
            absent = self.m.absent.get(((j,), action, target))
            b = absent.p if absent is not None and absent.n > 0 else base
            if (abs(lift(f.p, b)) >= self.block_strength
                    and abs(outcome - f.p) < self.block_error):
                out.add(j)
        return out

    def _track_spread(self, ck, kid, target: int, cc: Cell, c: Cell) -> None:
        """Running per-target sum of child-vs-parent squared deviations."""
        if cc.n <= 0. or c.n <= 0.:
            return
        pc = cc.k / cc.n if cc.kind == "binary" else cc.mean
        wk = min(1., c.n)
        d = wk * ((c.total + pc) / (c.n + 1.) - pc) ** 2
        M = self.m
        old = M.tau_dev.get((ck, kid), (0., 0.))
        M.tau_dev[(ck, kid)] = (wk, d)
        acc = M.tau_sum.setdefault(target, [0., 0.])
        acc[0] += wk - old[0]
        acc[1] += d - old[1]

    def _update_timing(self, y, action: int, prev_active: set[int], mask) -> None:
        """Advance open windows: (cue, target, age, action[, context]) cells
        count exposures and events weighted by the onset's salience; an event
        consumes the window, an unobserved target right-censors it. New
        windows open on rising edges of transient channels (mean run <= 3);
        a recurrence renews the cue's windows."""
        M = self.m
        self._time += 1
        ctx = tuple(sorted(prev_active))
        for target in range(self.n_inputs):
            if mask[target]:
                self._absorb_count(M.event_root.setdefault((target, action), Cell()),
                                   y[target] > .5)
        kept = []
        for w in self.windows:
            age = w.age + 1
            hkey = (w.cue, w.target, age, action)
            rec = M.hazard.get(hkey)
            if rec is None:
                rec = M.hazard[hkey] = Cell()
                M.hazard_index.setdefault((w.cue, w.target), []).append(hkey)
            ckey = hkey + (ctx,)
            crec = M.hazard_ctx.get(ckey)
            if crec is None:
                crec = M.hazard_ctx[ckey] = Cell()
                M.hctx_index.setdefault(hkey, []).append(ckey)
            if not mask[w.target]:
                rec.update(0., w.salience)
                crec.update(0., w.salience)
                continue
            ev = y[w.target] > .5
            self._absorb_count(rec, ev, w.salience)
            self._absorb_count(crec, ev, w.salience)
            if not ev and age < self.max_age:
                kept.append(Window(w.cue, w.target, age, w.salience))
        self.windows = kept
        active_now = set(self._active(y))
        for fell in prev_active - active_now:
            rise = self._rise.pop(fell, None)
            if rise is not None:
                run = float(self._time - rise)
                self._runlen[fell] = .9 * self._runlen.get(fell, run) + .1 * run
        for cue in sorted(active_now - prev_active):
            self._rise[cue] = self._time
            if self._runlen.get(cue, 0.) > 3.:
                continue
            self.windows = [w for w in self.windows if w.cue != cue]
            for target in range(self.n_inputs):
                if target != cue and target not in active_now and mask[target]:
                    self.windows.append(Window(cue, target, 0,
                                               self._onset_salience.get(cue, 1.)))
