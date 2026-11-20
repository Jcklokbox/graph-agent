"""Experimental sparse online posterior-predictive transition graph (v8).

Standard-library-only research kernel. It is NOT a claim of general-purpose
planning or causal identification; see research/posterior-graph-online-results.md
for measured behaviour and failures.

Model
-----
Observations x in [0,1]^n, A discrete actions, one channel is the value w.
Every statistic is a conjugate cell on [0,1] (Beta-Bernoulli for 0/1 data,
Welford moments for fractional data); there is no iterative optimisation.

Prediction of target j after (x, a) is a hierarchical back-off:

  factor prior     -> exact context  -> exact context + open windows (cue, tau)
  (pairs/singletons,  (active set, a)    (belief-state memory; tau advances
   inhibition)                           along imagined rollouts)

each level being the prior of the next. Additional terms:

* pair births: 2x2 interaction Bayes factor (both active vs exactly one),
  threshold log(C(n,2)*A/alpha); singletons combine by signed noisy-OR of
  present-vs-absent strengths, Kamin blocking, residual counts.
* inhibition: target-level 2x2 cancellation contrast, p = p_exc (1 - P_inhib);
  learned from observation under any action, transfers across excitors.
* hazard: windows open on rising edges of transient channels (renewal on
  recurrence), cells (cue, target, lag, action[, context]) with censoring and
  consumption, counts weighted by the surprise (salience) of the anchoring
  onset; credible lift over the root event rate. Context cells back off to
  a reduced context (channels that credibly modulate the cell, by a BIC
  split test, plus controllable channels) with empirical-Bayes strength, so
  iid noise channels do not fragment timing evidence. A window enters the
  belief-state key only at lags with credible non-baseline timing, and the
  lift is a prior that the exact belief-state cell overrides as it fills.
* value coding: reference point (median experience) and scale, stored value
  statistics re-expressed affinely when the frame changes.
* attention: Granger-style dependence Bayes factors (per action and lag,
  stratified by the target's own past) select attended channels; they define
  a coarse level with partial pooling whose spread prior is learned across
  all coarse cells (learned irrelevance generalises).
* controllability: a within-context Beta-Binomial BF decides whether a channel
  depends on the action; uncontrollable (Pavlovian) channels share an
  action-pooled exact context across actions (also with its open windows).
* reward model R(next observation) -> w, used through the predicted next state
  as a precision-weighted prior of the value prediction; a predicted state
  that was never observed carries the uninformed prior (uniform on [0, 1]),
  so "food will be visible at the place I can reach" is worth checking.

Forgetting is surprise-modulated, not blind: before absorbing y, each cell
discounts its counts by c = sigmoid(logit h + S), S an accumulated (CUSUM)
log likelihood ratio fresh-prior vs cell. A global change point with joint,
outlier-robust (per-step surprise <= log 1/eps), accumulated evidence over all
channels discounts every cell (including avoided actions) once per run of
surprise, only when the world as a whole has become unpredictable; the old
model is archived as a latent regime first, and stored regimes compete with
the current model by recent predictive evidence, so a return to a known
regime reinstates it instead of relearning it (savings).

Control: Q(x,a) = E[w'] + sqrt((explore * SD[w'])^2 + B(x,a,0)^2)
                 + gamma * E_{x'|x,a} max_a' (E[w''|x'] + B(x',a',1)),
the expectation enumerating up to `rollout_branch` uncertain (attended)
channels of the predicted next observation,
with outcome optimism at the root only and an expected-event bonus
B = event_explore * salience / (1 + n) * (1 - E[w]) at both depths, for open
windows on the value channel whose cue onset was unpredicted (salience) and
whose effect is not yet located (no informative lag), at any lag up to
max_age. n counts tests of the exact (cue, lag, action, context) combination, so the bonus is consistent between the plan and its second step.
At the root the two optimism sources (value posterior / novel-state prior and
the open expectation) are independent uncertainties about the same reward and
share one budget (variances add), so a transition that is both novel and an
open test is not counted twice. The imagined successor never activates the
value channel (its predicted mean is the reward term, not a cue). Tie noise
comes from a separate stream (tie_seed), so exact closed-loop equivariance
under relabeling is preserved.

Performance: posterior queries are memoised per decision (cleared by
_invalidate() at every learn(); call it after editing tables by hand).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, lgamma, log, pi, sqrt
from typing import Iterable

# Learned tables that together form one latent regime's model (stored and
# restored by contextual inference over regimes, see _regime_step).
_OP_TABLES = ("factors", "absent_factors", "context_factors", "reward_cells",
              "coarse_factors", "coarse_reward")
_LIST_TABLES = ("cancellation", "hazard", "hazard_ctx", "_event_root", "_tau_sum",
                "_cand_index", "_hazard_index", "_hctx_index")
_SET_TABLES = ("_kids", "_reward_kids")
_PLAIN_TABLES = ("_ctrl_lbf", "_ctrl_total", "_ctrl_total_c", "_tau_dev")
REGIME_TABLES = _OP_TABLES + _LIST_TABLES + _SET_TABLES + _PLAIN_TABLES + (
    "pair_candidates", "births")







@dataclass
class OutcomePosterior:
    """Discounted posterior moments; binary and continuous data never conflated."""
    kind: str | None = None
    n: float = 0.0
    positive: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    surprise: float = 0.0   # CUSUM log-evidence for an unabsorbed change

    def update(self, y: float) -> None:
        if not 0 <= y <= 1:
            raise ValueError("outcome must be in [0,1]")
        if self.kind is None:
            self.kind = "binary" if y in (0., 1.) else "continuous"
        if self.kind == "binary" and y not in (0., 1.):
            # Mixed-valued channel: promote the moments to the continuous
            # family; never encode a fractional value as Bernoulli counts.
            self.kind = "continuous"
        self.n += 1.0
        if self.kind == "binary":
            self.positive += float(y == 1.)
        delta = y - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (y - self.mean)

    def affine(self, a: float, b: float) -> None:
        """Re-express stored outcomes y as a*y + b (exact for the moments)."""
        if self.kind is None:
            return
        self.mean = a * self.mean + b
        self.m2 *= a * a
        if self.kind == "binary" and not (a == 1. and b == 0.):
            self.kind = "continuous"

    def decay(self, lam: float) -> None:
        # The weighted mean is invariant to uniform weight rescaling; the
        # count (hence prior pull and novelty) and the weighted m2 shrink.
        self.n *= lam
        self.positive *= lam
        self.m2 *= lam

    def predictive_density(self, y: float) -> float:
        """p(y | this cell): Bernoulli for 0/1 cells, Gaussian otherwise.

        The Gaussian variance carries one pseudo-observation of the
        Uniform(0,1) variance (1/12), so a cell with little data is broad.
        """
        if self.kind == "binary" and y in (0., 1.):
            p = self.predictive_mean
            return p if y == 1. else 1. - p
        k, v0 = 1., 1. / 12.
        var = (self.m2 + k * v0) / (self.n + k) * (1. + 1. / (self.n + k))
        return exp(-.5 * (y - self.mean) ** 2 / var) / sqrt(2. * pi * var)

    def change_discount(self, y: float, hazard: float) -> float:
        """Bayesian change-point step before absorbing y (Adams-MacKay, r=0 vs r>0).

        c = P(regime changed | evidence) = sigmoid(logit h + S), where S is
        the accumulated log likelihood ratio log p0(y)/p(y|cell) (CUSUM),
        with p0 the fresh-prior predictive (1/2 for 0/1, uniform density 1
        otherwise). Old evidence is down-weighted by (1 - c): predictable
        outcomes leave memory intact, a run of surprises restores
        plasticity within a few observations.
        """
        if self.n <= 0 or hazard <= 0.:
            return 0.
        p0 = .5 if (self.kind == "binary" and y in (0., 1.)) else 1.
        # Accumulated evidence (Page CUSUM / log Shiryaev-Roberts): a run of
        # surprises adds up, predictable outcomes drain it to zero.
        llr = log(p0) - log(max(1e-12, self.predictive_density(y)))
        s_t = min(50., self.surprise + llr)   # current step enters unfloored
        c = 1. / (1. + exp(-(log(hazard / (1. - hazard)) + s_t)))
        self.decay(1. - c)
        # Stored evidence is floored at 0 (CUSUM) and the part absorbed by
        # the discount is spent, so one change is not counted twice.
        self.surprise = max(0., s_t) * (1. - c)
        return c

    @property
    def predictive_mean(self) -> float:
        if self.kind == "binary":
            return (1. + self.positive) / (2. + self.n)
        return self.mean if self.n > 0 else .5

    @property
    def predictive_variance(self) -> float:
        if self.kind == "binary":
            p = self.predictive_mean
            return p * (1. - p)
        return self.m2 / max(1., self.n - 1.) if self.n > 1 else float("inf")

    @property
    def mean_parameter_variance(self) -> float:
        if self.kind == "binary":
            p = self.predictive_mean
            return p * (1. - p) / (self.n + 3.)
        return self.predictive_variance / self.n if self.n > 1 else float("inf")


def _count_change_discount(rec: list[float], i_event: int, i_risk: int,
                           event: bool, hazard: float) -> None:
    """Change-point discount for a binary [events, exposures] cell."""
    r = rec[i_risk]
    if r <= 0 or hazard <= 0.:
        return
    p = (rec[i_event] + 1.) / (r + 2.)
    like = p if event else 1. - p
    c = hazard * .5 / (hazard * .5 + (1. - hazard) * like)
    rec[i_event] *= 1. - c
    rec[i_risk] *= 1. - c


def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def binary_log_evidence(counts: Iterable[float]) -> float:
    pos, neg = counts
    if int(pos) != pos or int(neg) != neg:
        raise ValueError("Bernoulli counts must be integral")
    return _log_beta(1 + pos, 1 + neg) - _log_beta(1, 1)


@dataclass
class _PairCandidate:
    """Interaction cells: both-active vs exactly-one-active trials."""
    n_both: float = 0.0
    pos_both: float = 0.0
    n_rest: float = 0.0   # exactly one member active
    pos_rest: float = 0.0

    def decay(self, lam: float) -> None:
        self.n_both *= lam
        self.pos_both *= lam
        self.n_rest *= lam
        self.pos_rest *= lam

    @property
    def log_bf(self) -> float:
        pb, nb = self.pos_both, self.n_both - self.pos_both
        pr, nr = self.pos_rest, self.n_rest - self.pos_rest
        return (_log_beta(1. + pb, 1. + nb) + _log_beta(1. + pr, 1. + nr)
                - _log_beta(1. + pb + pr, 1. + nb + nr) - _log_beta(1., 1.))


@dataclass
class _PendingEvent:
    cue: int
    target: int
    age: int = 0
    salience: float = 1.   # 1 - predicted probability of the anchoring onset


class PosteriorGraph:
    """Online sparse graph; see module docstring for the exact semantics."""

    def __init__(self, n_inputs: int, n_actions: int, value_index: int, *,
                 seed: int = 0, gamma: float = .9, pair_alpha: float = .01,
                 pair_min_samples: int = 5, inhibit_threshold: float = .65,
                 explore: float = 2.0, max_age: int = 12,
                 change_hazard: float = .01, global_hazard: float = .001,
                 event_explore: float = .5, rollout_branch: int = 4,
                 value_scale: float = .5,
                 tie_seed: int | None = None) -> None:
        if n_inputs < 1 or n_actions < 2 or not 0 <= value_index < n_inputs:
            raise ValueError("invalid interface dimensions")
        if not 0 <= gamma <= 1 or not 0 < pair_alpha < 1:
            raise ValueError("invalid gamma or pair_alpha")
        self.n_inputs, self.n_actions, self.value_index = n_inputs, n_actions, value_index
        self.gamma, self.pair_alpha = gamma, pair_alpha
        self.pair_min_samples = pair_min_samples
        self.inhibit_threshold = inhibit_threshold
        self.credit_threshold = 0.5
        self.inhibitor_margin = 0.3
        self.block_strength = 0.6
        self.shrink = 2.0
        self.block_error = 0.25
        self.explore = explore
        self.max_age = max_age
        self.rng = _LCG(seed)
        self.tie_rng = _LCG(seed if tie_seed is None else tie_seed)
        # (context tuple, action, target) -> posterior
        self.factors: dict[tuple[tuple[int, ...], int, int], OutcomePosterior] = {}
        # Singleton-absent posteriors: ((cue,), action, target) -> posterior
        # over trials where the cue was NOT active. The present/absent
        # contrast gives each cue's causal strength in the noisy-OR.
        self.absent_factors: dict[tuple[tuple[int, ...], int, int], OutcomePosterior] = {}
        self.pair_candidates: dict[tuple[tuple[int, int], int, int], _PairCandidate] = {}
        self._cand_index: dict[tuple[int, int], list[tuple[tuple[int, int], int, int]]] = {}
        self.births: set[tuple[tuple[int, int], int, int]] = set()
        # (cue, target) -> [cancels|present, exposure|present, cancels|absent, exposure|absent]
        self.cancellation: dict[tuple[int, int], list[float]] = {}
        # (cue, target, age, action) -> [events, exposures]
        self.hazard: dict[tuple[int, int, int, int], list[float]] = {}
        # Context-specific timing cells, backed off to `hazard`.
        self._hazard_index: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        self.hazard_ctx: dict[tuple[int, int, int, int, tuple[int, ...]], list[float]] = {}
        self._hctx_index: dict[tuple[int, int, int, int], list] = {}
        self.pending_events: list[_PendingEvent] = []
        self._pending: tuple[tuple[float, ...], int] | None = None
        # Per-channel run-length statistics; only transient channels (mean
        # active run <= 3 steps) may anchor pending-event clocks. Persistent
        # channels (e.g. positions) correlate with delayed events through the
        # policy rather than through timing structure, and must not anchor.
        self._time = 0
        self._event_root: dict[tuple[int, int], list[float]] = {}
        self.context_factors: dict[tuple[tuple[int, ...], int, int], OutcomePosterior] = {}
        self.context_shrink = 2.0
        self.change_hazard = change_hazard
        self.global_hazard = global_hazard
        # Outlier rate of the global change detector (max surprise per step
        # = log(1/eps)) and contextual inference over stored regimes: stored
        # models (at most regime_max) compete by predictive evidence with
        # exponential memory regime_decay; a switch needs regime_bf nats.
        # Storing the current model is cheap insurance and is done once per
        # run of surprise as soon as P(change) exceeds regime_store, below
        # the point (1/2) where most of memory is already discounted; a
        # stored copy is only ever used if it later predicts better.
        self.global_eps = .05
        self.regime_decay = .8
        self.regime_bf = 3.
        self.regime_max = 4
        self.regime_store = .25
        self.event_explore = float(event_explore)
        # Continuation = expectation over the next observation: at most
        # `rollout_branch` uncertain channels are enumerated (a working-memory
        # budget; 2**k imagined successors), those with min(p, 1-p) > eps.
        self.rollout_branch = int(rollout_branch)
        self.rollout_eps = .05
        self.reward_cells: dict[tuple[int, ...], OutcomePosterior] = {}
        self._ctrl_lbf: dict[tuple[tuple[int, ...], int], float] = {}
        self._ctrl_total: dict[int, float] = {}
        self._ctrl_total_c: dict[int, float] = {}
        self._ctrl_c_alen = 0
        self._all_fk: set = set()
        self._fk_by_ak: dict = {}
        # Per-decision caches (pure functions of the current counts); they are
        # invalidated whenever counts or open windows change.
        self._memo: dict = {}
        self._cache: dict = {}
        self._onset_salience: dict[int, float] = {}
        self.last_global_change = 0.
        self._g_surprise = 0.
        self._g_applied = 0.
        self._g_pending = 0.
        self._regimes: list[dict] = []
        self._reg_evidence: list[float] = []
        self.regime_switches = 0
        self._rise: dict[int, int] = {}
        self._runlen: dict[int, float] = {}
        # Reference-point coding of the value channel (adaptation level):
        # z = 1/2 + (w - ref) / S, ref = median experienced value, S = twice
        # the largest deviation from it seen so far, at least `value_scale`
        # (semi-saturation: tiny early differences are not inflated). Stored
        # value statistics are re-expressed exactly when the frame changes.
        self.value_scale = float(value_scale)
        self._vlo: float | None = None
        self._vhi: float | None = None
        self._vref: float | None = None
        self._vcounts: dict[float, int] = {}
        # Attention (learned relevance). A channel is attended once it takes
        # part in a credible Granger-style dependency with *another* channel
        # (see _update_attention). Attended channels define a coarse level of
        # the model: an exact context is partially pooled with its attended
        # parent, so task-irrelevant inputs (noise, sticky distractors) do not
        # make every situation look novel, while exact tables still see every
        # channel. Monotone: attention once earned stays.
        self.attended: set[int] = set()
        # Coarse (attended-channels) cells and the exact contexts below them:
        # the spread among children decides how much an unvisited exact
        # context can differ from its coarse parent (partial pooling).
        self.coarse_factors: dict[tuple, OutcomePosterior] = {}
        self._kids: dict[tuple, set] = {}
        self.coarse_reward: dict[tuple[int, ...], OutcomePosterior] = {}
        self._reward_kids: dict[tuple[int, ...], set] = {}
        # Empirical-Bayes spread of exact contexts around coarse parents:
        # (coarse key, exact key) -> (weight, weighted squared deviation),
        # target -> [sum of weights, sum of deviations].
        self._tau_dev: dict[tuple, tuple[float, float]] = {}
        self._tau_sum: dict[int, list[float]] = {}
        # Dependence tables per (action, lag): (N[s][j] steps with j(t) = s,
        # Y[s][j] of them with j(t+1) on, {(cue, j): [cue on | s=0, both | s=0,
        # cue on | s=1, both | s=1]}); _dep_hist holds the last max_age cue sets.
        self._dep: dict[int, tuple] = {}
        self._dep_hist: list[list[int]] = []
        n_tests = n_inputs * (n_inputs - 1) * n_actions * max_age
        self._dep_threshold = log(max(2, n_tests) / pair_alpha)

    # ------------------------------------------------------------------ model
    def _factor(self, context: tuple[int, ...], action: int, target: int) -> OutcomePosterior:
        key = (tuple(sorted(context)), action, target)
        return self.factors.setdefault(key, OutcomePosterior())

    @staticmethod
    def _active(obs: tuple[float, ...]) -> list[int]:
        # In the reference-point frame "active" means above the middle: for
        # the value channel, better than the status quo. The tolerance keeps imagined states (predicted means) from
        # flipping a channel that sits exactly at .5 through rounding noise.
        return [i for i, u in enumerate(obs) if u > .5 + 1e-9]

    # ------------------------------------------------------- value range
    def _value_frame(self) -> tuple[float, float]:
        """(reference, scale) of the value representation.

        Reference = median experienced value (adaptation level / status quo),
        scale = max(2 * largest deviation from it, value_scale): the range
        seen so far is symmetric around the status quo and fills [0, 1].
        """
        ref = self._vref
        S = max(2. * max(self._vhi - ref, ref - self._vlo), self.value_scale)
        return ref, S

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
        # z = 1/2 + (w - ref) / S  ->  z' = a z + b, exactly
        a = round(S / S2, 12)
        b = round(.5 + (ref - ref2) / S2 - .5 * a, 12)
        self._affine_tables(self._current_tables(), a, b)
        for snap in self._regimes:
            self._affine_tables(snap, a, b)
        self._invalidate()

    def _affine_tables(self, T: dict, a: float, b: float) -> None:
        v = self.value_index
        for name in ("factors", "absent_factors", "context_factors", "coarse_factors"):
            for k, cell in T[name].items():
                if k[2] == v:
                    cell.affine(a, b)
        for name in ("reward_cells", "coarse_reward"):
            for cell in T[name].values():
                cell.affine(a, b)
        if v in T["_tau_sum"]:
            T["_tau_sum"][v][1] *= a * a
            tau_dev = T["_tau_dev"]
            for key, (w, d) in tau_dev.items():
                if key[0][2] == v:
                    tau_dev[key] = (w, d * a * a)

    # ------------------------------------------------------------ regimes
    def _current_tables(self) -> dict:
        return {name: getattr(self, name) for name in REGIME_TABLES}

    def _snapshot(self) -> dict:
        """Independent copy of every learned table (one latent regime)."""
        out = {}
        for name in _OP_TABLES:
            out[name] = {k: OutcomePosterior(c.kind, c.n, c.positive, c.mean, c.m2, c.surprise)
                         for k, c in getattr(self, name).items()}
        for name in _LIST_TABLES:
            out[name] = {k: list(v) for k, v in getattr(self, name).items()}
        for name in _SET_TABLES:
            out[name] = {k: set(v) for k, v in getattr(self, name).items()}
        for name in _PLAIN_TABLES:
            out[name] = dict(getattr(self, name))
        out["pair_candidates"] = {k: _PairCandidate(c.n_both, c.pos_both, c.n_rest, c.pos_rest)
                                  for k, c in self.pair_candidates.items()}
        out["births"] = set(self.births)
        return out

    def _load_tables(self, T: dict) -> None:
        for name, table in T.items():
            setattr(self, name, table)
        self._invalidate()

    @staticmethod
    def _transition_loglik(cf: dict, key, action: int, y, mask) -> float:
        """log p(y | exact context, action) under one regime's cells (fresh
        prior for channels that regime has not seen in this context)."""
        ll = 0.
        for target, outcome in enumerate(y):
            if not mask[target]:
                continue
            cell = cf.get((key, action, target))
            if cell is None or cell.n < 1.:
                ll += log(.5) if outcome in (0., 1.) else 0.
                continue
            ll += log(max(1e-12, cell.predictive_density(outcome)))
        return ll

    def _regime_step(self, key, action: int, y, mask, surprised: bool = True) -> bool:
        """Contextual inference over stored regimes: switch to a stored model
        whose recent predictive evidence beats the current one by
        `regime_bf` nats (exponentially weighted, `regime_decay` per step).
        Evaluated only on transitions the current model finds surprising
        (joint log LR of fresh prior vs current cells > 0); otherwise the
        evidence only decays. True if switched."""
        if not self._regimes:
            return False
        if not surprised:
            # Retrieval is driven by prediction error: a transition the
            # current model predicts gives no reason to prefer another one.
            self._reg_evidence = [d * self.regime_decay for d in self._reg_evidence]
            return False
        cur = self._transition_loglik(self.context_factors, key, action, y, mask)
        best, best_i = self.regime_bf, -1
        for i, snap in enumerate(self._regimes):
            d = self._reg_evidence[i] * self.regime_decay + (
                self._transition_loglik(snap["context_factors"], key, action, y, mask) - cur)
            self._reg_evidence[i] = d
            if d > best:
                best, best_i = d, i
        if best_i < 0:
            return False
        snap = self._regimes.pop(best_i)
        self._reg_evidence.pop(best_i)
        self._regimes.append(self._current_tables())
        self._reg_evidence.append(0.)
        self._reg_evidence = [0.] * len(self._regimes)
        self._load_tables(snap)
        self._g_surprise = self._g_applied = 0.
        self.regime_switches += 1
        return True

    def _new_regime(self) -> None:
        """Archive the current model before a change-point discount."""
        self._regimes.append(self._snapshot())
        self._reg_evidence.append(0.)
        if len(self._regimes) > self.regime_max:
            self._regimes.pop(0)
            self._reg_evidence.pop(0)

    def _normalise(self, x: tuple[float, ...]) -> tuple[float, ...]:
        v = self.value_index
        self._see_value(x[v])
        ref, S = self._value_frame()
        z = .5 + (x[v] - ref) / S
        # Rounding keeps range ends exactly 0/1 (Bernoulli cells) despite
        # floating-point error in the re-scaling.
        return x[:v] + (min(1., max(0., round(z, 9))),) + x[v + 1:]

    def _att(self, act: tuple[int, ...]) -> tuple[int, ...]:
        """Attended part of an active set (the key of the coarse level)."""
        A = self.attended
        return tuple(i for i in act if i in A)

    def _track_spread(self, ck, kid, target: int, cc: OutcomePosterior,
                      c: OutcomePosterior) -> None:
        """Running per-target sum of child-vs-parent squared deviations."""
        if cc.n <= 0. or c.n <= 0.:
            return
        pc = cc.positive / cc.n if cc.kind == "binary" else cc.mean
        tk = c.positive if c.kind == "binary" else c.mean * c.n
        wk = min(1., c.n)
        d = wk * ((tk + pc) / (c.n + 1.) - pc) ** 2
        old = self._tau_dev.get((ck, kid), (0., 0.))
        self._tau_dev[(ck, kid)] = (wk, d)
        acc = self._tau_sum.setdefault(target, [0., 0.])
        acc[0] += wk - old[0]
        acc[1] += d - old[1]

    def _update_attention(self, active: list[int], y: tuple[float, ...], action: int) -> None:
        """Granger-style dependence evidence cue(t+1-k) -> target(t+1), k = 1..max_age.

        Within each (action at t, lag k), and within each stratum of the
        target's own previous state j(t), a 2x2 Beta-Binomial Bayes factor
        compares "j(t+1) depends on i(t+1-k)" with "it does not"; the log
        BFs of the two strata add (rates free per stratum). Conditioning on
        j(t) removes spurious dependence between autocorrelated (sticky)
        processes; conditioning on the action removes dependence created by
        the agent's own policy; lags k > 1 let delayed causes (a cue whose
        food comes later) earn attention. A channel is attended once some
        partner shows such evidence with it, in either role; the threshold
        corrects for all ordered pairs, actions and lags. Attention only
        sets the coarse level of the model; exact tables still see every
        channel."""
        n = self.n_inputs
        cues = [i for i in active if i < n]
        prev = set(cues)
        on = [y[j] > .5 for j in range(n)]
        hist = self._dep_hist
        hist.insert(0, cues)
        del hist[self.max_age:]
        for k, old in enumerate(hist, 1):
            self._dep_count((action, k), old, prev, on)
        A = self.attended
        if len(A) == n:
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
        thr = self._dep_threshold
        for (i, j), cell in C.items():
            if i in A and j in A:
                continue
            lbf = 0.
            for sj in (0, 1):
                Ns, y1 = N[sj][j], Y[sj][j]
                n1, c = cell[2 * sj], cell[2 * sj + 1]
                if n1 < 1. or n1 >= Ns or y1 < 1. or y1 >= Ns:
                    continue
                b = y1 - c
                lbf += (_log_beta(1. + c, 1. + n1 - c) + _log_beta(1. + b, 1. + Ns - n1 - b)
                        - _log_beta(1. + y1, 1. + Ns - y1))
            if lbf > thr:
                A.add(i)
                A.add(j)

    def _excitation(self, x: tuple[float, ...], action: int, target: int,
                    act: tuple[int, ...] | None = None) -> tuple[float, float]:
        if act is None:
            act = tuple(self._active(x))
        ckey = ("exc", act, action, target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = self._cache[ckey] = self._excitation_uncached(act, action, target)
        return hit

    def _excitation_uncached(self, active: tuple[int, ...], action: int,
                             target: int) -> tuple[float, float]:
        """Specificity-ordered combination. Returns (mean, exploration n).

        Born pairs (pooled by counts) beat singletons. Singletons combine as
        a noisy-OR of independent cause strengths against a shared
        background: s_i = max(0, (p_i - b_i)/(1 - b_i)) with p_i the
        cue-present posterior and b_i the cue-absent posterior, and
        p = 1 - (1-b) * prod_i (1 - s_i), b the pooled cue-absent rate.
        A cue that merely correlates with the baseline (noise, position
        under the same action) has equal present/absent rates and zero
        strength, unlike a raw max over singletons which treats correlation
        as causation. Exploration n is the least-sampled active cue/action
        combination, so an untried cue keeps its novelty bonus even when a
        background cue is well sampled.
        """
        p_root, by_cue, cues = self._slice(action, target)
        factors = self.factors
        fs = []
        if by_cue:
            births = self.births
            born = [key for pair in combinations(active, 2)
                    if (key := (tuple(sorted(pair)), action, target)) in births]
            fs = [factors[k] for k in born if k in factors and factors[k].n > 0]
        # Per-cue terms (factor count, strength, background) are pure
        # functions of (cue, action, target) and memoised per decision.
        n_min = None
        missing = False
        strengths, backgrounds = [], []
        for i in active:
            c = cues.get(i)
            if c is None:
                c = cues[i] = self._cue_terms(i, action, target, p_root, by_cue)
            if c is False:
                missing = True
                continue
            n_i = c[0]
            if n_min is None or n_i < n_min:
                n_min = n_i
            if n_i == 0:
                continue
            strengths.append(c[1])
            backgrounds.append(c[2])
        if missing or n_min is None:
            n_min = 0.
        if fs:
            n = sum(f.n for f in fs)
            return sum(f.n * f.predictive_mean for f in fs) / n, min(n_min, n)
        if strengths:
            # Noisy-OR over enabling causes, then noisy-AND-NOT over
            # suppressive ones; each single-cue case reproduces p_i.
            # Suppressive singletons only lower the *baseline*: if any
            # enabling cause is active, cancelling its expected outcome
            # is the job of the discriminative inhibitor, not of a cue
            # whose negative marginal merely reflects the cause's absence.
            b = sum(backgrounds) / len(backgrounds)
            pos = [s_i for s_i in strengths if s_i > 0.]
            if pos:
                q = 1. - b
                for s_i in pos:
                    q *= 1. - s_i
                return min(1., max(0., 1. - q)), n_min
            gate = 1.
            for s_i in strengths:
                gate *= 1. + s_i
            return min(1., max(0., b * gate)), n_min
        return p_root, 0.0

    def _slice(self, action: int, target: int):
        """Per-decision data of one (action, target) slice: root rate, born
        pairs per member cue, and a lazily filled per-cue memo."""
        ckey = ("slice", action, target)
        hit = self._cache.get(ckey)
        if hit is None:
            root = self.factors.get(((), action, target))
            p_root = root.predictive_mean if root is not None and root.n > 0 else .5
            hit = self._cache[ckey] = (p_root, self._births_for(action, target), {})
        return hit

    def _cue_terms(self, i: int, action: int, target: int, p_root: float,
                   by_cue: dict[int, list]):
        """False if cue i has no factor in this slice, else (n, signed
        noisy-OR strength, background); strength/background only when n>0."""
        f = self.factors.get(((i,), action, target))
        if f is None:
            return False
        if f.n == 0:
            return (f.n, 0., 0.)
        # Residual rate: subtract the trials owned by born conjunctions
        # containing i (even when the partner is inactive now), so "cue i
        # without its partner" is priced by its own evidence rather than by
        # the pooled singleton.
        rn = f.n
        rsum = f.positive if f.kind == "binary" else f.mean * f.n
        for key in by_cue.get(i, ()):
            pf = self.factors.get(key)
            if pf is None or pf.n == 0:
                continue
            rn -= pf.n
            rsum -= pf.positive if pf.kind == "binary" else pf.mean * pf.n
        # Empirical-Bayes shrinkage toward the slice root: a cue with little
        # evidence predicts the base rate, not 0.5.
        m = self.shrink
        p_i = min(1., max(0., (max(0., rsum) + m * p_root) / (max(0., rn) + m)))
        absent = self.absent_factors.get(((i,), action, target))
        b_i = self._shrunk(absent, p_root)
        # Signed noisy-OR strength: positive when the cue raises the rate,
        # negative (suppressive) when it lowers it. Dropping the sign would
        # make "cue present -> outcome unlikely" unlearnable, which breaks
        # shock avoidance.
        if p_i >= b_i:
            s_i = (p_i - b_i) / max(1e-9, 1. - b_i)
        else:
            s_i = (p_i - b_i) / max(b_i, 1e-9)
        return (f.n, s_i, b_i)

    def _births_for(self, action: int, target: int) -> dict[int, list]:
        """Born pairs of this (action, target) slice per member cue, in the
        iteration order of `self.births` (so residual sums are bit-identical
        to a scan of the whole set). Invalidated with the per-decision cache
        and whenever a pair is born."""
        ckey = ("births", action, target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = {}
            for key in self.births:
                if key[1] == action and key[2] == target:
                    for i in key[0]:
                        hit.setdefault(i, []).append(key)
            self._cache[ckey] = hit
        return hit

    def _shrunk(self, f, prior: float) -> float:
        if f is None or f.n <= 0:
            return prior
        total = f.positive if f.kind == "binary" else f.mean * f.n
        return min(1., max(0., (total + self.shrink * prior) / (f.n + self.shrink)))

    def _strong_explainers(self, active, action: int, target: int, outcome: float) -> set[int]:
        root = self.factors.get(((), action, target))
        p_root = root.predictive_mean if root is not None and root.n > 0 else .5
        out = set()
        for j in active:
            f = self.factors.get(((j,), action, target))
            if f is None or f.n < 3.:
                continue
            absent = self.absent_factors.get(((j,), action, target))
            b = absent.predictive_mean if absent is not None and absent.n > 0 else p_root
            p = f.predictive_mean
            s = (p - b) / max(1e-9, 1. - b) if p >= b else (p - b) / max(b, 1e-9)
            if abs(s) >= self.block_strength and abs(outcome - p) < self.block_error:
                out.add(j)
        return out

    def _cancel_contrast(self, cue: int, target: int) -> float:
        rec = self.cancellation.get((cue, target))
        if rec is None:
            return 0.0
        cp, tp, ca, ta = rec
        if tp < 3. or ta < 3.:
            return 0.0
        return (cp + 1.) / (tp + 2.) - (ca + 1.) / (ta + 2.)

    def _established_inhibitors(self, active, target: int, excitors) -> set[int]:
        return {c for c in active if c not in excitors
                and self._cancel_contrast(c, target) >= self.inhibitor_margin}

    def _inhibition_probability(self, x: tuple[float, ...], target: int,
                                act: tuple[int, ...] | None = None) -> float:
        # Per-decision map cue -> credible cancellation contrast d > 0.
        ckey = ("inh", target)
        ds = self._cache.get(ckey)
        if ds is None:
            ds = self._cache[ckey] = {}
            for (i, t), rec in self.cancellation.items():
                if t != target:
                    continue
                cp, tp, ca, ta = rec
                if tp < 3. or ta < 3.:
                    continue  # no reliable present-vs-absent contrast
                d = (cp + 1.) / (tp + 2.) - (ca + 1.) / (ta + 2.)
                if d > 0:
                    ds[i] = d
        survival = 1.0
        if ds:
            for i in (act if act is not None else self._active(x)):
                d = ds.get(i)
                if d is not None:
                    survival *= 1. - d
        return 1. - survival

    def _timing_posterior(self, cue: int, target: int, lag: int, action: int,
                          ctx: tuple[int, ...]) -> tuple[float, float]:
        """Beta posterior (a, b) of a timing cell: exact context -> reduced
        context -> generic (cue, target, lag, action) cell.

        The reduced context keeps only channels that credibly modulate this
        timing cell (or are controllable, see _hsel); its pooled rate is the
        prior of the exact-context cell with an empirical-Bayes strength:
        the other exact contexts' evidence when they agree within binomial
        noise (e.g. they differ only in iid noise channels), the
        beta-binomial concentration when they disagree, and at least
        `context_shrink` pseudo-counts. Exact-context evidence therefore
        survives, while irrelevant channels no longer fragment the cell.
        """
        tk = ("tp", cue, target, lag, action, ctx)
        hit = self._cache.get(tk)
        if hit is not None:
            return hit
        g = self.hazard.get((cue, target, lag, action))
        ge, gr = (g[0], g[1]) if g is not None else (0., 0.)
        g_rate = (ge + 1.) / (gr + 2.)
        gkey = (cue, target, lag, action)
        irr, agg, groups = self._hsel(gkey)
        red = tuple(i for i in ctx if i not in irr) if irr else ctx
        r = agg.get(red)
        re_, rn = (r[0], r[1]) if r is not None else (0., 0.)
        c = self.hazard_ctx.get((cue, target, lag, action, ctx))
        ce, cr = (c[0], c[1]) if c is not None else (0., 0.)
        m = self.context_shrink
        r_rate = (re_ + m * g_rate) / (rn + m)
        s = max(m, min(self._red_concentration(gkey, red, groups), rn - cr))
        hit = self._cache[tk] = (1. + ce + s * r_rate, 1. + max(0., cr - ce) + s * (1. - r_rate))
        return hit

    def _reduced_ctx_counts(self, gkey, ctx) -> tuple[float, float]:
        """Counts of the timing cell conditioned only on the context channels
        that credibly modulate it. A channel is kept when splitting the
        generic cell's context children on it gives a dispersion above the
        BIC threshold for one extra rate (chi2 > max(3.84, log N));
        channels with no measurable effect (e.g. iid noise) are marginalized
        instead of fragmenting the cell."""
        irr, agg, _ = self._hsel(gkey)
        r = agg.get(tuple(i for i in ctx if i not in irr))
        return (r[0], r[1]) if r is not None else (0., 0.)

    def _hsel(self, gkey):
        """Per-decision analysis of one generic timing cell: the credibly
        relevant context channels, and its exact children grouped by their
        reduced key (summed counts and member cells)."""
        ck = ("hsel", gkey)
        hit = self._cache.get(ck)
        if hit is None:
            kids = [(key[4], c) for key in self._hctx_index.get(gkey, ())
                    for c in (self.hazard_ctx[key],) if c[1] > 1e-6]
            n_tot = sum(c[1] for _, c in kids)
            e_tot = sum(c[0] for _, c in kids)
            irr = set()
            if len(kids) > 1:
                p = (e_tot + .5) / (n_tot + 1.)
                q = p * (1. - p)
                thr = max(3.84, log(max(1., n_tot)))
                chans = set()
                for cx, _ in kids:
                    chans.update(cx)
                for j in chans:
                    if self._p_uncontrollable(j) < .5:
                        continue  # controllable channels keep their split
                    e1 = n1 = 0.
                    for cx, c in kids:
                        if j in cx:
                            e1 += c[0]
                            n1 += c[1]
                    n0 = n_tot - n1
                    if n1 < 1. or n0 < 1.:
                        continue  # untested: no evidence on one side
                    e0 = e_tot - e1
                    chi2 = ((e1 - n1 * p) ** 2 / n1 + (e0 - n0 * p) ** 2 / n0) / q
                    if chi2 <= thr:
                        irr.add(j)
            agg: dict = {}
            groups: dict = {}
            for cx, c in kids:
                red = tuple(i for i in cx if i not in irr)
                r = agg.get(red)
                if r is None:
                    agg[red] = [c[0], c[1]]
                    groups[red] = [c]
                else:
                    r[0] += c[0]
                    r[1] += c[1]
                    groups[red].append(c)
            hit = self._cache[ck] = (irr, agg, groups)
        return hit

    def _red_concentration(self, gkey, red, groups) -> float:
        """Beta-binomial concentration among the exact contexts that share
        this reduced key (method of moments; inf = no excess dispersion,
        i.e. the split channels look irrelevant here too)."""
        ck = ("kred", gkey, red)
        hit = self._cache.get(ck)
        if hit is not None:
            return hit
        kids = groups.get(red, ())
        kappa = float("inf")
        if len(kids) > 1:
            n_tot = sum(c[1] for c in kids)
            p = (sum(c[0] for c in kids) + .5) / (n_tot + 1.)
            x = sum((c[0] - c[1] * p) ** 2 / max(c[1], 1e-9) for c in kids) / (p * (1. - p))
            excess = x - (len(kids) - 1.)
            denom = sum(max(0., c[1] - 1.) for c in kids)
            if excess > 0. and denom > 0.:
                kappa = 1. / min(1., excess / denom) - 1.
        self._cache[ck] = kappa
        return kappa

    def _pending_for(self, target: int) -> list[_PendingEvent]:
        ckey = ("pend", target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = self._cache[ckey] = [p for p in self.pending_events if p.target == target]
        return hit

    def _hazard_boost(self, x: tuple[float, ...], action: int, target: int, depth: int,
                      act: tuple[int, ...] | None = None) -> float:
        """Timing lift: hazard above the slice root, not the raw rate."""
        pend = self._pending_for(target)
        if not pend:
            return 0.
        # Per decision: the open windows of this (target, action, depth)
        # whose generic timing cell has support (rec[1] >= 2), in order.
        ckey = ("hz", target, action, depth)
        cand = self._cache.get(ckey)
        if cand is None:
            cand = []
            for pending in pend:
                lag = pending.age + 1 + depth
                rec = self.hazard.get((pending.cue, target, lag, action))
                if rec is not None and rec[1] >= 2.:
                    cand.append((pending.cue, lag))
            self._cache[ckey] = cand
        if not cand:
            return 0.
        ev = self._event_root.get((target, action))
        p_root = (ev[0] + 1.) / (ev[1] + 2.) if ev is not None else .5
        h = 0.0
        act = act if act is not None else tuple(self._active(x))
        active = set(act)
        ctx = act
        for cue, lag in cand:
            if cue in active:
                continue  # one-step association is the excitation's job
            a, b = self._timing_posterior(cue, target, lag, action, ctx)
            rate = a / (a + b)
            sd = sqrt(a * b / ((a + b) ** 2 * (a + b + 1.)))
            if abs(rate - p_root) <= 2. * sd:
                continue  # not credibly different from baseline
            if rate >= p_root:
                lift = (rate - p_root) / max(1e-9, 1. - p_root)
            else:
                lift = (rate - p_root) / max(p_root, 1e-9)
            h = lift if abs(lift) > abs(h) else h
        return min(1., max(-1., h))

    def predict(self, obs: Iterable[float], action: int, target: int, *, depth: int = 0) -> float:
        x = tuple(float(v) for v in obs)
        self._validate_obs(x)
        if not 0 <= action < self.n_actions or not 0 <= target < self.n_inputs:
            raise ValueError("action/target out of range")
        self._invalidate()   # public entry: tables may have been edited directly
        return self._predict_core(x, action, target, depth)[0]

    def _predict_core(self, x: tuple[float, ...], action: int, target: int,
                      depth: int, act: tuple[int, ...] | None = None) -> tuple[float, float]:
        """Factor combination as a prior, exact observed context as data.

        The generalising combination (excitation x inhibition) is the prior
        mean of a hierarchical back-off: the exact active set observed under
        this action owns a posterior shrunk toward it with strength
        `context_shrink`. Correlated cues therefore cannot double-count in a
        context that has actually been visited, while unseen contexts are
        still priced by the factor graph (summation, transfer). Hazard
        timing is information outside the current observation and is
        applied last.
        """
        # Every term of the prediction depends on x only through its active
        # set, so imagined states that threshold to the same set share it.
        if act is None:
            act = tuple(self._active(x))
        mkey = (act, action, target, depth)
        hit = self._memo.get(mkey)
        if hit is not None:
            return hit
        out = self._predict_uncached(x, action, target, depth, act)
        self._memo[mkey] = out
        return out

    def _predict_uncached(self, x: tuple[float, ...], action: int, target: int,
                          depth: int, act: tuple[int, ...]) -> tuple[float, float]:
        exc, n_min = self._excitation(x, action, target, act)
        prior = min(1., max(0., exc * (1. - self._inhibition_probability(x, target, act))))
        # The factor prior is only as reliable as its least-supported active
        # cue under this action: a cue never seen with this action leaves
        # the prior (almost) weightless, so its uncertainty stays wide.
        m = self.context_shrink * n_min / (1. + n_min)
        sd_prior = sqrt(max(prior * (1. - prior), .25 / (1. + n_min)) / (m + 1.))
        if target == self.value_index:
            mb = self._model_based_value(x, action, depth, act)
            if mb is not None:
                # Model-based value (predicted next state -> learned reward)
                # combined with the factor prior by precision; its weight as a
                # prior grows with the support of the reward cell.
                p_mb, sd_mb, n_mb = mb
                w1, w2 = 1. / max(sd_prior ** 2, 1e-6), 1. / max(sd_mb ** 2, 1e-6)
                prior = (w1 * prior + w2 * p_mb) / (w1 + w2)
                sd_prior = sqrt(1. / (w1 + w2))
                m = max(m, self.context_shrink * n_mb / (1. + n_mb))
        active = act
        ak = self._att(act)
        if ak and ak != act:
            # Level 0.5: the attended (coarse) context, whose exact children
            # differ only in unattended channels. Partial pooling: the spread
            # tau^2 among visited children sets both the uncertainty of an
            # unvisited child and how strongly a visited one is shrunk
            # (m = within-cell variance / tau^2).
            ck = (ak, action, target)
            cc = self.coarse_factors.get(ck)
            if cc is not None and cc.n > 0:
                total = cc.positive if cc.kind == "binary" else cc.mean * cc.n
                pc = (total + m * prior) / (cc.n + m)
                if cc.kind == "binary":
                    within = cc.positive * (1. - pc) ** 2 + (cc.n - cc.positive) * pc ** 2
                else:
                    within = cc.m2 + cc.n * (cc.mean - pc) ** 2
                w_obs = (within + m * prior * (1. - prior)) / (cc.n + m)
                sdc2 = max(0., w_obs) / (cc.n + m + 1.)
                # Prior of the local spread: the spread of exact contexts
                # around their coarse parents over *all* parents for this
                # target (learned irrelevance generalises), itself with one
                # pseudo-child at maximal spread.
                W, D = self._tau_sum.get(target, (0., 0.))
                dev, k = (.25 + D) / (1. + W), 1.
                for kid in self._kids.get(ck, ()):
                    c = self.context_factors.get((kid, action, target))
                    if c is not None and c.n > 0.:
                        tk = c.positive if c.kind == "binary" else c.mean * c.n
                        mu = (tk + pc) / (c.n + 1.)
                        wk = min(1., c.n)
                        dev += wk * (mu - pc) ** 2
                        k += wk
                tau2 = dev / k
                prior = pc
                sd_prior = sqrt(sdc2 + tau2)
                m = min(50., max(w_obs, 1e-3) / tau2)
        ctx = self.context_factors.get((act, action, target))
        # Level 2 of the back-off: exact context *and* the set of open
        # expectation windows for this target (belief-state memory). A
        # context whose outcome depends on whether a timed event is due is
        # thereby split; level 1 acts as its prior.
        oc = self._open_cues(active, target, depth)
        mem = self.context_factors.get((act, action, target, oc))
        if ctx is not None and ctx.n > 0:
            total = ctx.positive if ctx.kind == "binary" else ctx.mean * ctx.n
            p = (total + m * prior) / (ctx.n + m)
            # Within-context spread: data moments plus m pseudo-observations
            # at Bernoulli-level spread around the prior (conservative).
            if ctx.kind == "binary":
                within = ctx.positive * (1. - p) ** 2 + (ctx.n - ctx.positive) * p ** 2
            else:
                within = ctx.m2 + ctx.n * (ctx.mean - p) ** 2
            v = (within + m * prior * (1. - prior)) / (ctx.n + m)
            sd = sqrt(max(0., v) / (ctx.n + m + 1.))
        else:
            p = prior
            # Unvisited context: prior spread around the prior mean, never
            # tighter than a Bernoulli at that mean with m pseudo-counts.
            sd = sd_prior
        if mem is not None and mem.n > 0:
            m2 = self.context_shrink
            total = mem.positive if mem.kind == "binary" else mem.mean * mem.n
            p1 = p
            p = (total + m2 * p1) / (mem.n + m2)
            if mem.kind == "binary":
                within = mem.positive * (1. - p) ** 2 + (mem.n - mem.positive) * p ** 2
            else:
                within = mem.m2 + mem.n * (mem.mean - p) ** 2
            v = (within + m2 * p1 * (1. - p1)) / (mem.n + m2)
            sd = min(sd, sqrt(max(0., v) / (mem.n + m2 + 1.)))
        # Controllability: if the evidence says this channel does not depend
        # on the action (Pavlovian), the action-pooled exact context carries
        # what was observed under any action, e.g. "cue+stop -> no food"
        # learned while sitting at base transfers to a rarely taken action.
        n_obs = mem.n if mem is not None else 0.
        ps = self._p_uncontrollable(target)
        if ps > 1e-3:
            pool = self.context_factors.get((act, -1, target))
            if pool is not None and pool.n > 0 and pool.kind == "binary":
                mc = self.context_shrink
                p_pool = (pool.positive + mc * p) / (pool.n + mc)
                sd_pool = sqrt(p_pool * (1. - p_pool) / (pool.n + mc + 1.))
                # ... refined by the action-pooled belief-state cell
                # (context + open windows), e.g. "precursor 2 steps ago ->
                # food now" observed under any action.
                pm = self.context_factors.get((act, -1, target, oc))
                if pm is not None and pm.n > 0 and pm.kind == "binary":
                    p_pool = (pm.positive + mc * p_pool) / (pm.n + mc)
                    sd_pool = min(sd_pool, sqrt(p_pool * (1. - p_pool) / (pm.n + mc + 1.)))
                    n_obs = max(n_obs, ps * pm.n)
                p = ps * p_pool + (1. - ps) * p
                sd = ps * sd_pool + (1. - ps) * sd
        h = self._hazard_boost(x, action, target, depth, act)
        if h != 0. and n_obs > 0:
            # The timing cells are the prior of a timed effect; the exact
            # belief-state cell (context + open windows) has observed it.
            h *= self.context_shrink / (self.context_shrink + n_obs)
        if h > 0.:
            p = 1. - (1. - p) * (1. - h)
        elif h < 0.:
            p = p * (1. + h)
        # Note: adding the timing-cell posterior sd to this bonus was
        # tried and removed: with renewal anchoring every action always has
        # some unsampled open cell, so it acts as uniform exploration noise.
        return min(1., max(0., p)), sd

    def _model_based_value(self, x: tuple[float, ...], action: int, depth: int,
                           act: tuple[int, ...] | None = None) -> tuple[float, float, float] | None:
        """E[w'] via the predicted next observation and the reward model R(y)."""
        v = self.value_index
        if act is None:
            act = tuple(self._active(x))
        yhat = [self._predict_core(x, action, j, depth, act)[0] if j != v else 0.
                for j in range(self.n_inputs)]
        A = self.attended
        key = tuple(j for j in range(self.n_inputs) if j != v and yhat[j] > .5 + 1e-9)
        cell = self.reward_cells.get(key)
        # Uncertainty about which state comes next only matters for channels
        # that carry structure (attended); before attention, for all.
        rel = A if A else range(self.n_inputs)
        q = 1.
        for j in rel:
            if j != v:
                q *= max(yhat[j], 1. - yhat[j])   # probability of the modal next state
        if (cell is None or cell.n < 1.) and A:
            # Unvisited exact state whose attended part is known: partial
            # pooling over its siblings (states differing only in
            # unattended channels).
            ckey = tuple(j for j in key if j in A)
            cc = self.coarse_reward.get(ckey)
            if cc is not None and cc.n >= 1.:
                mc = cc.mean
                dev, k = 1. / 12., 1.
                for kid in self._reward_kids.get(ckey, ()):
                    c = self.reward_cells.get(kid)
                    if c is not None and c.n > 0.:
                        wk = min(1., c.n)
                        dev += wk * ((c.mean * c.n + mc) / (c.n + 1.) - mc) ** 2
                        k += wk
                within = (cc.m2 + 1. / 12.) / (cc.n + 1.)
                return mc, sqrt(within + dev / k + (1. - q) * .25), 0.
        if cell is None or cell.n < 1.:
            return .5, sqrt(1. / 12. + (1. - q) * .25), 0.
        # Predictive spread of w given the next state (how much the state
        # determines the reward) plus uncertainty of its mean: a reward cell
        # with mixed outcomes is a weak predictor for this (x, a).
        if cell.kind == "binary":
            mean = cell.predictive_mean
            within = mean * (1. - mean)
        else:
            mean = cell.mean
            within = (cell.m2 + 1. / 12.) / (cell.n + 1.)
        var = within * (1. + 1. / (cell.n + 1.))
        return mean, sqrt(var + (1. - q) * .25), cell.n

    def _p_uncontrollable(self, target: int) -> float:
        # Coarse (attended context + open windows) evidence once attention
        # exists; exact-context evidence before.
        t = self._ctrl_total_c.get(target) if self.attended else None
        if t is None:
            t = self._ctrl_total.get(target)
        if t is None:
            return 0.
        return 1. / (1. + exp(min(50., max(-50., t))))

    def _rebuild_ctrl_c(self) -> None:
        """Attention changed, hence the coarse partition: recompute every
        coarse controllability BF from the exact belief-state cells."""
        self._ctrl_c_alen = len(self.attended)
        for k in [k for k in self._ctrl_lbf if k[0] == "c"]:
            del self._ctrl_lbf[k]
        self._ctrl_total_c.clear()
        by_ak: dict = {}
        self._fk_by_ak = by_ak
        if not self.attended:
            return
        for fk in self._all_fk:
            by_ak.setdefault(self._att(fk), set()).add(fk)
        todo = set()
        for k in self.context_factors:
            if len(k) == 4 and k[1] >= 0:
                todo.add((self._att(k[0]), k[2], k[3]))
        for ak, target, oc in todo:
            self._update_controllability_c(ak, target, oc)

    def _update_controllability_c(self, key: tuple[int, ...], target: int, oc) -> None:
        """Controllability BF of one attended (coarse) context and set of
        open windows, from the summed exact belief-state cells of its
        members: unattended channels (noise) do not split the comparison
        into many small contexts that each pay an Occam penalty, and the
        open windows remove the confound of a policy that acts on memory."""
        cf = self.context_factors
        ns, ps = [], []
        binary = True
        kids = self._fk_by_ak.get(key, ())
        for a in range(self.n_actions):
            n = pos = 0.
            for fk in kids:
                c = cf.get((fk, a, target, oc))
                if c is not None and c.n > 0:
                    if c.kind != "binary":
                        binary = False
                        break
                    n += c.n
                    pos += c.positive
            if not binary:
                break
            if n > 0:
                ns.append(n)
                ps.append(pos)
        if not binary:
            lbf = 50.
        elif len(ns) < 2:
            lbf = 0.
        else:
            lbf = sum(_log_beta(1. + k, 1. + n - k) for k, n in zip(ps, ns))
            K, N = sum(ps), sum(ns)
            lbf -= _log_beta(1. + K, 1. + N - K)
        ck = ("c", key, target, oc)
        old = self._ctrl_lbf.get(ck, 0.)
        self._ctrl_lbf[ck] = lbf
        self._ctrl_total_c[target] = self._ctrl_total_c.get(target, 0.) + lbf - old

    def _update_controllability(self, key: tuple[int, ...], target: int) -> None:
        """Within-context Beta-Binomial BF: action-specific vs shared rate.

        Summed over contexts it is the evidence that the action controls
        this channel; conditioning on the exact context removes the policy
        confound (which actions happen to be taken where).
        """
        cells = [self.context_factors.get((key, a, target)) for a in range(self.n_actions)]
        cells = [c for c in cells if c is not None and c.n > 0]
        if any(c.kind != "binary" for c in cells):
            lbf = 50.  # continuous channels are treated as controllable
        elif len(cells) < 2:
            lbf = 0.
        else:
            lbf = sum(_log_beta(1. + c.positive, 1. + c.n - c.positive) for c in cells)
            K = sum(c.positive for c in cells)
            N = sum(c.n for c in cells)
            lbf -= _log_beta(1. + K, 1. + N - K)
        old = self._ctrl_lbf.get((key, target), 0.)
        self._ctrl_lbf[(key, target)] = lbf
        self._ctrl_total[target] = self._ctrl_total.get(target, 0.) + lbf - old

    def _global_change_step(self, active, action: int, y, mask) -> None:
        """Environment-level change point shared by every cell.

        First, contextual inference over stored regimes (_regime_step): if a
        previously learned model of the world explains recent transitions
        clearly better, it is reinstated (savings on serial reversals).

        Otherwise, evidence for a new change point is the joint likelihood
        ratio of this transition under a fresh prior vs the exact-context
        cells, LR = prod_j p0(y_j)/p(y_j), made outlier-robust: with
        probability `global_eps` a transition is something the model does not
        capture (an unmodelled window, noise), so one step contributes at most
        log(1/eps). Surprise accumulates over a run of steps (CUSUM, as in the
        per-cell discount); c = P(change since the run began) with prior
        hazard H. Memory is discounted to (1 - c) once per run (not compounded
        per step), so a confirmed change wipes the old model (archived as a
        regime first) while an isolated surprise costs about eps. In quiet
        periods only the prior hazard applies, accumulated and applied lazily.
        """
        H = self.global_hazard
        if H <= 0.:
            self.last_global_change = 0.
            return
        key = tuple(active)
        log_lr = 0.
        for target, outcome in enumerate(y):
            if not mask[target]:
                continue
            cell = self.context_factors.get((key, action, target))
            if cell is None or cell.n < 1.:
                continue
            p0 = .5 if (cell.kind == "binary" and outcome in (0., 1.)) else 1.
            log_lr += log(p0) - log(max(1e-12, cell.predictive_density(outcome)))
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
            self._g_surprise = 0.
            self._g_applied = 0.
            self._g_pending += -log(1. - c)
            if self._g_pending > .01:
                self._discount_all(exp(-self._g_pending))
                self._g_pending = 0.

    def _invalidate(self) -> None:
        """Drop the per-decision caches (pure functions of the counts)."""
        self._memo = {}
        self._cache = {}

    def _discount_all(self, lam: float) -> None:
        """Scale every count by lam (evidence surviving a global change point)."""
        self._cache = {}
        # Inlined OutcomePosterior.decay / PairCandidate.decay (same
        # arithmetic; this loop touches every cell).
        for table in (self.factors, self.absent_factors, self.context_factors,
                      self.reward_cells):
            for f in table.values():
                f.n *= lam
                f.positive *= lam
                f.m2 *= lam
        for cand in self.pair_candidates.values():
            cand.n_both *= lam
            cand.pos_both *= lam
            cand.n_rest *= lam
            cand.pos_rest *= lam
        for rec in self.cancellation.values():
            for k in range(4):
                rec[k] *= lam
        for table in (self.hazard, self._event_root, self.hazard_ctx):
            for rec in table.values():
                rec[0] *= lam
                rec[1] *= lam

    def _absorb(self, post: OutcomePosterior, y: float) -> None:
        post.change_discount(y, self.change_hazard)
        post.update(y)

    def _open_cues(self, active, target: int, depth: int = 0) -> tuple[tuple[int, int], ...]:
        """Belief-state memory: informative open windows with their exact age.

        Each entry is (cue, tau + depth): the age advances along an
        imagined rollout, so "one step after the cue" and "two steps after"
        are different states. Only windows whose cue has a credibly
        non-baseline hazard for this target at some lag enter the key;
        uninformative windows (noise, positions) would only fragment it.
        """
        ckey = ("open", tuple(active), target, depth)
        hit = self._cache.get(ckey)
        if hit is not None:
            return hit
        act = set(active)
        out = tuple(sorted({(p.cue, p.age + depth) for p in self._pending_for(target)
                            if p.cue not in act
                            and self._lag_informative(p.cue, target, p.age + 1 + depth)}))
        self._cache[ckey] = out
        return out

    def _lag_informative(self, cue: int, target: int, lag: int) -> bool:
        """Credible non-baseline timing of (cue -> target) at this lag, under
        any action. Without such evidence an open window is not part of the
        belief state: it promises nothing beyond the baseline."""
        ckey = ("linf", cue, target, lag)
        hit = self._cache.get(ckey)
        if hit is not None:
            return hit
        hit = False
        for a in range(self.n_actions):
            rec = self.hazard.get((cue, target, lag, a))
            if rec is None or rec[1] < 2.:
                continue
            e, r = rec
            ev = self._event_root.get((target, a))
            p_root = (ev[0] + 1.) / (ev[1] + 2.) if ev is not None else .5
            rate = (e + 1.) / (r + 2.)
            if abs(rate - p_root) > 2. * sqrt(rate * (1. - rate) / (r + 3.)):
                hit = True
                break
        self._cache[ckey] = hit
        return hit

    def _window_informative(self, cue: int, target: int) -> bool:
        ckey = ("inf", cue, target)
        hit = self._cache.get(ckey)
        if hit is None:
            hit = self._cache[ckey] = self._window_informative_uncached(cue, target)
        return hit

    def _window_informative_uncached(self, cue: int, target: int) -> bool:
        for key in self._hazard_index.get((cue, target), ()):
            e, r = self.hazard[key]
            if r < 2.:
                continue
            ev = self._event_root.get((target, key[3]))
            p_root = (ev[0] + 1.) / (ev[1] + 2.) if ev is not None else .5
            rate = (e + 1.) / (r + 2.)
            if abs(rate - p_root) > 2. * sqrt(rate * (1. - rate) / (r + 3.)):
                return True
        return False

    def _event_sd(self, x: tuple[float, ...], action: int, depth: int) -> float:
        """Novelty of an open, unexplained expectation on the value channel.

        For every open window whose anchoring cue is not active now, whose
        effect on the value is not yet located at any lag, and whose lag
        tau + 1 + depth lies within the planning horizon, the score is
        salience / (1 + n), n = tests of the exact (cue, lag, action,
        context) combination and salience = 1 - P(onset) before the cue
        appeared (a predicted / self-caused onset adds nothing beyond its
        predictor's window). Deterministic in the counts, hence
        permutation-equivariant.
        """
        v = self.value_index
        act = tuple(self._active(x))
        active = set(act)
        ctx = act
        out = 0.
        for pending in self._pending_for(v):
            if pending.cue in active:
                continue
            lag = pending.age + 1 + depth
            if self._window_informative(pending.cue, v):
                # A cue whose effect on the value is already located at some
                # lag is exploited through the hazard model, not explored.
                continue
            # Tests count per reduced context (_hsel): a channel shown not
            # to modulate this timing cell does not make a test novel.
            r = self._reduced_ctx_counts((pending.cue, v, lag, action), ctx)[1]
            out = max(out, pending.salience / (1. + r))
        return out

    def _value_predict(self, x: tuple[float, ...], action: int, depth: int) -> tuple[float, float]:
        return self._predict_core(x, action, self.value_index, depth)

    def predict_next(self, obs: Iterable[float], action: int, *, depth: int = 0) -> tuple[float, ...]:
        x = tuple(float(v) for v in obs)
        self._invalidate()   # public entry: tables may have been edited directly
        return self._predict_next(x, action, depth)

    def _predict_next(self, x: tuple[float, ...], action: int, depth: int) -> tuple[float, ...]:
        return tuple(self._predict_core(x, action, j, depth)[0] for j in range(self.n_inputs))

    def _best_next(self, s: tuple[float, ...]) -> float:
        ke = self.event_explore
        best = 0.
        for a in range(self.n_actions):
            p, _ = self._value_predict(s, a, depth=1)
            if ke > 0.:
                p += ke * self._event_sd(s, a, 1) * (1. - p)
            best = max(best, p)
        return best

    def _expected_continuation(self, nxt: tuple[float, ...]) -> float:
        """E over the next observation of max_a' value: the agent will see
        the outcome before it acts again. Channels whose predicted state is
        uncertain (min(p, 1-p) > eps; attended ones once attention exists)
        are enumerated as independent Bernoullis, the rest set to their
        modal value; at most K channels, the most uncertain."""
        v = self.value_index
        A = self.attended
        unc = [(min(p, 1. - p), j) for j, p in enumerate(nxt)
               if j != v and min(p, 1. - p) > self.rollout_eps and (not A or j in A)]
        unc.sort(key=lambda u: -u[0])
        U = [j for _, j in unc[:self.rollout_branch]]
        base = [(1. if p > .5 else 0.) if j != v else p for j, p in enumerate(nxt)]
        total = 0.
        for bits in range(1 << len(U)):
            s = list(base)
            w = 1.
            for i, j in enumerate(U):
                on = (bits >> i) & 1
                s[j] = float(on)
                w *= nxt[j] if on else 1. - nxt[j]
            if w < 1e-6:
                continue
            total += w * self._best_next(tuple(s))
        return total

    def _q2(self, obs: tuple[float, ...], action: int) -> float:
        """Two-step optimistic Bellman backup over predicted transitions.

        Q(b,a) = UCB(r|b,a) + gamma * E_{b'}[max_a' (E[r|b',a'] + event bonus)],
        the expectation over the predicted next observation (the agent sees
        it before acting again; see _expected_continuation), with
        UCB = posterior mean + sqrt((explore * sd)^2 + event bonus^2), sd the
        posterior sd of the hierarchical (factor prior -> exact context)
        prediction, at the root only:
        imagined successor states use plain posterior means, otherwise the
        bonus of an imagined state double-counts uncertainty and biases
        the agent toward actions whose successors are merely unfamiliar.
        """
        immediate, sd_now = self._value_predict(obs, action, depth=0)
        nxt = self._predict_next(obs, action, 0)
        # The imagined successor is a state, not a reward: the value
        # channel's predicted *mean* is already the immediate term, and
        # thresholding it (e.g. .503 -> "above the middle of the range")
        # would place the rollout in a context that never occurs.
        v = self.value_index
        nxt = nxt[:v] + (min(nxt[v], .5),) + nxt[v + 1:]
        ke = self.event_explore
        continuation = self._expected_continuation(nxt)
        # One optimism budget per transition. The value posterior sd (which
        # for an unseen predicted state is the novel-state prior) and the
        # open (cue, tau) expectation are two independent uncertainties about
        # the *same* reward: their variances add, their sds do not. Adding
        # the sds double-counts a transition that is both novel and an open
        # test; max() would drop a genuine second source. In the rollout
        # only the event term exists (imagined states use plain means), so
        # there is nothing to merge there.
        bonus = self.explore * sd_now
        if ke > 0.:
            e = ke * self._event_sd(obs, action, 0) * (1. - immediate)
            bonus = sqrt(bonus * bonus + e * e)
        return immediate + bonus + self.gamma * continuation

    # ---------------------------------------------------------------- control
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
        scores = [self._q2(x, a) for a in range(self.n_actions)]
        best = max(scores)
        tied = [a for a, v in enumerate(scores) if abs(v - best) < 1e-12]
        action = max(tied, key=lambda a: noise[a])
        self._pending = (x, action)
        return action

    def _validate_obs(self, x: tuple[float, ...]) -> None:
        if len(x) != self.n_inputs or any(not 0 <= v <= 1 for v in x):
            raise ValueError("observation width/range mismatch")

    # ----------------------------------------------------------------- update
    def _update_hazard(self, y: tuple[float, ...], action: int,
                       prev_active: set[int], mask: tuple[bool, ...]) -> None:
        self._time += 1
        ctx = tuple(sorted(prev_active))
        for target in range(self.n_inputs):
            if mask[target]:
                rec = self._event_root.setdefault((target, action), [0., 0.])
                _count_change_discount(rec, 0, 1, y[target] > .5, self.change_hazard)
                rec[1] += 1.
                rec[0] += float(y[target] > .5)
        kept = []
        for pending in self.pending_events:
            age = pending.age + 1
            hkey = (pending.cue, pending.target, age, action)
            rec = self.hazard.get(hkey)
            if rec is None:
                rec = self.hazard[hkey] = [0., 0.]
                self._hazard_index.setdefault((pending.cue, pending.target), []).append(hkey)
            ckey = (pending.cue, pending.target, age, action, ctx)
            crec = self.hazard_ctx.get(ckey)
            if crec is None:
                crec = self.hazard_ctx[ckey] = [0., 0.]
                self._hctx_index.setdefault(hkey, []).append(ckey)
            if mask[pending.target]:
                ev = y[pending.target] > .5
                _count_change_discount(rec, 0, 1, ev, self.change_hazard)
                _count_change_discount(crec, 0, 1, ev, self.change_hazard)
            # Associability = surprise of the anchoring onset: a predicted
            # (e.g. self-caused) onset carries no timing information beyond
            # its own predictor, so its window adds (almost) no evidence.
            w = pending.salience
            rec[1] += w
            crec[1] += w
            if not mask[pending.target]:
                continue  # right-censored at this age: exposure counted, episode ends
            if y[pending.target] > .5:
                rec[0] += w  # event consumes this expectation
                crec[0] += w
            elif age < self.max_age:
                kept.append(_PendingEvent(pending.cue, pending.target, age, pending.salience))
        self.pending_events = kept
        # Anchor new expectation windows on rising edges of transient channels.
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
            # Renewal: a new occurrence right-censors any still-open window
            # of the same cue (its exposures are already counted), so lags
            # are always measured from the most recent occurrence.
            self.pending_events = [p for p in self.pending_events if p.cue != cue]
            for target in range(self.n_inputs):
                if target == cue or target in active_now or not mask[target]:
                    continue
                self.pending_events.append(_PendingEvent(cue, target, 0,
                                                         self._onset_salience.get(cue, 1.)))

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
        # Salience of each onset = 1 - its prediction before learning: an
        # onset that was itself predicted (e.g. food after its cue) adds no
        # timing information beyond the window of its predictor.
        self._onset_salience = {
            j: 1. - self._predict_core(x, action, j, 0)[0]
            for j in range(self.n_inputs)
            if mask[j] and y[j] > .5 and x[j] <= .5}
        self._global_change_step(active, action, y, mask)
        att_key = self._att(tuple(active))
        full_key = tuple(active)
        # Membership of exact contexts in attended (coarse) contexts, for
        # the coarse controllability evidence (rebuilt when attention grows).
        self._all_fk.add(full_key)
        if len(self.attended) != self._ctrl_c_alen:
            self._rebuild_ctrl_c()
        if self.attended:
            self._fk_by_ak.setdefault(att_key, set()).add(full_key)
        # Global multiple-comparison penalty across all pair/action candidates.
        n_pairs_total = self.n_inputs * (self.n_inputs - 1) // 2 * self.n_actions
        birth_threshold = log(max(1, n_pairs_total) / self.pair_alpha)
        for target, outcome in enumerate(y):
            if not mask[target]:
                continue
            root = self._factor((), action, target)
            ctx_key = (full_key, action, target)
            self._absorb(self.context_factors.setdefault(ctx_key, OutcomePosterior()), outcome)
            self._absorb(self.context_factors.setdefault((full_key, -1, target),
                                                         OutcomePosterior()), outcome)
            self._update_controllability(full_key, target)
            if att_key:
                ck = (att_key, action, target)
                cc = self.coarse_factors.setdefault(ck, OutcomePosterior())
                self._absorb(cc, outcome)
                self._kids.setdefault(ck, set()).add(full_key)
                if att_key != full_key:
                    self._track_spread(ck, full_key, target, cc,
                                       self.context_factors[ctx_key])
            oc = self._open_cues(active, target)
            mem_key = ctx_key + (oc,)
            self._absorb(self.context_factors.setdefault(mem_key, OutcomePosterior()), outcome)
            self._absorb(self.context_factors.setdefault((full_key, -1, target, oc),
                                                         OutcomePosterior()), outcome)
            if self.attended:
                self._update_controllability_c(att_key, target, oc)
            # Interaction evidence for pair candidates (binary channels only).
            # A pair earns a birth only if trials with BOTH members active
            # have a different outcome rate than trials with exactly one
            # active; evidence is a 2x2 Beta-Binomial log Bayes factor
            # charged against the global number of possible pairs.
            if outcome in (0., 1.):
                active_set = set(active)
                for key in self._cand_index.get((action, target), []):
                    if key in self.births:
                        continue
                    cand = self.pair_candidates[key]
                    i, k = key[0]
                    in_i, in_k = i in active_set, k in active_set
                    if in_i != in_k:
                        cell = [cand.pos_rest, cand.n_rest]
                        _count_change_discount(cell, 0, 1, outcome == 1., self.change_hazard)
                        cand.pos_rest, cand.n_rest = cell
                        cand.n_rest += 1.
                        cand.pos_rest += float(outcome == 1.)
                for pair in combinations(active, 2):
                    key = (tuple(sorted(pair)), action, target)
                    if key in self.births:
                        continue
                    cand = self.pair_candidates.get(key)
                    if cand is None:
                        cand = self.pair_candidates[key] = _PairCandidate()
                        self._cand_index.setdefault((action, target), []).append(key)
                    cell = [cand.pos_both, cand.n_both]
                    _count_change_discount(cell, 0, 1, outcome == 1., self.change_hazard)
                    cand.pos_both, cand.n_both = cell
                    cand.n_both += 1.
                    cand.pos_both += float(outcome == 1.)
                    if cand.n_both >= self.pair_min_samples and cand.log_bf > birth_threshold:
                        self.births.add(key)
                        self._cache.pop(("births", action, target), None)
                        self._cache.pop(("slice", action, target), None)
                        born = self._factor(key[0], action, target)
                        if born.n == 0:
                            born.kind = "binary"
                            born.n, born.positive = cand.n_both, cand.pos_both
                            born.mean = cand.pos_both / cand.n_both if cand.n_both else 0.
                            born.m2 = born.mean * (1. - born.mean) * cand.n_both
            # Excitor support within this action slice.
            excitors = {i for i in active
                        if ((i,), action, target) in self.factors
                        and self.factors[((i,), action, target)].n >= 2.
                        and self.factors[((i,), action, target)].predictive_mean >= self.inhibit_threshold}
            expected = max((self.factors[((i,), action, target)].predictive_mean
                            for i in active
                            if ((i,), action, target) in self.factors
                            and self.factors[((i,), action, target)].n >= 2.), default=.0)
            # A cancel event requires a non-excitor cue present to take the
            # blame; if the excitor is alone, the outcome is evidence against
            # the excitation belief itself and must update it normally.
            # Two thresholds: the 2x2 cancellation contrast is credited at a
            # weaker expectation (so evidence can accumulate while excitors
            # are still being established), but excitor *preservation* only
            # happens when a statistically established inhibitor is active.
            # Innocent bystander cues (noise channels) never qualify, so an
            # inflated singleton cannot dodge its own negative evidence.
            credit_cancel = (outcome == 0. and expected >= self.credit_threshold
                             and bool(set(active) - excitors))
            credit_deliver = outcome == 1. and expected >= self.credit_threshold
            inhibitors = self._established_inhibitors(active, target, excitors)
            cancel_event = (outcome == 0. and expected >= self.inhibit_threshold
                            and bool(inhibitors))
            # Discriminative 2x2 cancellation counts over all sensory cues.
            if credit_cancel or credit_deliver:
                for cue in range(self.n_inputs):
                    if credit_cancel and cue in excitors:
                        continue  # an excitor is never blamed for its own cancellation
                    rec = self.cancellation.setdefault((cue, target), [0., 0., 0., 0.])
                    if cue in active:
                        _count_change_discount(rec, 0, 1, credit_cancel, self.change_hazard)
                    else:
                        _count_change_discount(rec, 2, 3, credit_cancel, self.change_hazard)
                    if cue in active:
                        rec[1] += 1.
                        if credit_cancel:
                            rec[0] += 1.
                    else:
                        rec[3] += 1.
                        if credit_cancel:
                            rec[2] += 1.
            # Kamin blocking: a trial whose outcome is already explained by
            # another strongly-established cue (small prediction error) is
            # not credited to the remaining cues. This keeps policy-induced
            # co-occurrence (e.g. "center+left" mostly seen together with
            # the left precursor) from inflating innocent singletons.
            strong = self._strong_explainers(active, action, target, outcome)
            self._absorb(root, outcome)
            active_set = set(active)
            for cue in active:
                if cancel_event and cue in excitors:
                    continue
                if strong - {cue}:
                    continue
                self._absorb(self._factor((cue,), action, target), outcome)
            for cue in range(self.n_inputs):
                if cue in active_set or (cancel_event and cue in excitors):
                    continue
                akey = ((cue,), action, target)
                if akey in self.factors or akey in self.absent_factors:
                    self._absorb(self.absent_factors.setdefault(akey, OutcomePosterior()), outcome)
            for key in list(self.births):
                if key[1] != action or key[2] != target or not set(key[0]).issubset(active):
                    continue
                if cancel_event and excitors.intersection(key[0]):
                    continue
                self._absorb(self._factor(key[0], action, target), outcome)
        v = self.value_index
        if mask[v]:
            A = self.attended
            rkey = tuple(j for j in range(self.n_inputs) if j != v and y[j] > .5)
            self._absorb(self.reward_cells.setdefault(rkey, OutcomePosterior()), y[v])
            ckey = tuple(j for j in rkey if j in A)
            self._absorb(self.coarse_reward.setdefault(ckey, OutcomePosterior()), y[v])
            self._reward_kids.setdefault(ckey, set()).add(rkey)
        self._update_hazard(y, action, set(active), mask)
        self._invalidate()
        self._update_attention(active, y, action)
        self._invalidate()
        self._pending = None


class _LCG:
    def __init__(self, seed: int):
        self.state = int(seed) & 0x7fffffff

    def random(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7fffffff
        return self.state / 0x80000000
