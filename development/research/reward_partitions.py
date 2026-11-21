"""Experimental reward readout: one comparison and one predictive law per arrival.

This is not an agent or a qualification entry point. The caller supplies
real transitions in a single control/reward scope. No statistics are written
back to the controller.
"""
from dataclasses import dataclass
from math import comb, lgamma, log

import numpy as np

from models.factor_information import log_marginal
from models.factor_joint_reward import JointEvidence
from models.factor_projection import factor_current, finite_marginal
from models.factor_reward import conditional_draws


@dataclass(frozen=True)
class Partition:
    groups: int
    cue: int = -1
    moderator: int = -1

    def classify(self, features, onsets):
        if self.groups == 1:
            return np.zeros(len(features), dtype=int)
        if self.groups == 2:
            return features[:, self.cue].astype(int)
        return np.where(onsets[:, self.cue], features[:, self.moderator], 2).astype(int)


POOLED = Partition(1)


@dataclass(frozen=True)
class Law:
    partition: Partition
    evidence: float
    counts: np.ndarray
    sums: np.ndarray
    # Pairs of group and arrival-row index of its first new observation.
    changes: tuple[tuple[int, int], ...] = ()

    def atoms(self, group):
        bins = self.counts.shape[-1]
        centers = (np.arange(bins) + .5) / bins
        alpha = self.counts[group] + 1. / bins
        mass = alpha.sum(axis=-1)
        reps = (self.sums[group] + centers / bins) / alpha
        return alpha / mass[..., None], reps, mass


def _row_partition(group):
    """A label-invariant signature of which observations belong together."""
    labels = {}
    return tuple(labels.setdefault(int(g), len(labels)) for g in group)


def _time_splits(group, atom, counts, gamma, norm, contextual):
    """Choose a subset of groups to split. Their visit boundaries are local."""
    groups = len(counts)
    stationary = gamma[counts].sum(axis=-1) - norm[counts.sum(axis=-1)]
    single, raw = [], []
    for g, total in enumerate(counts):
        rows = np.flatnonzero(group == g)
        n = len(rows)
        if n < 2:
            continue
        # A cut partitions the same rows. Silence creates no observations.
        before = np.cumsum(atom[rows, None] == np.arange(len(total)), axis=0)[:-1]
        after = total - before
        size = np.arange(1, n)
        evidence = (gamma[before].sum(axis=1) + gamma[after].sum(axis=1)
                    - norm[size] - norm[n - size] - stationary[g] - log(n - 1))
        rank = np.argsort(-evidence, kind='stable')
        cut = rank[0]
        raw.append((float(evidence[cut]), g, int(rows[cut + 1]), after[cut]))
        if groups + 1 > 3:
            single.append(raw[-1])
            continue
        for cut in rank:
            if evidence[cut] <= log(groups):
                break
            current = rows[cut + 1:]
            labels = group.copy()
            labels[current] = groups
            if _row_partition(labels) not in contextual:
                single.append((float(evidence[cut]), g, int(current[0]), after[cut]))
                break

    best, chosen = 0., ()
    if single:
        candidate = max(single, key=lambda x: x[0])
        score = candidate[0] - log(groups)
        if score > best:
            best, chosen = score, (candidate,)

    # Two or more cuts imply at least four groups. With the current static
    # basis (at most three groups), none can alias a static partition.
    raw.sort(key=lambda x: x[0], reverse=True)
    for k in range(2, len(raw) + 1):
        score = sum(c[0] for c in raw[:k]) - log(comb(groups, k))
        if score > best:
            best, chosen = score, raw[:k]
    return best, tuple(sorted(chosen, key=lambda x: x[1]))


def fit(features, onsets, rewards, unary, pairs, *, bins=16, allow_change=False):
    """Select a partition of the same rows by context and optionally time.

    A temporal candidate splits any subset of context groups into past and
    current visits. Its extra code length names the subset at fixed size and
    one boundary among each chosen group's visits. Both segments count
    towards evidence; only the current segment predicts. Exact aliases of
    static candidates are excluded. On ties prefer no changes, then pooled,
    unary and joint in that order. Family costs include unsupported proposals;
    an unobserved group cannot win against pooled.
    """
    rewards = np.asarray(rewards, dtype=float)
    if not np.all(np.isfinite(rewards) & (0 <= rewards) & (rewards <= 1)):
        raise ValueError('expected finite real rewards in [0, 1]')
    atom = np.minimum(bins - 1, (rewards * bins).astype(int))
    prior = 1. / bins
    baseline = log_marginal(np.bincount(atom, minlength=bins), prior)
    families = ([POOLED], [Partition(2, i) for i in unary], [Partition(3, i, j) for i, j in pairs])
    if allow_change:
        gamma = np.asarray([lgamma(n + prior) - lgamma(prior) for n in range(len(atom) + 1)])
        norm = np.asarray([lgamma(n + 1.) for n in range(len(atom) + 1)])
    candidates = []
    for family in families:
        if not family:
            continue
        cost = log(len(family))
        for partition in family:
            group = partition.classify(features, onsets)
            key = group * bins + atom
            counts = np.bincount(key, minlength=partition.groups * bins).reshape(partition.groups, bins)
            if partition == POOLED or counts.any(axis=1).all():
                candidates.append((partition, group, key, counts, cost))
    contextual = {_row_partition(g) for p, g, _, _, _ in candidates if p != POOLED} if allow_change else set()
    best = None
    for partition, group, key, counts, cost in candidates:
        score = 0. if partition == POOLED else sum(log_marginal(row, prior) for row in counts) - baseline - cost
        cuts = ()
        if allow_change:
            gain, cuts = _time_splits(group, atom, counts, gamma, norm, contextual)
            score += gain
        if best is None or (score, -len(cuts)) > (best.evidence, -len(best.changes)):
            sums = np.bincount(key, weights=rewards, minlength=counts.size).reshape(counts.shape)
            for _, g, start, suffix in cuts:
                keep = group[start:] == g
                counts[g] = suffix
                sums[g] = np.bincount(atom[start:][keep], weights=rewards[start:][keep], minlength=bins)
            best = Law(partition, score, counts, sums, tuple((g, start) for _, g, start, _ in cuts))
    return best


def _arrival_rows(head, transitions):
    by_arrival = [[] for _ in range(head.C)]
    rows = head.replay if transitions is None else transitions
    for _, _, reward, arrival, features, onsets in rows:
        by_arrival[int(head.code(arrival))].append((reward, features, onsets))
    empty = np.empty((0, head.F - 1), dtype=bool)
    for rows in by_arrival:
        if rows:
            reward, features, onsets = zip(*rows)
            yield np.asarray(features, bool), np.asarray(onsets, bool), np.asarray(reward)
        else:
            yield empty, empty, np.empty(0)


def fit_arrivals(head, transitions=None, *, allow_change=False):
    """Use the exposed candidate basis and caller-selected same-scope rows."""
    unary, pairs, _ = JointEvidence.joint_basis(head)
    return [fit(*rows, unary, pairs, bins=head.B, allow_change=allow_change)
            for rows in _arrival_rows(head, transitions)]


def refit_arrivals(head, laws, transitions):
    """Estimate each selected partition on other rows, without reselection.

    For the diagnostic only: keep the selection evidence, including when
    the forecast rows never visited a branch (then it has only the prior).
    """
    bins = head.B
    out = []
    for law, (features, onsets, rewards) in zip(laws, _arrival_rows(head, transitions), strict=True):
        if law.changes:
            raise ValueError('refitting a time split requires a shared time index')
        atom = np.minimum(bins - 1, (rewards * bins).astype(int))
        key = law.partition.classify(features, onsets) * bins + atom
        counts = np.bincount(key, minlength=law.counts.size).reshape(law.counts.shape)
        sums = np.bincount(key, weights=rewards, minlength=counts.size).reshape(counts.shape)
        out.append(Law(law.partition, law.evidence, counts, sums))
    return out


def fixed_values(factor, x, uniforms, laws):
    """Two-step diagnostic with the same motion and sensory projection as v60.

    Only the selected laws supply predictions and single-atom updates. No
    legacy unary/context fallback, soft refinement, or exploration is scored.
    This does not run or replace the live policy.
    """
    head = factor.reward
    discount = getattr(head, 'discount', 1.)
    b = factor._bits(x)
    active = head.slots >= 0
    start = int(head.code(b))
    trans = head.all_transitions()
    states = np.flatnonzero(np.any(trans[:, start, :] > 0, axis=0))
    marks = (states[:, None] & head.binary[active]) != 0
    relevant = np.array(sorted({j for law in laws for j in (law.partition.cue, law.partition.moderator)
                                if 0 <= j < head.J}), dtype=int)
    quiet = not b[~factor.slow].any()
    projections = factor_current(factor, x) if relevant.size else [None] * factor.A

    def query(sensory, hist):
        features = np.asarray([head.input(s, hist)[1:] for s in sensory], dtype=bool)
        onsets = np.asarray([head.phase(s, hist) for s in sensory], dtype=bool)
        groups = [law.partition.classify(features, onsets) for law in laws]
        return groups, [law.atoms(group) for law, group in zip(laws, groups)]

    root_group, root_law = query([b], factor.hist)
    scores = np.zeros(factor.A)
    for a, projection in enumerate(projections):
        if projection is None:
            branches, weights = np.zeros((1, head.J)), np.ones(1)
        else:
            mass, p, slow_p, _ = projection
            if len(relevant) <= 3:
                values, weights = finite_marginal(mass, p, factor.slow, slow_p, quiet, relevant)
                branches = np.zeros((len(values), head.J))
                branches[:, relevant] = values
            else:
                branches = conditional_draws(mass, p, factor.slow, slow_p, quiet, uniforms[:, :head.J + 3])
                weights = np.full(len(branches), 1. / len(branches))
        for c, mark in zip(states, marks):
            arrivals = branches.copy()
            arrivals[:, head.slots[active]] = mark
            next_group, next_law = query(arrivals, (*factor.hist, b))
            mean = np.stack([(p * r).sum(-1) for p, r, _ in next_law], axis=1)
            p0, amounts, _ = root_law[c]
            motion = trans[:, c, :]
            base = mean @ motion.T
            same_group = next_group[c] == root_group[c][0]
            delta = (amounts - mean[:, c, None]) / (1. + next_law[c][2][:, None])
            delta *= same_group[:, None]
            inner = np.clip(base[:, :, None] + motion[None, :, c, None] * delta[:, None, :], 0., 1.)
            scores[a] += trans[a, start, c] * (p0[0] @ amounts[0] + discount * (weights @ (inner.max(axis=1) @ p0[0])))
    return scores
