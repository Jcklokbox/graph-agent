"""Read-only one-cue ablation for the fixed v60 forecast on exact branches."""
import numpy as np

from models.factor_projection import factor_current, finite_marginal
from models.factor_rollout import provisional_values


def _tail_allocation(reach, prob, rank):
    ordered = reach[rank]
    allocated = np.empty_like(reach)
    allocated[rank] = np.clip(prob - (np.cumsum(ordered) - ordered), 0., ordered)
    return allocated


def coupling_bounds(reach, prob, value):
    """Exact range given both marginals; value[z, c] is conditional utility."""
    contrast = value[1] - value[0]
    rank = np.argsort(contrast, kind='stable')
    base = reach @ value[0]
    return tuple(float(base + w @ contrast) for w in (
        _tail_allocation(reach, prob, rank), _tail_allocation(reach, prob, rank[::-1])))


def pattern_bounds(reach, mass, value):
    """Conservative bounds on all-pattern coupling, with both marginals fixed."""
    columns = reach > 0.
    rows = mass > 0.
    values = value[rows][:, columns]
    reach, mass = reach[columns], mass[rows]
    lo, hi = reach @ values.min(0), reach @ values.max(0)
    for anchor in values:
        # Its contribution is fixed by the arrival marginal. Bound the
        # residual by the sensory marginal; this relaxes row capacities.
        base = reach @ anchor
        delta = values - anchor
        lo = max(lo, base + mass @ delta.min(1))
        hi = min(hi, base + mass @ delta.max(1))
    return float(lo), float(hi)


def matched_one_mass(reach, prob, counts):
    """Calibrate empirical log odds to both marginals on witnessed arrivals.

    Require two visits per used arrival, at least two such arrivals, and both
    cue values in their data. Leave all other arrivals independent. These
    support checks do not make this a consistent estimate under hidden drift.
    """
    base = reach * prob
    seen = (counts.sum(1) >= 2) & (reach > 0.)
    if seen.sum() < 2 or not counts[seen].sum(0).all() or not 0. < prob < 1.:
        return base, False
    # Adding a common offset leaves the within-support odds ratios unchanged.
    odds = np.log(counts[seen, 1] + .5) - np.log(counts[seen, 0] + .5)
    weights = reach[seen]
    logit = np.log(prob) - np.log1p(-prob)
    lo, hi = logit - odds.max(), logit - odds.min()
    for _ in range(50):
        mid = (lo + hi) / 2
        cond = 1. / (1. + np.exp(-(odds + mid)))
        if weights @ cond < prob * weights.sum():
            lo = mid
        else:
            hi = mid
    base[seen] = weights / (1. + np.exp(-(odds + (lo + hi) / 2)))
    return base, True


def fixed_surfaces(factor, x, uniforms):
    """Independent enumeration; check the product result against shipped v60."""
    h = factor.reward
    b = factor._bits(x)
    X, O = h.input(b, factor.hist)[1:][None, :], h.phase(b, factor.hist)[None, :]
    rule = h.rule()
    splits = h.signed_rules(rule)
    active = h.slots >= 0
    known = rule[0][rule[1]]
    relevant = np.unique(np.r_[known[known < h.J], np.array([m for _, m, l, _, _, _ in splits if l == 0], dtype=int)])
    if not active.any() or not 1 <= len(relevant) <= 3:
        return None
    if np.isin(relevant, h.slots[active]).any():
        raise ValueError('expected residual cue columns')
    trans = h.all_transitions()
    reach = trans[:, int(h.code(b))]
    states = np.flatnonzero(np.any(reach > 0, axis=0))
    reach = reach[:, states]
    root_mean, root_prob, *_ = h.likelihood(rule, X)
    root_atoms = rule[3]
    if splits:
        root_mean, root_prob, root_atoms, _, root_group = h.context_moment(
            splits, X, O, root_mean, root_prob, root_atoms)
        root_atoms, root_group = root_atoms[0], root_group[0]
    root_mean, root_prob, root_atoms, _ = h.refine_root(root_prob[0], root_atoms, rule, X[0], O[0])
    hist = (*factor.hist, b)
    marks = (states[:, None] & h.binary[active]) != 0
    product, surfaces = [], []
    for a, (w, p, slow_p, _) in enumerate(factor_current(factor, x)):
        bits, mass = finite_marginal(w, p, factor.slow, slow_p, not b[~factor.slow].any(), relevant)
        value = np.empty((len(bits), len(states)))
        for row, z in enumerate(bits):
            for col, (c, mark) in enumerate(zip(states, marks)):
                arrival = np.zeros(h.J)
                arrival[relevant], arrival[h.slots[active]] = z, mark
                X1, O1 = h.input(arrival, hist)[1:][None, :], h.phase(arrival, hist)[None, :]
                mean, prob, lik, norm = h.likelihood(rule, X1.astype(bool))
                if splits:
                    mean, _, _, count, group = h.context_moment(splits, X1, O1, mean, prob, rule[3])
                diff = root_atoms[c] - mean[0, c]
                delta = lik[0, c] / (norm[0, c] + lik[0, c]) * diff
                if splits:
                    if group[0, c] != root_group[c]:
                        delta = np.zeros_like(delta)
                    elif group[0, c] > 0:
                        delta = diff / (1. + count[0, c])
                q = (trans[:, c] @ mean[0])[:, None] + trans[:, c, c, None] * delta
                value[row, col] = root_mean[c] + root_prob[c] @ np.clip(q.max(0), 0., 1.)
        product.append(mass @ value @ reach[a])
        surfaces.append((bits, mass, value))
    fixed = provisional_values(factor, x, uniforms, False)
    np.testing.assert_allclose(product, fixed, rtol=0., atol=2e-14)
    return b, states, relevant, reach, fixed, surfaces


def compare_cues(factor, x, uniforms):
    """Change one coupling at a time, preserving the full residual marginal."""
    surface = fixed_surfaces(factor, x, uniforms)
    if surface is None:
        return None
    b, states, relevant, reach, fixed, surfaces = surface
    h = factor.reward
    counts = np.zeros((h.A, h.C, len(relevant), 2), dtype=int)
    for origin, action, _, arrival, *_ in h.replay:
        if np.array_equal(origin, b):
            counts[action, int(h.code(arrival)), np.arange(len(relevant)), arrival[relevant].astype(int)] += 1
    envelope = [pattern_bounds(reach[a], mass, value) for a, (_, mass, value) in enumerate(surfaces)]
    reports = {}
    for j, cue in enumerate(relevant):
        bounds, adjusted, supported = [], [], []
        for a, (bits, mass, value) in enumerate(surfaces):
            p = float(mass[bits[:, j]].sum())
            means = np.zeros((2, len(states)))
            for z in (False, True):
                weight = mass[bits[:, j] == z]
                if weight.sum() > 0.:
                    means[int(z)] = weight @ value[bits[:, j] == z] / weight.sum()
            bounds.append(coupling_bounds(reach[a], p, means))
            one, ok = matched_one_mass(reach[a], p, counts[a, states, j])
            adjusted.append(float((reach[a] - one) @ means[0] + one @ means[1]))
            supported.append(ok)
        reports[int(cue)] = dict(bounds=bounds, adjusted=adjusted, supported=supported,
                                 counts=counts[:, states, j].tolist())
    return dict(fixed=fixed.tolist(), states=states.tolist(), cues=reports, envelope=envelope)
