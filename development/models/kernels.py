"""Fused single-threaded numerical kernels for the online predictors.

No fast-math: likelihood, posterior and sufficient-statistic computations must
remain deterministic when the same tuple is resumed on the same snapshot.
"""
import numpy as np
from math import lgamma, log, pi
from numba import njit


@njit(cache=True)
def hmm_params(ntr, non, ntot, alpha, sticky, shrink, tr, em):
    actions, k, _ = ntr.shape
    for i in range(k):
        pool = np.empty(k)
        total = 0.
        for j in range(k):
            q = 0.
            for a in range(actions):
                q += ntr[a, i, j]
            q += alpha
            if i == j:
                q += sticky
            pool[j] = q
            total += q
        for a in range(actions):
            n = 0.
            for j in range(k):
                n += ntr[a, i, j]
            for j in range(k):
                tr[a, i, j] = (ntr[a, i, j] + shrink * (pool[j] / total)) / (n + shrink)
        for d in range(non.shape[1]):
            em[i, d] = (non[i, d] + .5) / (ntot[i] + 1.)


@njit(cache=True)
def hmm_forward(b, t, em, read):
    prior = np.zeros(len(b))
    for j in range(len(b)):
        for i in range(len(b)):
            prior[j] += b[i] * t[i, j]
    out = np.zeros(em.shape[1] if read else 0)
    for d in range(len(out)):
        for j in range(len(b)):
            out[d] += prior[j] * em[j, d]
    return prior, out


@njit(cache=True)
def hmm_step(y, a, prior, b, tr, em, h_b, h_a, h_y, ix, age, ntr, non, ntot):
    size, k = h_b.shape
    m = em.shape[1]
    h_b[ix] = b
    h_a[ix] = a
    h_y[ix] = y
    likelihood = np.empty(k)
    score = 0.
    for j in range(k):
        prob = 1.
        for d in range(m):
            prob *= em[j, d] if y[d] > .5 else 1. - em[j, d]
        likelihood[j] = prob
        score += prior[j] * prob
    for j in range(k):
        b[j] = prior[j] * likelihood[j] / score
    if age + 1 < size:
        return np.log(score)
    beta = np.ones(k)
    new = np.empty(k)
    msg = np.empty(k)
    for offset in range(size):
        pos = (ix - offset) % size
        for j in range(k):
            prob = 1.
            for d in range(m):
                prob *= em[j, d] if h_y[pos, d] > .5 else 1. - em[j, d]
            msg[j] = prob * beta[j]
        action = h_a[pos]
        norm = 0.
        if offset < size - 1:
            for i in range(k):
                v = 0.
                for j in range(k):
                    v += tr[action, i, j] * msg[j]
                new[i] = v
                norm += v
            for i in range(k):
                beta[i] = new[i] / norm
        else:
            xi = np.empty((k, k))
            for i in range(k):
                for j in range(k):
                    v = h_b[pos, i] * tr[action, i, j] * msg[j]
                    xi[i, j] = v
                    norm += v
            for j in range(k):
                g = 0.
                for i in range(k):
                    q = xi[i, j] / norm
                    g += q
                    ntr[action, i, j] += q
                ntot[j] += g
                for d in range(m):
                    non[j, d] += g * h_y[pos, d]
    return np.log(score)


@njit(inline='always')
def _read_regime(out, q1, q0, weights, mu, v, center, cells, pg, fast, deficit):
    J, S = cells.shape
    for j in range(J):
        if not fast[j]:
            continue
        add = center[j]
        for p in range(v.shape[1]):
            add += mu[j, p] * v[j, p]
        w = weights[j, S]
        pa = np.float32(max(.01, min(.99, add)))
        num = w * pa
        den = w
        for s in range(S):
            c = cells[j, s]
            a, b = q1[j, c], q0[j, c]
            p = (a + pg[j, s]) / (a + b + np.float32(1.))
            w = weights[j, s]
            num += w * p
            den += w
        q = num / den
        rho = deficit[j]
        out[j] = q if rho == 0. else (1. - rho) * q + rho * pa


@njit(cache=True)
def factor_read(q1, q0, weights, mu, v, center, cells, pg, kk, fast, used, deficit):
    """Read live regimes separately and evaluate the common birth row once."""
    n = q1.shape[0]
    J = cells.shape[0]
    out = np.full((n, kk, J), .5)
    birth = np.full(J, .5)
    have_birth = False
    for i in range(n):
        for z in range(min(kk, used[i])):
            _read_regime(out[i, z], q1[i, z], q0[i, z], weights[i, z], mu[i, z],
                         v, center, cells, pg, fast, deficit)
        z = used[i]
        if z < kk:
            # Slot reset and resampling preserve identical prior statistics in
            # each never-used row. Particle and place effects are applied later.
            if not have_birth:
                _read_regime(birth, q1[i, z], q0[i, z], weights[i, z], mu[i, z],
                             v, center, cells, pg, fast, deficit)
                have_birth = True
            out[i, z] = birth
    return out


@njit(cache=True)
def factor_update(q1, q0, ev, weights, mu, v, center, cells, pg, sampled, y, fast, base, credit, deficit):
    """Score then count one observed transition in each sampled regime."""
    J, S = cells.shape
    for i in range(q1.shape[0]):
        z = sampled[i]
        for j in range(J):
            if not fast[j]:
                continue
            add = center[j]
            for p in range(v.shape[1]):
                add += mu[i, z, j, p] * v[j, p]
            pred = np.float32(max(.01, min(.99, add)))
            sign = y[j]
            bg = np.float32(base[i, j])
            wt = np.float32(credit[i, j])
            pred = bg + (np.float32(1.) - bg) * pred
            ev[i, z, j, S] += np.log(pred if sign else np.float32(1.) - pred)
            peak = ev[i, z, j, S]
            for s in range(S):
                c = cells[j, s]
                a, b = q1[i, z, j, c], q0[i, z, j, c]
                prob = (a + pg[j, s]) / (a + b + np.float32(1.))
                prob = bg + (np.float32(1.) - bg) * prob
                prob = min(np.float32(1. - 1e-6), max(np.float32(1e-6), prob))
                table_like = prob if sign else np.float32(1.) - prob
                rho = np.float32(deficit[j])
                if rho > 0.:
                    add_like = pred if sign else np.float32(1.) - pred
                    prob_like = (np.float32(1.) - rho) * table_like + rho * add_like
                    count = wt * (np.float32(1.) - rho) * table_like / max(np.float32(1e-12), prob_like)
                else:
                    prob_like, count = table_like, wt
                ev[i, z, j, s] += np.log(prob_like)
                peak = max(peak, ev[i, z, j, s])
                if sign:
                    q1[i, z, j, c] += count
                else:
                    q0[i, z, j, c] += count
            for s in range(S + 1):
                weights[i, z, j, s] = np.exp(ev[i, z, j, s] - peak)


@njit(cache=True)
def factor_gaussian(mu, cov, shared, shared_cov, v, center, sampled, y, fast, noise, credit):
    """Sparse-vector rank-one conditioning in place (no particle covariance copies)."""
    d = v.shape[1]
    for j in range(v.shape[0]):
        if not fast[j]:
            continue
        active = np.nonzero(v[j])[0]
        sv = np.empty(d)
        tgt = (1. if y[j] else 0.) - center[j]
        avg = np.mean(credit[:, j])
        for i in range(len(sampled) + 1):
            wt = avg if i == len(sampled) else credit[i, j]
            if wt <= 0.:
                continue
            if i == len(sampled):
                W, V = shared[j], shared_cov[j]
            else:
                z = sampled[i]
                W, V = mu[i, z, j], cov[i, z, j]
            var = noise / wt
            err = tgt
            for p in range(d):
                s = 0.
                for r in active:
                    s += V[p, r] * v[j, r]
                sv[p] = s
                var += s * v[j, p]
            if i < len(sampled):
                for r in active:
                    err -= W[r] * v[j, r]
            for p in range(d):
                g = sv[p] / var
                W[p] += g * err
                for q in range(d):
                    V[p, q] -= g * sv[q]


@njit(cache=True)
def systematic_ancestry(w, offset):
    """Ordered ancestors in one CDF pass (the left inverse at exact ties)."""
    n = len(w)
    idx = np.empty(n, np.int64)
    i, c = 0, w[0]
    for d in range(n):
        u = (offset + d) / n
        while i < n - 1 and c < u:
            i += 1
            c += w[i]
        idx[d] = i
    return idx


@njit(cache=True)
def copy_ancestors(bank, idx, live):
    """Gather ordered ancestors in place, skipping identity and pristine tails.

    If idx[d] > d, that source cannot need a left-ancestor copy (monotonicity);
    read it before its right-ancestor copy. The reverse order handles the other
    direction. Layout is [particle, regime, contiguous statistics].
    """
    for d in range(len(idx)):
        s = idx[d]
        if s > d:
            for z in range(live):
                bank[d, z, :] = bank[s, z, :]
    for d in range(len(idx) - 1, -1, -1):
        s = idx[d]
        if s < d:
            for z in range(live):
                bank[d, z, :] = bank[s, z, :]


@njit(cache=True)
def binary_dependency_scores(moments):
    """Max log Jeffreys/Krichevsky–Trofimov evidence over retained lags.

    This is only a scheduling statistic: serially correlated pairs violate its
    iid null. The matched online joint/separate comparison validates proposals.
    """
    lags, targets, degree = moments[0].shape
    best = np.full((targets, degree), -np.inf)
    norm = log(pi)
    for j in range(targets):
        for p in range(degree):
            for d in range(lags):
                n = moments[0, d, j, p]
                x = moments[1, d, j, p]
                y = moments[2, d, j, p]
                both = moments[3, d, j, p]
                l0 = lgamma(y + .5) + lgamma(n - y + .5) - lgamma(n + 1.)
                l1 = lgamma(both + .5) + lgamma(x - both + .5) - lgamma(x + 1.)
                l1 += lgamma(y - both + .5) + lgamma(n - x - y + both + .5) - lgamma(n - x + 1.)
                best[j, p] = max(best[j, p], l1 - l0 - norm)
    return best


@njit(cache=True)
def reset_factor_slot(q1, q0, ev, mu, cov, j, pos, off, affected, parent, live, s0):
    """Erase one replaced parent's statistics; other factor members are intact."""
    for i in range(q1.shape[0]):
        for r in range(live):
            for k in range(len(affected)):
                m, sub = affected[k], parent[k]
                for c in range(off[m], off[m + 1]):
                    q1[i, r, j, c] = 0.
                    q0[i, r, j, c] = 0.
                ev[i, r, j, m] = ev[i, r, j, sub]
            mu[i, r, j, pos] = 0.
            for u in range(mu.shape[-1]):
                cov[i, r, j, pos, u] = 0.
                cov[i, r, j, u, pos] = 0.
            cov[i, r, j, pos, pos] = s0


@njit(cache=True)
def choose_slots(cev, sources, source_slots, lag_slots, own_lag, slow, cond, margin):
    """Scan fixed-width local proposals, preserving row-major first ties."""
    jn, width, nl = cev.shape
    ks = source_slots.shape[1]
    update = np.full((jn, 3), -1, np.int64)
    for j in range(jn):
        best = -np.inf
        worst = np.inf
        wi, wl, wp = 0, 0, 0
        for p in range(ks):
            u = source_slots[j, p]
            old = -np.inf
            if u >= 0:
                for i in range(width):
                    if sources[j, i] == u:
                        old = cev[j, i, lag_slots[j, p]]
                        break
            if old < worst:
                worst, wp = old, p
        for i in range(width):
            u = sources[j, i]
            if cond and slow[u]:
                continue
            for l in range(nl):
                if own_lag[l] and u == j:
                    continue
                admitted = True
                for p in range(ks):
                    if u == source_slots[j, p] and (cond or l == lag_slots[j, p]):
                        admitted = False
                        break
                v = cev[j, i, l]
                if admitted and v > best:
                    best, wi, wl = v, i, l
        if best > margin and best > worst + margin:
            update[j, 0] = wp
            update[j, 1] = sources[j, wi]
            update[j, 2] = wl
    return update


@njit(cache=True)
def copy_ancestor_rows(bank, idx, used):
    """As copy_ancestors, but each move only touches its known non-prior rows."""
    width = bank.shape[2]
    for d in range(len(idx)):
        s = idx[d]
        if s > d:
            live = max(used[d], used[s])
            for j in range(live):
                for p in range(width):
                    bank[d, j, p] = bank[s, j, p]
    for d in range(len(idx)-1,-1,-1):
        s = idx[d]
        if s < d:
            live = max(used[d], used[s])
            for j in range(live):
                for p in range(width):
                    bank[d, j, p] = bank[s, j, p]
