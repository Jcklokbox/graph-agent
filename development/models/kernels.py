"""Fused single-threaded numerical kernels for the online predictors.

No fast-math: likelihood, posterior and sufficient-statistic computations must
remain deterministic when the same tuple is resumed on the same snapshot.
"""
import numpy as np
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


@njit(cache=True)
def factor_read(q1, q0, weights, mu, v, center, cells, pg, kk, fast, used):
    """Read all live regimes; sufficient statistics are contiguous by member."""
    n = q1.shape[0]
    J, S = cells.shape
    out = np.full((n, kk, J), .5)
    for i in range(n):
        for z in range(min(kk, used[i] + 1)):
            for j in range(J):
                if not fast[j]:
                    continue
                add = center[j]
                for p in range(v.shape[1]):
                    add += mu[i, z, j, p] * v[j, p]
                w = weights[i, z, j, S]
                num = w * np.float32(max(.01, min(.99, add)))
                den = w
                for s in range(S):
                    c = cells[j, s]
                    a, b = q1[i, z, j, c], q0[i, z, j, c]
                    p = (a + pg[j, s]) / (a + b + np.float32(1.))
                    w = weights[i, z, j, s]
                    num += w * p
                    den += w
                out[i, z, j] = num / den
    return out


@njit(cache=True)
def factor_update(q1, q0, ev, weights, mu, v, center, cells, pg, sampled, y, fast):
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
            ev[i, z, j, S] += np.log(pred if sign else np.float32(1.) - pred)
            peak = ev[i, z, j, S]
            for s in range(S):
                c = cells[j, s]
                a, b = q1[i, z, j, c], q0[i, z, j, c]
                prob = (a + pg[j, s]) / (a + b + np.float32(1.))
                prob = min(np.float32(1. - 1e-6), max(np.float32(1e-6), prob))
                ev[i, z, j, s] += np.log(prob if sign else np.float32(1.) - prob)
                peak = max(peak, ev[i, z, j, s])
                if sign:
                    q1[i, z, j, c] += 1.
                else:
                    q0[i, z, j, c] += 1.
            for s in range(S + 1):
                weights[i, z, j, s] = np.exp(ev[i, z, j, s] - peak)


@njit(cache=True)
def factor_gaussian(mu, cov, shared, shared_cov, v, center, sampled, y, fast, noise):
    """Sparse-vector rank-one conditioning in place (no particle covariance copies)."""
    d = v.shape[1]
    for j in range(v.shape[0]):
        if not fast[j]:
            continue
        active = np.nonzero(v[j])[0]
        sv = np.empty(d)
        tgt = (1. if y[j] else 0.) - center[j]
        for i in range(len(sampled) + 1):
            if i == len(sampled):
                W, V = shared[j], shared_cov[j]
            else:
                z = sampled[i]
                W, V = mu[i, z, j], cov[i, z, j]
            var = noise
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
