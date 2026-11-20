"""Bayesian mixture over context depths along a path (context-tree weighting).

Setting: a binary outcome y is predicted in a nested sequence of contexts
s_0 (root) ⊃ s_1 ⊃ ... ⊃ s_D (finest), e.g. v11's levels "no context" ->
attended context -> exact context -> exact context + open windows. Each node s
is an expert with its own sequential predictor p_e(s) (here: Beta(1/2, 1/2),
i.e. Krichevsky–Trofimov, counts of the data that reached s), and the model
class is every pruning of the context tree: below a node either the node is a
leaf (its own estimator predicts) or it splits into its children. With prior
1/2 for "leaf" at every non-finest node, the exact Bayesian mixture over all
prunings (Willems, Shtarkov & Tjalkens 1995) is computed along the current path
alone, from one number per node:

    eta(s) = log [ P_e(s) / prod_children P_w(c) ]   (data that reached s)
    beta(s) = sigmoid(eta(s) + log prior_leaf/(1 - prior_leaf))
    p_w(s) = beta(s) p_e(s) + (1 - beta(s)) p_w(child on the path),  p_w(s_D) = p_e(s_D)
    after y:  eta(s) += log p_e(s)(y) - log p_w(child)(y)

Every data point reaches every node of its path exactly once, so no data are
counted twice (unlike a chain of shrinkage, where a child's data also sit in
its parent's prior). The predictive is a proper posterior predictive and its
cumulative log loss is the log marginal likelihood of the mixture: usable for
comparing models. The marginal posterior P(s is a leaf | data) answers the
question "does refining the context by this split help?" — the question v11
answers with separate split Bayes factors (attention, controllability, timing
relevance).
"""
from __future__ import annotations

from math import exp, log


class Node:
    __slots__ = ("k", "n", "eta")

    def __init__(self):
        self.k = self.n = 0.
        self.eta = 0.

    def pe(self, a=.5):
        return (self.k + a) / (self.n + 2. * a)


def path_predict(path, prior_leaf=.5, a=.5):
    """p_w(root) for a path of Nodes (root first); also the per-level weights
    w_l (posterior that level l is the leaf on this path; they sum to 1)."""
    lo = log(prior_leaf / (1. - prior_leaf))
    D = len(path) - 1
    pw = path[D].pe(a)
    pws = [0.] * (D + 1)
    pws[D] = pw
    betas = [1.] * (D + 1)
    for d in range(D - 1, -1, -1):
        s = path[d]
        z = s.eta + lo
        b = 1. / (1. + exp(-z)) if z > -50 else 0.
        betas[d] = b
        pw = b * s.pe(a) + (1. - b) * pw
        pws[d] = pw
    w, rest = [], 1.
    for d in range(D + 1):
        w.append(rest * betas[d])
        rest *= 1. - betas[d]
    return pws[0], pws, w


def path_update(path, y, prior_leaf=.5, a=.5):
    _, pws, _ = path_predict(path, prior_leaf, a)
    D = len(path) - 1
    for d in range(D):
        s = path[d]
        pe = s.pe(a)
        pc = pws[d + 1]
        s.eta += log(pe if y else 1. - pe) - log(pc if y else 1. - pc)
    for s in path:
        s.k += float(y)
        s.n += 1.
