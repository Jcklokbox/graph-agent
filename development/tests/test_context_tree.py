"""T0: the path recursion of models/context_tree.py equals the brute-force
Bayesian mixture over every pruning of a small context tree."""
import itertools
import unittest
from math import log

import numpy as np

from models.context_tree import Node, path_predict, path_update


def prunings(depth):
    """Prunings of a complete binary tree of the given depth, as sets of leaves
    (tuples of context bits), with CTW prior weights (1/2 leaf at inner nodes)."""
    def rec(prefix):
        if len(prefix) == depth:
            return [([prefix], 1.)]
        out = [([prefix], .5)]
        for l0, w0 in rec(prefix + (0,)):
            for l1, w1 in rec(prefix + (1,)):
                out.append((l0 + l1, .5 * w0 * w1))
        return out
    return rec(())


def kt_seq_prob(ys):
    k = n = 0
    p = 1.
    for y in ys:
        pe = (k + .5) / (n + 1.)
        p *= pe if y else 1. - pe
        k += y
        n += 1
    return p


class ContextTreeTests(unittest.TestCase):
    def test_path_recursion_equals_brute_force_mixture(self):
        rng = np.random.default_rng(0)
        depth, T = 2, 40
        ctx = [tuple(int(b) for b in rng.integers(0, 2, depth)) for _ in range(T)]
        ys = [int(rng.random() < (.9 if c[0] == 1 and c[1] == 0 else .3)) for c in ctx]
        nodes = {}
        models = prunings(depth)
        for t in range(T):
            path = [nodes.setdefault(ctx[t][:d], Node()) for d in range(depth + 1)]
            p_ctw, _, w = path_predict(path)
            self.assertAlmostEqual(sum(w), 1., places=12)
            # brute force: posterior over prunings, then the leaf's KT predictive
            num = den = 0.
            for leaves, prior in models:
                like = 1.
                for leaf in leaves:
                    like *= kt_seq_prob([ys[u] for u in range(t) if ctx[u][:len(leaf)] == leaf])
                leaf = next(l for l in leaves if ctx[t][:len(l)] == l)
                past = [ys[u] for u in range(t) if ctx[u][:len(leaf)] == leaf]
                pe = (sum(past) + .5) / (len(past) + 1.)
                num += prior * like * pe
                den += prior * like
            self.assertAlmostEqual(p_ctw, num / den, places=10)
            path_update(path, ys[t])


if __name__ == "__main__":
    unittest.main()
