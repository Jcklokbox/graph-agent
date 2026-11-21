import unittest
import numpy as np
from models.discovery import NeighborBank
from models.kernels import copy_ancestors, systematic_ancestry
from models.factor import FactorModel
from models.decompose import Decomposer


class DiscoveryTests(unittest.TestCase):
    def test_ordered_resample_is_exact_with_repeats_and_unused_tail(self):
        r = np.random.default_rng(202)
        for n in (1, 2, 25, 200):
            for _ in range(25):
                logits = r.normal(size=n) * 5
                w = np.exp(logits - logits.max()); w /= w.sum()
                u = r.random()
                idx = systematic_ancestry(w, u)
                np.testing.assert_array_equal(idx, np.minimum(n - 1, np.searchsorted(np.cumsum(w), (u + np.arange(n)) / n)))
                bank = r.normal(size=(n, 5, 2)); bank[:, 3:] = np.arange(2)[None, :, None]
                expected = bank[idx].copy(); copy_ancestors(bank, idx, 3)
                np.testing.assert_array_equal(bank, expected)

    def test_exploration_pins_slots_and_covers_unseen_sources(self):
        n = 49
        nb = NeighborBank(n, 6, 0, include_self=False)
        pinned = nb.source[:, :3].copy()
        seen = [set(x) for x in nb.source]
        for _ in range(n):
            rr, _ = nb.rotate(pinned)
            self.assertLessEqual(len(rr), n)
            for j, row in enumerate(nb.source):
                self.assertEqual(len(set(row)), 6); self.assertNotIn(j, row)
                self.assertTrue(set(pinned[j]).issubset(row))
                seen[j].update(row)
        self.assertTrue(all(len(s) == n - 1 for s in seen))

    def test_active_sparse_stores_are_linear(self):
        def make(n):
            f = FactorModel(n + 1, 2, n, N=2, cond=True, process=True)
            d = f.proc
            factor = sum(getattr(f, x).size for x in ('ct', 'cx', 'cr', 'cxr', 'cev'))
            return factor, d.pair_s.size + d.pair_y2.size
        self.assertEqual(tuple(2 * x for x in make(24)), make(48))
        d = Decomposer(2, 2, 1)
        d.predict([1., 0.], 0)
        d.learn([1., 0.], 0, [0., 0.])
        self.assertEqual(d._sparse_pairs(set()), {})
