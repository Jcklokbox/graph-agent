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
            return factor, d.pair_s.size
        self.assertEqual(tuple(2 * x for x in make(24)), make(48))
        d = Decomposer(2, 2, 1)
        d.predict([1., 0.], 0)
        d.learn([1., 0.], 0, [0., 0.])
        self.assertEqual(d._sparse_pairs(set()), {})

    def test_cue_routes_only_the_destination_and_does_not_mutate_counts(self):
        f = FactorModel(4, 1, 3, N=3, K=2, cond=True)
        f.slow[2] = True
        f.cue[:, 0, 0] = 14.
        f.cue[:, 1, 1] = 14.
        prior = np.full((3, 2), .5)
        b0 = np.array([1., 0., 1.]); b1 = np.array([0., 1., 1.])
        before = f.cue.copy()
        q0, q1 = f._cue_prior(prior, b0, 2), f._cue_prior(prior, b1, 2)
        self.assertTrue(np.all(q0[:, 0] > .9)); self.assertTrue(np.all(q1[:, 1] > .9))
        np.testing.assert_allclose(q0.sum(1), 1.)
        np.testing.assert_array_equal(f._cue_prior(prior, np.array([0., 0., 1.]), 2), prior)
        np.testing.assert_array_equal(f.cue, before)

    def test_quiet_log_likelihood_remains_finite_for_many_independent_marks(self):
        f = 1500
        pp = np.full((3, 2, f), .98)
        pp[0, 0, :] = .9
        weights = np.full((3, 2), .5)
        idle, norm = FactorModel._activity(pp, weights, np.zeros(f), np.ones(f, bool))
        self.assertTrue(np.isfinite(idle)); self.assertLess(idle, -1000.)
        np.testing.assert_allclose(norm, 1.)

    def test_live_slot_reset_preserves_unaffected_and_pristine_tail(self):
        rng = np.random.default_rng(919)
        f = FactorModel(4, 2, 3, N=3, K=4)
        f.used[:] = 2
        f.ev[:, :2] = rng.normal(size=f.ev[:, :2].shape)
        f.q1[:, :2] = 4.
        f.q0[:, :2] = 6.
        f.mu[:, :2] = .13
        f.Sg[:, :2] = rng.random(f.Sg[:, :2].shape)
        expected = {name: getattr(f,name).copy() for name in ('ev','q1','q0','_E','mu','Sg')}
        j,p=1,2
        for i, parents in enumerate(f.sets):
            if p in parents:
                sub = f.sets.index(tuple(x for x in parents if x!=p))
                expected['ev'][:,:,j,i]=expected['ev'][:,:,j,sub]
                for bank in ('q1','q0'):
                    expected[bank][:,:,j,f.off[i]:f.off[i+1]]=0
        ev = expected['ev'][:,:,j]; expected['_E'][:,:,j]=np.exp(ev-ev.max(-1,keepdims=True))
        expected['mu'][:,:,j,p]=0
        expected['Sg'][:,:,j,p,:]=0; expected['Sg'][:,:,j,:,p]=0; expected['Sg'][:,:,j,p,p]=f.s0
        f._reset_pos(j,p)
        for k, v in expected.items():
            np.testing.assert_allclose(getattr(f,k),v,rtol=2e-7,atol=1e-9)
