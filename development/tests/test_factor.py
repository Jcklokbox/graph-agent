import unittest

import numpy as np

from models.factor import FactorModel


def _run(steps=120, **kw):
    rng = np.random.default_rng(0)
    n, A, v = 5, 2, 4
    m = FactorModel(n, A, v, seed=0, N=40, K=4, **kw)
    x = np.zeros(n)
    for t in range(steps):
        a = int(rng.integers(A))
        p = m.predict(x, a)
        assert p.shape == (n,) and np.all((p > 0) & (p < 1))
        y = np.zeros(n)
        y[0] = float(t % 5 == 0)                     # cue
        y[1] = float(t % 5 == 3)                     # outcome 3 steps after the cue
        y[2] = float(rng.random() < .3)
        m.learn(x, a, y)
        x = y
    return m


class TestFactor(unittest.TestCase):
    def test_unlagged_runs(self):
        _run()

    def test_lagged_process_runs(self):
        _run(lags=range(5), own_lags=(0, 1, 2), k=5, process=True)

    def test_conditional_screening_finds_delay(self):
        m = _run(steps=200, lags=range(5), own_lags=(0, 1, 2), k=5, cond=True)
        self.assertIn((0, 2), m.slots[1])

    def test_places_only_use_persistent_observations(self):
        m = FactorModel(4, 1, 3, N=10, lags=range(5), cond=True)
        for t in range(35):
            x = [1., float(t % 3 == 0), float(t % 3 == 1), 0.]
            m._observe_place(m._bits(x))
        self.assertEqual(m._signature(np.array([1., 0., 0.])), (0,))
        self.assertEqual(m._signature(np.array([1., 1., 1.])), (0,))
        self.assertEqual(m._signature(np.array([0., 1., 1.])), ())

    def test_gap_relaxes_a_place_without_erasing_regime_parameters(self):
        m = FactorModel(4, 1, 3, N=10, K=3)
        m.started = True
        m.used[:] = 2
        m.t = 3
        post = np.tile([0., 1., 0.], (10, 1))
        counts = np.tile([5., 1., 0.], (10, 1))
        m.places[(0,)] = (0, post.copy(), counts.copy())
        m.mu[:, 0, 1, 0] = -.5
        near = m._regime_prior((0,), 3)
        m.t = 250
        far = m._regime_prior((0,), 3)
        self.assertTrue(np.all(far[:, 0] > near[:, 0]))
        np.testing.assert_allclose(far.sum(1), 1.)
        np.testing.assert_array_equal(m.places[(0,)][1], post)
        np.testing.assert_array_equal(m.mu[:, 0, 1, 0], -.5)

    def test_slots_do_not_admit_multiple_lags_for_a_source(self):
        m = _run(steps=200, lags=range(5), own_lags=(0, 1, 2), k=5, cond=True)
        for row in m.slots:
            ids = [i for slot in row if slot is not None for i, _ in [slot]]
            self.assertEqual(len(set(ids)), len(ids))

    def test_canonical_prediction_is_idempotent_and_uses_dev_n(self):
        from models.factor import CANONICAL, make_factor
        from eval.predictive import make

        m = make(CANONICAL + "-N25", 5, 2, 4, 0)
        self.assertEqual(m.N, 25)
        with self.assertRaises(ValueError):
            make_factor("factor-L4-P-V", 5, 2, 4)
        x = np.array([1., 0., 1., 0., 0.])
        y = np.array([0., 1., 0., 0., 0.])
        m.learn(x, 0, y)
        pred = m.predict(y, 0, hypothetical=True)
        state = m.trace()
        np.testing.assert_array_equal(pred, m.predict(y, 0, hypothetical=True))
        self.assertEqual(state, m.trace())




if __name__ == "__main__":
    unittest.main()
