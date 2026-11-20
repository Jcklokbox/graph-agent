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

    def test_one_lag_per_source(self):
        m = _run(steps=200, lags=range(5), own_lags=(0, 1, 2), k=5, one_lag=True)
        for sl in m.slots:
            src = [s[0] for s in sl if s is not None]
            self.assertEqual(len(src), len(set(src)))
        self.assertIn((0, 2), m.slots[1])            # x_1(t+1) depends on x_0(t-2)

    def test_conditional_screening_finds_delay(self):
        m = _run(steps=200, lags=range(5), own_lags=(0, 1, 2), k=5, cond=True)
        self.assertIn((0, 2), m.slots[1])

    def test_view_mode_runs(self):
        _run(place="view")


if __name__ == "__main__":
    unittest.main()
