import unittest

import numpy as np

from agents.v12 import PosteriorGraph12
from agents.v15 import Agent as V15
from envs.family import World, WorldEnv, constant, hidden_hmm
from models.decompose import Decomposer


def _world():
    A = 2
    return World("t", 4, A, 3, [hidden_hmm("h", [0, 1], .95, .9, .1, A),
                                hidden_hmm("g", [2], .9, .8, .2, A), constant("v", 3, A)])


class TestV15(unittest.TestCase):
    def test_equals_v12_when_the_witness_has_no_weight(self):
        w = _world()
        env, rng = WorldEnv(w, 1), np.random.default_rng(0)
        a12, a15 = PosteriorGraph12(4, 2, 3, seed=0), V15(4, 2, 3, seed=0)
        a15._weights = lambda: np.vstack([np.ones(4), np.zeros(4)])
        x = env.reset()
        for _ in range(120):
            x = tuple(float(u) for u in x)
            act12, act15 = a12.step(x), a15.step(x)
            self.assertEqual(act12, act15)
            p12 = a12.predict_next(a12._pending[0], act12)
            p15 = a15.predict_next(a15._pending[0], act15)
            self.assertTrue(np.allclose(p12, p15, atol=0., rtol=0.))
            y = env.step(act12)
            a12.learn(y)
            a15.learn(y)
            x = y

    def test_conjunctive_candidates_stay_valid(self):
        w = _world()
        env, rng = WorldEnv(w, 2), np.random.default_rng(1)
        d = Decomposer(4, 2, 3, seed=0, windows=True, conj=True)
        x = env.reset()
        for _ in range(600):
            a = int(rng.integers(2))
            p = d.predict(x, a)
            self.assertTrue(np.all((p >= 0.) & (p <= 1.)))
            y = env.step(a)
            d.learn(x, a, y)
            x = y
        self.assertTrue(any(d.cj.values()))


if __name__ == "__main__":
    unittest.main()
