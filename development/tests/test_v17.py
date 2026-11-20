import unittest

import numpy as np

from agents.v12 import PosteriorGraph12
from agents.v17 import Agent as V17
from envs.family import World, WorldEnv, constant, hidden_hmm


def _world():
    A = 2
    return World("t", 4, A, 3, [hidden_hmm("h", [0, 1], .95, .9, .1, A),
                                hidden_hmm("g", [2], .9, .8, .2, A), constant("v", 3, A)])


class TestV17(unittest.TestCase):
    def test_equals_v12_when_the_cause_predictor_has_no_weight(self):
        """The construction adds a member to v12's prediction and nothing else."""
        env = WorldEnv(_world(), 1)
        a12, a17 = PosteriorGraph12(4, 2, 3, seed=0), V17(4, 2, 3, seed=0)
        a17._weights = lambda: np.vstack([np.ones(4), np.zeros(4)])
        x = env.reset()
        for _ in range(60):
            x = tuple(float(u) for u in x)
            act12, act17 = a12.step(x), a17.step(x)
            self.assertEqual(act12, act17)
            p12 = a12.predict_next(a12._pending[0], act12)
            p17 = a17.predict_next(a17._pending[0], act17)
            self.assertTrue(np.allclose(p12, p17, atol=0., rtol=0.))
            y = env.step(act12)
            a12.learn(y)
            a17.learn(y)
            x = y

    def test_cause_predictions_are_probabilities(self):
        env = WorldEnv(_world(), 2)
        a = V17(4, 2, 3, seed=1)
        x = env.reset()
        for _ in range(40):
            act = a.step(tuple(float(u) for u in x))
            p = a._lc_pred(a._pending[0], act)
            self.assertTrue(np.all((p >= 0.) & (p <= 1.)))
            x = env.step(act)
            a.learn(x)
        self.assertEqual(p[3], .5)                   # the value channel is v12's alone


if __name__ == "__main__":
    unittest.main()
