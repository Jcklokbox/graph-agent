import unittest

import numpy as np

from envs.family import World, WorldEnv, constant, hidden_hmm, noise
from models.decompose import Decomposer


class TestDecomposer(unittest.TestCase):
    def test_merges_one_process_and_keeps_independent_noise_apart(self):
        A = 2
        w = World("t", 4, A, 3, [hidden_hmm("h", [0, 1], .95, .9, .1, A),
                                 noise("n", 2, p=.3, A=A), constant("v", 3, A)])
        env, rng = WorldEnv(w, 3), np.random.default_rng(0)
        d = Decomposer(4, A, 3, seed=0)
        x = env.reset()
        for _ in range(1500):
            a = int(rng.integers(A))
            d.predict(x, a)
            y = env.step(a)
            d.learn(x, a, y)
            x = y
        self.assertEqual(d.partition(), [(0, 1), (2,)])


if __name__ == "__main__":
    unittest.main()
