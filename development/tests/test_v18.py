import unittest

import numpy as np

from agents.v18 import Agent as V18
from envs.family import World, WorldEnv, constant, hidden_hmm


def _world():
    A = 2
    return World("t", 4, A, 3, [hidden_hmm("h", [0, 1], .95, .9, .1, A),
                                hidden_hmm("g", [2], .9, .8, .2, A), constant("v", 3, A)])


class TestV18(unittest.TestCase):
    def test_slots_are_other_channels_and_predictions_probabilities(self):
        env = WorldEnv(_world(), 3)
        a = V18(4, 2, 3, seed=0)
        x = env.reset()
        for _ in range(80):
            act = a.step(tuple(float(u) for u in x))
            p = a._lc_pred(a._pending[0], act)
            self.assertTrue(np.all((p >= 0.) & (p <= 1.)))
            x = env.step(act)
            a.learn(x)
        for jj, j in enumerate(a.chans):
            sl = [s for s in a.slots[j] if s >= 0]
            self.assertNotIn(jj, sl)
            self.assertEqual(len(sl), len(set(sl)))
        # channel 0 shares a hidden process with channel 1, not with channel 2: the screening
        # evidence ranks 1 above 2 as a parent of 0
        self.assertGreater(a.ev[1, 0], a.ev[2, 0])


if __name__ == "__main__":
    unittest.main()
