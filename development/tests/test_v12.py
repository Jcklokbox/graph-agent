import unittest

import agents
from envs.tasks import SPLITS, VARIANTS


class V12Smoke(unittest.TestCase):
    def test_runs_and_predicts(self):
        v = VARIANTS["dist"]
        _, _, eb, bb = SPLITS["dev"]
        env = v.make(eb)
        b = agents.agent_class(agents.load("v12"))(v.n_inputs, v.n_actions, 11, seed=bb)
        o = env.reset()
        for _ in range(60):
            a = b.step(o)
            o, _, _, _ = env.step(a)
            b.learn(o)
        p = b.predict_next(o, 0)
        self.assertEqual(len(p), v.n_inputs)
        self.assertTrue(all(0. <= q <= 1. for q in p))
        self.assertTrue(any(k[0] == "zeta" for k in b.m.eta))

    def test_voi_vanishes_with_a_clear_gap(self):
        from agents.v12 import _voi
        self.assertGreater(_voi(0., .2), .07)
        self.assertLess(_voi(.8, .2), 1e-4)


if __name__ == "__main__":
    unittest.main()
