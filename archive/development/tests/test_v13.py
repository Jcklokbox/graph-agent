import unittest

from agents.v13 import PosteriorGraph13

N, A, V = 4, 2, 3


def run(agent, xs):
    for x, y in zip(xs, xs[1:]):
        agent.step(x)
        agent.learn(y)


class ContextInference(unittest.TestCase):
    def test_no_birth_on_stationary_data(self):
        b = PosteriorGraph13(N, A, V, seed=0)
        cyc = [(1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.)] * 200
        run(b, cyc)
        self.assertEqual(b.births, 0)

    def test_birth_after_a_lasting_change_and_retrieval(self):
        b = PosteriorGraph13(N, A, V, seed=0)
        cue_us = [(1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 0., 0.)] * 80
        cue_none = [(1., 0., 0., 0.), (0., 0., 0., 0.), (0., 0., 0., 0.)] * 80
        run(b, cue_us + cue_none)
        self.assertGreaterEqual(b.births, 1)
        self.assertEqual(len(b._ctx_models), len(b._belief))
        self.assertAlmostEqual(sum(b._belief), 1., places=9)
        run(b, cue_us[-3:] + cue_us)
        self.assertGreater(b.predict_next((1., 0., 0., 0.), 0)[1], .5)


if __name__ == "__main__":
    unittest.main()
