"""Shared forecast work must not merge action-specific laws or learned regimes."""
import unittest
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent
from models.factor_rollout import provisional_values
from models.kernels import factor_read


class SharedForecastTests(unittest.TestCase):
    def test_shared_birth_matches_individually_read_rows(self):
        rng = np.random.default_rng(27)
        used = np.array([3, 1, 2, 1, 4])  # The birth row varies by particle.
        N, K, J, F, M, D = 5, 4, 3, 8, 5, 4
        live = np.arange(K)[None, :] < used[:, None]
        q1 = rng.random((N, K, J, F), dtype=np.float32)
        q0 = rng.random(q1.shape, dtype=np.float32)
        weight = rng.random((N, K, J, M), dtype=np.float32)
        mu = rng.normal(size=(N, K, J, D))
        q1[~live], q0[~live], weight[~live], mu[~live] = 0., 0., 1., 0.
        v, center = rng.normal(size=(J, D)), rng.normal(size=J)
        cells = rng.integers(F, size=(J, M - 1))
        pg = rng.random(cells.shape, dtype=np.float32)
        fast, deficit = np.array([True, True, False]), np.array([.2, 0., .3])
        for kk in (2, K):
            args = (q1, q0, weight, mu, v, center, cells, pg, kk, fast)
            reference = factor_read(*args, np.full(N, kk), deficit)
            reference[np.arange(kk)[None, :] > used[:, None]] = .5
            actual = factor_read(*args, used, deficit)
            self.assertEqual(actual.tobytes(), reference.tobytes())

    def test_shares_exact_patterns_but_not_sampled_branches(self):
        f = Agent(7, 3, 6, seed=49).factor
        h = f.reward
        h.P = 4  # Directly expose the sampled fallback without constructing signed rules.
        h.slots[0] = 0
        h.control_mass[0] = 80.
        h.type_count[0, [2, 14]] = 40.
        h.type_sum[0, [2, 14]] = [6., 34.]
        h.on[0, 14, 1:5] = 40.
        x = np.r_[np.zeros(f.J), .5]
        u = np.linspace(.1, .9, 5 * (f.J + 4)).reshape(5, f.J + 4)
        projections = []
        for cue_prob in (.1, .5, .9):
            p = np.full((1, 1, f.J), cue_prob)
            projections.append((np.ones((1, 1)), p, np.empty(0), p))
        for relevant, expected_calls in ((0, 2), (1, 2), (4, 4)):
            h.evidence_type.fill(0.)
            h.evidence_type[0, 1:1 + relevant] = 100.
            with (patch('models.factor_rollout.factor_current', return_value=projections) as project,
                  patch.object(h, 'likelihood', wraps=h.likelihood) as likelihood):
                result = provisional_values(f, x, u, optimistic=False)
            self.assertEqual(likelihood.call_count, expected_calls)
            self.assertEqual(project.call_count, int(relevant != 0))
            if relevant:
                self.assertLess(result[0], result[-1])  # Root sensory weights remain action-specific.
