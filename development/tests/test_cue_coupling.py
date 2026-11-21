"""Check that the cue ablation changes dependence, not the two marginals."""
import unittest
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent
from research.cue_coupling import compare_cues, coupling_bounds, matched_one_mass, pattern_bounds


class CueCouplingTests(unittest.TestCase):
    def test_bounds_against_two_arrival_frechet_limits(self):
        reach = np.array([.3, .7])
        value = np.eye(2)
        lo, hi = coupling_bounds(reach, .4, value)
        self.assertAlmostEqual(lo, .1)
        self.assertAlmostEqual(hi, .7)
        p_lo, p_hi = pattern_bounds(reach, np.array([.6, .4]), value)
        self.assertLessEqual(p_lo, lo + 1e-15)
        self.assertGreaterEqual(p_hi, hi - 1e-15)
        # An arrival-only payoff is invariant to any coupling.
        np.testing.assert_allclose(pattern_bounds(reach, np.array([.6, .4]), np.tile([1., 0.], (2, 1))), [.3, .3])
        one, supported = matched_one_mass(reach, .4, np.array([[8, 2], [2, 8]]))
        self.assertTrue(supported)
        self.assertAlmostEqual(one.sum(), .4)
        self.assertTrue(np.all((0 <= one) & (one <= reach)))
        self.assertGreater(one[1] / reach[1], one[0] / reach[0])
        unsupported, ok = matched_one_mass(reach, .4, np.array([[8, 2], [0, 0]]))
        self.assertFalse(ok)
        np.testing.assert_array_equal(unsupported, reach * .4)

    def test_actual_fixed_v60_tree_loses_action_ranking_at_fixed_marginals(self):
        f = Agent(4, 2, 3).factor
        h = f.reward
        h.slots[:2] = [0, 2]  # cue 1 is not controlled
        trans = np.zeros_like(h.trans)
        trans[:, 0, 1:3] = .5
        trans[:, 1, 1] = trans[:, 2, 2] = 1.
        r = np.array([.1, .9])
        bins = (r * h.B).astype(int)
        for c, preferred_cue in ((1, 0), (2, 1)):
            h.type_count[c, bins] = 40.
            h.type_sum[c, bins] = 40. * r
            h.on[c, bins[preferred_cue], 1] = 40.
            h.control_mass[c] = 80.
            h.evidence_type[c, 1] = 100.

        b = np.zeros(h.J)
        features, phase = h.input(b, ())[1:], h.phase(b, ())
        for a in range(2):
            for c in (1, 2):
                arrival = np.array([c == 1, (c == 2) != bool(a), c == 2], dtype=float)
                h.replay.extend([(b, a, .5, arrival, features, phase)] * 8)
        p = np.array([[[0., .5, 0.]]])
        proj = [(np.ones((1, 1)), p, np.empty(0), p)] * 2
        with (patch.object(h, 'all_transitions', return_value=trans),
              patch('models.factor_rollout.factor_current', return_value=proj),
              patch('research.cue_coupling.factor_current', return_value=proj)):
            report = compare_cues(f, np.r_[b, .5], np.full((5, h.J + 4), .5))
        np.testing.assert_array_equal(report['fixed'], [report['fixed'][0]] * 2)
        cue = report['cues'][1]
        self.assertEqual(cue['supported'], [True, True])
        np.testing.assert_array_equal(cue['bounds'][0], cue['bounds'][1])
        self.assertGreater(cue['adjusted'][0] - cue['adjusted'][1], .5)
