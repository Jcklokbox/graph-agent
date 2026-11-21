import unittest

import numpy as np

from eval.scripts.distal_contact_probe import ActionOverride


class DistalContactProbeTests(unittest.TestCase):
    def test_forced_action_still_draws_the_nominal_choice(self):
        control = np.random.default_rng(7)
        override = ActionOverride(np.random.default_rng(7))
        np.testing.assert_array_equal(override.random(5), control.random(5))
        override.force = 2
        self.assertEqual(override.choice([0, 1]), 2)
        self.assertEqual(override.nominal, control.choice([0, 1]))
        override.force = None
        self.assertEqual(override.choice([0, 1, 2]), control.choice([0, 1, 2]))
        np.testing.assert_array_equal(override.random(5), control.random(5))
