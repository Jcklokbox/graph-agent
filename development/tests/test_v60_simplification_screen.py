"""Keep the component cuts local to the experimental factory."""
import unittest

import numpy as np

from envs.tasks import VARIANTS
from eval.scripts.v60_simplification_screen import BANKS, build


class SimplificationScreenTests(unittest.TestCase):
    def test_ablation_does_not_leak_to_reference(self):
        variant = VARIANTS['aut']
        seed = 113
        stripped = build(variant, seed, 'no-process')
        small = build(variant, seed, '32-particles')
        reference = build(variant, seed, 'reference')
        self.assertIsNone(stripped.factor.proc)
        self.assertIsNotNone(small.factor.proc)
        self.assertIsNotNone(reference.factor.proc)
        self.assertEqual(stripped.factor.N, reference.factor.N)
        self.assertEqual(small.factor.N, 32)
        np.testing.assert_array_equal(stripped.factor.hz, reference.factor.hz)
        for attr in BANKS:
            full = getattr(reference.factor, attr)
            compact = getattr(small.factor, attr)
            self.assertEqual(len(full) * compact.nbytes, len(compact) * full.nbytes)
        # Cutting the expert does not alter the controller's random stream.
        self.assertEqual(stripped.rng.bit_generator.state, reference.rng.bit_generator.state)
