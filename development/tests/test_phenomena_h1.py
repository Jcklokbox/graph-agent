"""Held-out T2 sets h1 (D-23) and h2 (D-30): translations checked with the references only."""
import unittest

from envs.phenomena import SETS
from eval.phenomena import run_group


def effect(agent, phen, seed=0, wset="h1"):
    table = SETS[wset][0]
    P = table[phen]
    r = {g: run_group(agent, phen, g, seed, wset)[4] for g in P.groups}
    return P.effect(r)


class TestH1(unittest.TestCase):
    def test_rescorla_wagner_pattern(self):
        self.assertGreater(effect("rw", "overshadowing"), .3)      # credit is shared
        self.assertGreater(effect("rw", "overexpectation"), .3)    # additive error
        self.assertAlmostEqual(effect("rw", "latent-inhibition"), 0., places=6)   # RW cannot
        self.assertLess(effect("rw", "negative-patterning"), -.1)  # elemental: AB > A, B

    def test_memoryless_table_cannot_show_aab(self):
        """Acquisition and extinction in one configuration pool to the prior of a novel one."""
        self.assertAlmostEqual(effect("cfg", "renewal-AAB"), 0., places=6)


class TestH2(unittest.TestCase):
    """Held-out set h2 (D-30): the controls carry no confound a memoryless table can exploit."""

    def test_references(self):
        self.assertGreater(effect("rw", "superconditioning", wset="h2"), .3)
        self.assertGreater(effect("rw", "reinstatement-context", wset="h2"), .1)
        self.assertGreater(effect("cfg", "LI-context", wset="h2"), .3)
        for row in ("sensory-precond", "US-preexposure"):
            self.assertAlmostEqual(effect("cfg", row, wset="h2"), 0., places=6)


if __name__ == "__main__":
    unittest.main()
