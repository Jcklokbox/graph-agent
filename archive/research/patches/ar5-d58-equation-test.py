"""Equation checks against independent quadrature of the masked probit."""
import unittest
from math import erfc, sqrt
import numpy as np
from models.kernels import masked_probit_moments


class TestMaskedProbit(unittest.TestCase):
    def test_moments_against_numerical_likelihood(self):
        grid = np.linspace(-8., 8., 6001)
        prior = np.exp(-.5 * grid**2)
        for mean, v in ((-1.4, .1), (-.7, 1.3), (.8, .8)):
            x = mean + sqrt(v) * grid
            cue = np.array([.5*erfc(-u/sqrt(2.)) for u in x])
            for bg, retention in ((0., 1.), (.4, .6), (.9, 1.)):
                p = bg + (1.-bg)*retention*cue
                for y in (True, False):
                    mass = prior * (p if y else 1.-p); mass /= mass.sum()
                    m, c = masked_probit_moments(mean, v, y, bg, retention)
                    self.assertAlmostEqual(mean+v*m, mass @ x, places=9)
                    self.assertAlmostEqual(v-v*v*c, mass @ (x-mass @ x)**2, places=9)
