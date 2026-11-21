"""Compare hyper-role odds to explicit enumeration on a tiny fixture."""
from itertools import product
import unittest

import numpy as np
from models.factor import tonic_log_odds


class TestRoleMarginal(unittest.TestCase):
    def test_shared_hazard_role_odds_match_enumeration(self):
        logt = np.log([[.2,.5,.7],[.03,.8,.6],[.4,.1,.5]])
        logf = np.log([.3,.4,.2]); n = len(logf)
        m = np.zeros((n,2))
        for h in range(logt.shape[1]):
            for roles in product((0,1), repeat=n):
                p = np.prod([np.exp(logt[j,h])/n if r else np.exp(logf[j])
                             for j,r in enumerate(roles)])
                for j,r in enumerate(roles): m[j,r] += p
        np.testing.assert_allclose(tonic_log_odds(logt, logf), np.log(m[:,1]/m[:,0]))

    def test_unseen_channel_cannot_gain_role_from_other_channels(self):
        x = tonic_log_odds(np.log([[.08,.9],[1.,1.],[.9,.2]]),np.log([.02,1.,.04]))
        self.assertAlmostEqual(x[1], -np.log(3.))
