import unittest
import numpy as np
from models.factor_information import log_marginal
from research.ar6_partition_math import WindowEvidence


class SharedEvidenceTests(unittest.TestCase):
    def test_stream_matches_recomputed_shared_window_and_real_amounts(self):
        rng=np.random.default_rng(681)
        for cap in (1,8,96):
            bank=WindowEvidence(2,cap)
            for i in range(400):
                r=float(rng.uniform()) if i%4 else float(i%8==0)
                classes=np.array([i%2,i%3]); bank.append(r,classes)
                pooled=log_marginal(bank.total,1./16)
                want=np.array([sum(log_marginal(group,1./16) for group in model)-pooled for model in bank.n])
                np.testing.assert_allclose(bank.score,want,rtol=0,atol=3e-12)
                expected=np.zeros_like(bank.amount)
                for v,k,g in bank.history:expected[bank.model,g,k]+=v
                np.testing.assert_allclose(bank.amount,expected,rtol=0,atol=3e-14)
                np.testing.assert_array_equal(bank.n.sum(1),np.broadcast_to(bank.total,(2,16)))
                self.assertEqual(int(bank.total.sum()),min(i+1,cap))

    def test_reject_invalid_configuration_before_eviction(self):
        bank=WindowEvidence(1,1);bank.append(.35,[0])
        for value,groups in ((2.,[0]),(.25,[3]),(.11,[1.3])):
            with self.assertRaises(ValueError):bank.append(value,groups)
            self.assertAlmostEqual(bank.amount.sum(),.35)
