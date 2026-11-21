import unittest
import numpy as np


class V50RewardTests(unittest.TestCase):
    def test_selective_anchor_swap_preserves_unchanged_anchor_counts(self):
        from agents.v50 import GroundedSignedReward

        h = GroundedSignedReward(6, 2, 1)
        h.slots[0] = 0
        h.total = 200.
        h.onset_total[1:5] = [50., 5., 40., 30.]
        h.past_on[1:5] = [50., 5., 40., 30.]
        h.anchors[:] = [1, 2, 3, 4]
        h.anchor_n[0] = 100.
        h.anchor_n[1] = 10.
        # Record a rare shock on anchor 0 (j=1, cfg=1) well before the 96-step replay window
        h.ctx_n[1, 0, 0, 4] = [40., 1.]
        h.ctx_s[1, 0, 0, 4] = [40., 0.]
        h.ctx_sq[1, 0, 0, 4] = [40., 0.]
        # Populate replay with unrelated observations while keeping j=1,2,3,4 top-4
        b = np.zeros(6)
        b[0] = 1.
        bn = b.copy()
        hist = [b.copy()]
        for _ in range(h.replay.maxlen or 96):
            h.learn_joint(b, hist, 0, 0.5, bn)
        # Swap anchor 1 only: j=5 surpasses j=2 while j=1,3,4 stay pinned
        h.onset_total[5] = 20.
        h.past_on[5] = 20.
        before_n = h.ctx_n[1, 0, 0, 4].copy()
        h.learn_joint(b, hist, 0, 0.5, bn)
        self.assertEqual(int(h.anchors[0]), 1)
        self.assertEqual(int(h.anchors[1]), 5)
        np.testing.assert_allclose(h.ctx_n[1, 0, 0, 4], before_n)
        self.assertEqual(float(h.ctx_n[1, 1, 0, 4].sum()), 0.)

    def test_empirical_variance_signed_rules_avoid_bic_cliff_and_unary_blindness(self):
        from agents.v50 import GroundedSignedReward

        h = GroundedSignedReward(5, 2, 1)
        h.slots[0] = 0
        h.anchors[0] = 1
        h.context_t = 400
        h.context_x[1] = 200
        h.context_x[2] = 20
        h.anchor_n[0] = 200.
        h.pair_on[0, 2] = 20.
        h.assoc[0, 2] = 25.
        # Marginal unary evidence for j=1 is weak/diluted (below log1p(F-1)),
        # while configural split (j=1, m=2) has deterministic 0 vs 1 payoff even at N=201.
        h.evidence_type[1, 1] = 1.0
        h.ctx_n[1, 0, 0, 2] = [200., 1.]
        h.ctx_s[1, 0, 0, 2] = [200., 0.]
        h.ctx_sq[1, 0, 0, 2] = [200., 0.]
        rule = h.rule()
        splits = h.signed_rules(rule)
        self.assertEqual(len(splits), 1)
        j, m, l, support, _, d = splits[0]
        self.assertEqual((j, m, l, d), (1, 2, 0, 0))
        self.assertTrue(bool(support[1]))

    def test_grounded_pair_context_anchors_to_base_for_untested_low_base_cue(self):
        from agents.v47 import CredibleEventReward
        from agents.v50 import GroundedSignedReward

        outs = []
        for cls in (CredibleEventReward, GroundedSignedReward):
            h = cls(5, 2, 1)
            h.slots[:2] = [0, 1]
            h.anchors[0] = 2
            h.context_t = 100
            h.context_x[2] = 40
            h.context_x[3] = 20
            h.anchor_n[0] = 20.
            h.pair_on[0, 3] = 8.
            h.assoc[0, 3] = 15.
            h.onset_total[2] = 10.
            h.past_on[2] = 40.
            # Low-base state c=1 has rule[5][1] ~ 0.02 and zero observations on (j=2, m=3)
            h.type_count[1, 1] = 48.
            h.type_sum[1, 1] = 0.
            rule = h.rule()
            x = np.zeros((h.L, h.J), int)
            x[0, 1] = 1
            x[0, 2] = 1
            phase = x.copy()
            outs.append(float(h.pair_context(rule, x, phase)[1]))
        v47_out, v50_out = outs
        self.assertLess(v50_out, 0.15)
        self.assertGreater(v47_out - v50_out, 0.35)

    def test_v50_agent_steps_and_preserves_prior_reward_contracts(self):
        from agents import agent_class, load
        from agents.v44 import SparseSignedReward
        from agents.v47 import CredibleEventReward
        from agents.v48 import ScopedSignedReward

        self.assertFalse(getattr(SparseSignedReward, 'context_requires_parent', False))
        self.assertTrue(getattr(CredibleEventReward, 'one_window', False))
        self.assertTrue(getattr(ScopedSignedReward, 'context_requires_parent', False))

        cls = agent_class(load('v50'))
        ag = cls(6, 3, 5, seed=0)
        rng = np.random.default_rng(0)
        x = np.zeros(6)
        x[0] = 1.
        for _ in range(12):
            a = ag.step(x)
            y = np.zeros(6)
            y[int(rng.integers(3))] = 1.
            y[5] = 0.5
            ag.learn(y)
            x = y
        self.assertEqual(len(ag.factor.reward.ctx_sq.shape), 5)


if __name__ == '__main__':
    unittest.main()
