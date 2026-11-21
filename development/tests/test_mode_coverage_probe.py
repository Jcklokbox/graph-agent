import unittest

import numpy as np

from agents.v60 import JointReward
from eval.scripts.mode_coverage_probe import OverrideBudget, RevisionGate, coverage_action


class OverrideBudgetTests(unittest.TestCase):
    def test_only_new_modes_refill_credits(self):
        budget = OverrideBudget(2)
        budget.spend()
        budget.on_switch(recalled=True)
        self.assertEqual(budget.remaining, 1)
        budget.spend()
        budget.on_switch(recalled=True)
        self.assertEqual(budget.remaining, 0)
        with self.assertRaises(AssertionError):
            budget.spend()
        budget.on_switch(recalled=False)
        self.assertEqual(budget.remaining, 2)

    def test_warm_fork_has_no_fictitious_startup_credit(self):
        budget = OverrideBudget(1, exhausted=True)
        self.assertEqual(budget.remaining, 0)
        budget.on_switch(recalled=False)
        self.assertEqual(budget.remaining, 1)


class RevisionGateTests(unittest.TestCase):
    def test_revision_requires_live_departure_per_origin(self):
        gate = RevisionGate()
        self.assertTrue(gate.allows(0))
        gate.observe(0, revised=True, switched=False)
        # The transition triggering the rebuild belongs to the previous code.
        self.assertFalse(gate.allows(0))
        gate.observe(0, revised=False, switched=False)
        self.assertTrue(gate.allows(0))
        self.assertFalse(gate.allows(1))
        gate.observe(1, revised=True, switched=False)
        self.assertFalse(gate.allows(0))
        self.assertFalse(gate.allows(1))

    def test_mode_switch_supersedes_revision_gate(self):
        gate = RevisionGate()
        gate.observe(0, revised=True, switched=False)
        self.assertFalse(gate.allows(1))
        gate.observe(0, revised=False, switched=True)
        self.assertTrue(gate.allows(1))


class ModeCoverageTests(unittest.TestCase):
    def test_bank_reset_is_not_loss_of_witnessed_actions(self):
        h = JointReward(3, 3, 1)
        h.slots[0] = 0
        b = np.zeros(h.J)
        features, phase = h.input(b, ())[1:], h.phase(b, ())
        for action in (0, 1):
            h._record(b, action, .5, b, features, phase)
        h._switch(recalled=False)
        h._record(b, 1, .5, b, features, phase)
        active = h.trans_total[:, 0]
        witnessed = h.recent_trans[:, 0].any(1)
        q = [.8, .9, .2]
        np.testing.assert_array_equal(active, [0., 1., 0.])
        np.testing.assert_array_equal(witnessed, [True, True, False])
        self.assertEqual(coverage_action(active, witnessed, q, policy='active'), 0)
        self.assertEqual(coverage_action(active, witnessed, q, policy='witnessed'), 2)
        h._record(b, 2, .5, b, features, phase)
        self.assertIsNone(coverage_action(active, h.recent_trans[:, 0].any(1), q, policy='witnessed'))

    def test_archive_recall_restores_coverage_without_new_trials(self):
        h = JointReward(3, 2, 1)
        h.slots[0] = 0
        b = np.zeros(h.J)
        features, phase = h.input(b, ())[1:], h.phase(b, ())
        for action in range(h.A):
            h._record(b, action, .5, b, features, phase)
        h._switch(recalled=False)
        h._record(b, 0, .5, b, features, phase)
        h._switch(recalled=True)
        active = h.trans_total[:, 0]
        np.testing.assert_array_equal(active, [1., 1.])
        self.assertIsNone(coverage_action(active, h.recent_trans[:, 0].any(1), [.9, .3], policy='active'))

    def test_neither_policy_probes_an_entirely_empty_active_origin(self):
        for policy in ('off', 'active', 'witnessed'):
            self.assertIsNone(coverage_action(np.zeros(2), np.array([True, False]), [.9, .3], policy=policy))
