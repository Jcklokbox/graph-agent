"""Short fixtures for common-support structure selection and its limits."""
import unittest
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent
from eval.scripts.reward_change_probe import episode
from research.reward_partitions import POOLED, Partition, fit, fixed_values, refit_arrivals


class RewardPartitionTests(unittest.TestCase):
    def test_selects_unary_without_any_pair(self):
        bits = np.repeat([[0], [1]], 8, axis=0).astype(bool)
        law = fit(bits, bits, np.repeat([.9, .1], 8), [0], [])
        self.assertEqual(law.partition, Partition(2, 0))
        p, atoms, mass = law.atoms(np.array([0, 1]))
        np.testing.assert_allclose((p * atoms).sum(1), (np.array([.9, .1]) * 8 + .5) / 9)
        np.testing.assert_array_equal(mass, [9., 9.])

    def test_pooled_when_no_split_has_support_or_evidence(self):
        bits = np.repeat([[0, 0], [1, 0]], 8, axis=0).astype(bool)
        law = fit(bits, bits, np.full(len(bits), .5), [0, 1], [(0, 1)])
        self.assertEqual(law.partition, POOLED)
        self.assertEqual(law.evidence, 0.)

    def test_rare_joint_depends_on_support_not_just_readout(self):
        # The most recent window misses both branches of the rare onset.
        bits = np.repeat([[1, 0], [1, 1], [0, 0]], [4, 4, 64], axis=0).astype(bool)
        rewards = np.repeat([1., 0., .5], [4, 4, 64])
        short = fit(bits[-64:], bits[-64:], rewards[-64:], [0, 1], [(0, 1)])
        long = fit(bits, bits, rewards, [0, 1], [(0, 1)])
        self.assertEqual(short.partition, POOLED)
        self.assertEqual(long.partition, Partition(3, 0, 1))
        p, atoms, _ = long.atoms(np.array([0, 1, 2]))
        np.testing.assert_allclose((p * atoms).sum(1), [.9, .1, .5])

    def test_time_split_needs_local_evidence_and_preserves_other_groups(self):
        block = [([1, 0], .9), ([0, 0], .4), ([1, 1], .2), ([0, 0], .4)]
        quiet = [([0, 0], .4)] * 100
        history = block * 12 + quiet

        def fit_rows(rows, *, allow_change=True):
            x, r = zip(*rows)
            bits = np.asarray(x, bool)
            return fit(bits, bits, r, [0, 1], [(0, 1)], allow_change=allow_change)

        stationary = fit_rows(history + block * 2)
        self.assertEqual(stationary.partition, Partition(3, 0, 1))
        self.assertEqual(stationary.changes, ())
        # The new law cannot be observed during the quiet interval. Even one
        # contradictory contact does not justify a split on this tape.
        changed = [([1, 0], .1), *block[1:]]
        self.assertEqual(fit_rows(history + changed).changes, ())
        rows = history + changed * 2
        persistent = fit_rows(rows, allow_change=False)
        adaptive = fit_rows(rows)
        self.assertEqual(adaptive.partition, stationary.partition)
        self.assertEqual(adaptive.changes, ((0, len(history)),))
        np.testing.assert_array_equal(adaptive.counts[1:], persistent.counts[1:])
        np.testing.assert_array_equal(adaptive.sums[1:], persistent.sums[1:])
        p0, a0, _ = persistent.atoms(0)
        p1, a1, _ = adaptive.atoms(0)
        self.assertGreater(p0 @ a0, .4)
        self.assertLess(p1 @ a1, .4)
        # Change only the duration of silence, not the changed group's visits.
        no_gap = fit_rows(block * 12 + changed * 2)
        self.assertEqual(no_gap.changes, ((0, len(block) * 12),))
        np.testing.assert_array_equal(no_gap.counts[0], adaptive.counts[0])
        np.testing.assert_array_equal(no_gap.sums[0], adaptive.sums[0])

    def test_time_is_an_alternative_to_context_not_a_required_split(self):
        # No context feature can explain this change. Quantities are compared
        # in the same bin model, so an equally large sample from one bin alone
        # cannot justify a change merely because real within-bin amounts move.
        bits = np.zeros((24, 1), dtype=bool)
        reversal = fit(bits, bits, [.9] * 12 + [.1] * 12, [0], [], allow_change=True)
        self.assertEqual(reversal.partition, POOLED)
        self.assertEqual(reversal.changes, ((0, 12),))
        within_bin = fit(bits, bits, [.07] * 12 + [.11] * 12, [0], [], allow_change=True)
        self.assertEqual(within_bin.changes, ())

    def test_time_cannot_replace_an_observationally_identical_context(self):
        # The collection policy visits one cue only after calibration. Reward
        # depends on that cue, not on time within either cue value.
        bits = np.zeros((20, 24), dtype=bool)
        bits[16:, 0] = True
        rewards = bits[:, 0].astype(float)
        aliased = fit(bits, bits, rewards, range(24), [], allow_change=True)
        self.assertEqual(aliased.partition, Partition(2, 0))
        self.assertEqual(aliased.changes, ())
        # If discovery never offers the relevant cue the alias is invisible.
        omitted = fit(bits, bits, rewards, range(1, 24), [], allow_change=True)
        self.assertEqual(omitted.partition, POOLED)
        self.assertEqual(omitted.changes, ((0, 16),))

    def test_same_pooled_rewards_distinguish_exposure_from_reward_change(self):
        before = [0.] * 15 + [1.]
        after = [0.] + [1.] * 15
        rewards = np.array(before * 2 + after * 2)
        # In one world only the mix of contexts changes. Each context also
        # occurs on the other side of the boundary, but just twice there.
        exposure = rewards[:, None].astype(bool)
        # In the other world the context mix is balanced but both laws change.
        reversal = np.tile(np.repeat([False, True], 16), 2)[:, None]
        stable = fit(exposure, exposure, rewards, [0], [], allow_change=True)
        moved = fit(reversal, reversal, rewards, [0], [], allow_change=True)
        self.assertEqual(stable.partition, Partition(2, 0))
        self.assertEqual(stable.changes, ())
        self.assertEqual(moved.partition, POOLED)
        self.assertEqual(len(moved.changes), 1)

    def test_opposite_changes_cancel_in_pooled_rewards_but_not_conditionally(self):
        rewards = np.tile([0., 1.], 32)
        features = np.r_[rewards[:32], 1. - rewards[32:]][:, None].astype(bool)
        # Same reward sequence and same exposure counts in both periods.
        # Only the pairing of reward with context reverses.
        self.assertEqual(fit(features, features, rewards, [], [], allow_change=True).changes, ())
        law = fit(features, features, rewards, [0], [], allow_change=True)
        self.assertEqual(law.partition, Partition(2, 0))
        self.assertEqual(law.changes, ((0, 33), (1, 32)))
        p, atoms, _ = law.atoms(np.array([0, 1]))
        np.testing.assert_allclose((p * atoms).sum(1), [16.5 / 17., .5 / 17.])

    def test_forecast_memory_does_not_change_partition_or_borrow_other_arrivals(self):
        h = Agent(5, 2, 4).factor.reward
        h.slots[0] = 0
        off, on = np.zeros(h.F - 1, bool), np.zeros(h.F - 1, bool)
        on[1] = True
        features = np.array([off, on] * 8)
        rewards = np.tile([.2, .8], 8)
        unary = fit(features, features, rewards, [1], [], bins=h.B)
        pooled = fit(features, features, rewards, [], [], bins=h.B)
        self.assertEqual(unary.partition, Partition(2, 1))
        origin, other = np.zeros(h.J), np.eye(h.J)[0]
        rows = [(origin, 0, .31, origin, on, on), (origin, 1, .99, other, off, off)]
        refitted = refit_arrivals(h, [unary] + [pooled] * (h.C - 1), rows)
        self.assertEqual(refitted[0].partition, unary.partition)
        self.assertEqual(refitted[0].evidence, unary.evidence)
        np.testing.assert_array_equal(refitted[0].counts.sum(axis=1), [0, 1])
        p, amounts, _ = refitted[0].atoms(np.array([0, 1]))
        np.testing.assert_allclose((p * amounts).sum(1), [.5, (.5 + .31) / 2])
        self.assertAlmostEqual(refitted[1].sums.sum(), .99)

    def test_closed_loop_reversal_is_visible_but_untried_improvement_is_not(self):
        def compare(scenario, probes=False):
            return [episode(scenario, 0, cuts=cuts, probes=probes) for cuts in (False, True)]

        baseline, modified = compare('reversal')
        self.assertEqual(baseline['common'], modified['common'])
        self.assertGreater(modified['rare_post'], baseline['rare_post'] + .5)
        self.assertEqual(*compare('stable', probes=True))
        baseline, modified = compare('unseen_improvement')
        self.assertEqual(baseline, modified)
        self.assertEqual(modified['post_alternative'], 0)
        baseline, modified = compare('unseen_improvement', probes=True)
        self.assertGreater(modified['rare_post'], baseline['rare_post'])

    def test_same_rule_at_new_lag_preserves_a_useful_action_reversal(self):
        f = Agent(5, 2, 4).factor
        h = f.reward
        h.slots[0] = 0
        h.trans[0, :, 1] = 40
        h.trans[1, :, 0] = 40
        h.trans_total[:, :] = 40
        h.control_mass[:2] = [64, 16]

        # Select the joint law at lag 1, with no special case for lag 1.
        j, m = h.J + 1, h.J + 2
        features = np.zeros((32, h.F - 1), dtype=bool)
        features[:16, j] = True
        features[8:16, m] = True
        payoffs = np.repeat([1., 0., .5], [8, 8, 16])
        laws = [fit(features, features, np.full(32, .5), [], [], bins=h.B) for _ in range(h.C)]
        laws[1] = fit(features, features, payoffs, [j, m], [(j, m)])
        self.assertEqual(laws[1].partition, Partition(3, j, m))
        f.hist = [np.zeros(h.J), np.array([0., 1., 0., 0.])]
        x = np.array([0., 1., 0., 0., .5])
        u = np.full((5, h.J + 4), .5)
        with patch('research.reward_partitions.factor_current', side_effect=AssertionError('no lag-zero parent')):
            q_positive = fixed_values(f, x, u, laws)
            f.hist[-1][2] = 1.
            q_adverse = fixed_values(f, x, u, laws)
        self.assertGreater(q_positive[0], q_positive[1])
        self.assertLess(q_adverse[0], q_adverse[1])
