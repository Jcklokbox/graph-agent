"""Independently calculate the selected real atom law on its retained support."""
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent, JointReward
from eval.scripts.ar6_parent_evidence import structure_scores
from eval.scripts.ar6_parent_projection import real_atoms


class JointRewardTests(unittest.TestCase):
    def test_controller_attaches_only_the_final_event_reward(self):
        with patch('models.factor_reward.EventReward') as placeholder, \
                patch('agents.v60.JointReward', wraps=JointReward) as factory:
            agent = Agent(5, 2, 4, seed=7)
        factor = agent.factor
        placeholder.assert_not_called()
        factory.assert_called_once_with(factor.J, factor.A, factor.D)
        self.assertIsInstance(factor.reward, JointReward)
        self.assertTrue(factor.reward_event and factor.reward_control)
        self.assertIsNone(factor.action_roles)

    def test_common_support_and_real_amplitudes(self):
        h=JointReward(4,2,1)
        h.slots[0]=0;h.anchors[0]=1;h.context_t=80;h.context_x[2]=11
        h.anchor_n[0]=20;h.pair_on[0,2]=10.;h.assoc[0,2]=15.
        x=np.zeros((3,h.F-1));x[:2,1]=1.;x[1,2]=1.
        ph=x.astype(bool);by=np.array([1.,0.,0.,0.]);c=int(h.code(by))
        seq=((.99,.97,.95,.96),(.02,.03,.04,.05),(.45,.46)*12)
        for i,values in enumerate(seq):
            for value in values:h.replay.append((by,0,value,by,x[i],ph[i]))
        ref=structure_scores(h,c,True);best=ref['best_compound'][0]
        h.refresh_joint()
        j,m,l,d=h.joint_key[c];self.assertEqual((j,m,l),best[1:4]);self.assertEqual(int(d),0)
        self.assertAlmostEqual(h.joint_edge[c],best[0]-max(0.,ref['best_unary'][0][0]),places=11)
        n,s=real_atoms(h,c,j,m,l);alpha=n+1./h.B
        orig=pickle.dumps(h);law=h.rule();splits=h.signed_rules(law)
        mu,p,*_=h.likelihood(law,x.astype(bool));mu,p,r,count,g=h.context_moment(splits,x,ph,mu,p,law[3]);self.assertEqual(orig,pickle.dumps(h))
        np.testing.assert_array_equal(count[:,c],alpha.sum(1));np.testing.assert_array_equal(p[:,c],alpha/alpha.sum(1,keepdims=True))
        np.testing.assert_array_equal(r[:,c],(s+h.center/h.B)/alpha)
        self.assertEqual(len(set(g[:,c])),3);self.assertGreater(mu[0,c],mu[2,c]);self.assertGreater(mu[2,c],mu[1,c])
        h.replay.clear();h.refresh_joint();self.assertFalse(h.joint_edge.any())

    def test_onset_offer_retains_first_moment_across_regime_recall(self):
        h = JointReward(4, 2, 1)
        cue = np.array([0., 1., 0., 0.])
        arrival = np.zeros_like(cue)
        for r in (.25, .75):
            h.learn_joint(cue, (), 0, r, arrival)
        offer = h.provisional(h.rule())
        intensity = 2. / 2.5
        expected = .5 + .5 * intensity * (1. - 2.5 / 3.) / np.sqrt(3.)
        self.assertEqual(h.onset[0, 1], 2.)
        self.assertEqual(h.onset_reward[0, 1], 1.)
        self.assertAlmostEqual(offer[0, 1], expected)

        h._switch(recalled=False)
        self.assertEqual(h.onset[0, 1], 0.)
        self.assertEqual(h.onset_reward[0, 1], 0.)
        h._switch(recalled=True)
        self.assertEqual(h.onset[0, 1], 2.)
        self.assertEqual(h.onset_reward[0, 1], 1.)
        np.testing.assert_array_equal(h.provisional(h.rule()), offer)

    def test_matched_action_reward_reuses_the_same_atom_update(self):
        h = JointReward(3, 2, 1)
        b = np.zeros(h.J)
        x, o = h.input(b, ())[1:], h.phase(b, ())
        payoffs = ((.1, .4, .8), (.2, .5, .9))
        # Equal visits, disjoint bins, with real representatives.
        for action, amounts in enumerate(payoffs):
            for amount in amounts:
                h.learn_joint(b, (), action, amount, b)
        actual = h.constant_context_action_values(b, x, o)
        p, u, means = [], [], []
        for amounts in payoffs:
            n, amount = np.full(h.B, 1. / h.B), h.center / h.B
            for reward in amounts:
                k = int(reward * h.B)
                n[k] += 1.
                amount[k] += reward
            p.append(n / n.sum())
            u.append(amount / n)
            means.append(amount.sum() / n.sum())
        expected = []
        for action in range(h.A):
            revised = (4. * means[action] + u[action]) / 5.
            expected.append(means[action] + p[action] @ np.maximum(means[1 - action], revised))
        np.testing.assert_allclose(actual, expected, rtol=0., atol=4e-16)

    def test_controller_learns_which_action_pays_without_a_sensory_change(self):
        agent = Agent(6, 2, 5, seed=1)
        x = np.r_[np.zeros(5), .5]
        actions = []
        for _ in range(48):
            a = agent.step(x)
            actions.append(a)
            y = x.copy()
            y[-1] = .9 if a == 0 else .1
            agent.learn(y)
            x = y
        self.assertGreaterEqual(actions[-24:].count(0), 22)
        self.assertFalse((agent.factor.reward.slots >= 0).any())

    def test_action_split_requires_matched_context_and_competing_action_data(self):
        h = JointReward(3, 2, 1)
        b = np.zeros(h.J)
        x, o = h.input(b, ())[1:], h.phase(b, ())
        for action in range(h.A):
            arrival = np.eye(h.J)[action]
            for _ in range(5):
                h.replay.append((b, action, .1 + .8 * action, arrival, x, o))
        self.assertIsNone(h.constant_context_action_values(b, x, o))
        h.replay.clear()
        for action in range(h.A):
            for reward in (.1, .9) * 3:
                h.replay.append((b, action, reward, b, x, o))
        self.assertIsNone(h.constant_context_action_values(b, x, o))
        h.replay.clear()
        h.replay.extend([(b, 0, .9, b, x, o)] * 8)
        self.assertIsNone(h.constant_context_action_values(b, x, o))

    def test_never_admit_unwitnessed_partition(self):
        h=JointReward(4,2,1);h.slots[0]=0;h.anchors[0]=1
        h.context_t=100;h.context_x[2]=4;h.anchor_n[0]=12;h.pair_on[0,2]=6.;h.assoc[0,2]=7.
        by=np.array([1.,0.,0.,0.]);x=np.zeros(h.F-1,bool);x[1:3]=True
        for y in (.01,.99,.53)*10:h.replay.append((by,0,y,by,x,x))
        h.refresh_joint();self.assertFalse(h.joint_edge.any())
