"""Independently calculate the selected real atom law on its retained support."""
import pickle
import unittest

import numpy as np

from agents.v60 import JointReward
from eval.scripts.ar6_parent_evidence import structure_scores
from eval.scripts.ar6_parent_projection import real_atoms


class JointRewardTests(unittest.TestCase):
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

    def test_never_admit_unwitnessed_partition(self):
        h=JointReward(4,2,1);h.slots[0]=0;h.anchors[0]=1
        h.context_t=100;h.context_x[2]=4;h.anchor_n[0]=12;h.pair_on[0,2]=6.;h.assoc[0,2]=7.
        by=np.array([1.,0.,0.,0.]);x=np.zeros(h.F-1,bool);x[1:3]=True
        for y in (.01,.99,.53)*10:h.replay.append((by,0,y,by,x,x))
        h.refresh_joint();self.assertFalse(h.joint_edge.any())
