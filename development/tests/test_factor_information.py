import unittest
import numpy as np

from models.factor_information import posterior_option


class InformationTests(unittest.TestCase):
    def test_untried_real_atom_has_known_one_step_decision_value(self):
        # Equal prior odds, two real atoms and no observations. When the two
        # actions tie, a high atom moves the left utility by .25 under H1.
        q=np.ones(2); reach=np.array([1.,0.]); n=np.zeros(2); atom=np.array([0.,1.])
        self.assertAlmostEqual(posterior_option(q,reach,n,n,n,atom,0.),1./16.)
        self.assertAlmostEqual(posterior_option(q,reach[::-1],n,n,n,atom,0.),1./16.)

    def test_known_bad_and_unreachable_reward_have_no_test_value(self):
        q=np.array([.1,1.]);reach=np.array([1.,0.]); n=np.array([30.,0.]); atom=np.array([0.,1.])
        self.assertEqual(posterior_option(q,reach,n,n,n*atom,atom,0.),0.)
        self.assertEqual(posterior_option(q,np.ones(2),n*0,n*0,n*0,atom,0.),0.)

    def test_preference_never_flipped_means_zero_information(self):
        q=np.array([.85,.2]);reach=np.array([.8,.01]);n=np.zeros(2); atom=np.array([.2,.9])
        self.assertAlmostEqual(posterior_option(q,reach,n,n,n,atom,0.),0.)

    def test_mixed_law_keeps_actual_within_atom_amounts(self):
        from models.factor_information import mixed_atoms
        pp,aa=mixed_atoms(np.array([0.,1.]),np.array([0.,1.]),np.array([1.,0.]),
                           np.array([.45,0.]),np.array([.2,.8]),0.)
        self.assertAlmostEqual(pp.sum(),1.)
        self.assertAlmostEqual(pp@aa,.5+.5*(.45+.1+.4)/2.)


class SignedSelectionTests(unittest.TestCase):
    def test_sensory_favorite_does_not_override_outcome_support(self):
        from agents.v56 import SignedSelection
        h=SignedSelection(5,2,1); h.slots[0]=0; h.anchors[0]=1
        h.context_t=100; h.context_x[[2,3]]=12; h.anchor_n[0]=40
        h.pair_on[0,[2,3]]=20; h.assoc[0,[2,3]]=[20.,10.]
        # Both moderators are associated with the same cue; only the second
        # explains real non-Bernoulli utility conditional on the destination.
        h.ctx_n[1,0,0,[2,3]]=[[15.,15.],[15.,15.]]
        h.ctx_s[1,0,0,[2,3]]=[[7.5,7.5],[12.,3.]]
        for cfg,y in enumerate((.8,.2)):
            r=int(y*h.B)
            h.ctx_atoms[1,0,0,3,cfg,r]=15.;h.ctx_amount[1,0,0,3,cfg,r]=15*y
        self.assertEqual(h.signed_rules(h.rule()),[])
        h.evidence_type[1,1]=9.
        rule=h.rule(); splits=h.signed_rules(rule)
        self.assertEqual([(j,m,l) for j,m,l,*_ in splits],[(1,3,0)])
        x=np.zeros((3,h.F-1),int);x[:2,1]=1;x[1:,3]=1
        m0,p0,_,_=h.likelihood(rule,x)
        m,p,a,_,g=h.context_moment(splits,x,x.astype(bool),m0,p0,rule[3])
        np.testing.assert_allclose(m[:,1],[12.5/16.,3.5/16.,m0[2,1]])
        np.testing.assert_allclose((p*a).sum(-1),m)
        self.assertEqual(g[2,1],0)
