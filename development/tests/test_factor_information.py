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
