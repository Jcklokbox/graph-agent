import unittest

import numpy as np
from models.factor_reward import BoundedReward, conditional_draws


class FactorRewardTests(unittest.TestCase):
    def test_fractional_wellbeing_stays_a_scalar(self):
        b=BoundedReward(3,2,2)
        x=np.zeros(3)
        for _ in range(80): b.learn(x,[],0,.45)
        f,raw,_,rewards,mass=b.root(x,[],0)
        self.assertAlmostEqual(raw,.45,places=3)
        self.assertAlmostEqual(mass@rewards,raw)
        self.assertEqual(b.n[1],0.)
        with self.assertRaises(ValueError): b.learn(x,[],1,1.1)

    def test_exact_quiet_clock_and_particle_mixture(self):
        q=np.array([[[.08,.12],[.80,.70]]]); w=np.array([[.55,.45]])
        u=np.random.default_rng(99).random((30000,5))
        y=conditional_draws(w,q,np.zeros(2,bool),np.empty(0),True,u)
        no=np.prod(1-q,axis=-1)
        idle=float(np.sum(w*no))
        mean=(1-idle)*np.sum(w[...,None]*q/(1-no[...,None]),axis=(0,1))
        np.testing.assert_allclose(y.mean(0),mean,atol=.012,rtol=0)
        self.assertAlmostEqual((~y.any(1)).mean(),idle,delta=.012)
        nonquiet=conditional_draws(w,q,np.zeros(2,bool),np.empty(0),False,u)
        self.assertGreater(np.cov(nonquiet.T)[0,1],.06)

    def test_rank_one_branch_matches_explicit_update(self):
        b=BoundedReward(2,2,1); b.slots[:]=np.arange(b.k)
        x=np.array([1.,0.]); ys=np.array([[0.,1.],[1.,1.]])
        for _ in range(5): b.learn(x,[],0,.35)
        root=b.root(x,[],0); f,raw,gain,rs,ws=root
        prior=b.mean.copy(); v=b.cov.copy()
        value=b.utility(root,0,ys,x,[])
        terms=[]
        for r in rs:
            new=prior.copy(); new[0]+=gain*(r-raw)
            terms.append(r+np.clip(b.features(ys,(x,))@new.T,0,1).max(1).mean())
        self.assertAlmostEqual(value,np.dot(terms,ws))
        np.testing.assert_array_equal(prior,b.mean);np.testing.assert_array_equal(v,b.cov)

    def test_action_history_keeps_controllable_dwell_out_of_exogenous_scope(self):
        from models.factor import FactorModel
        m=FactorModel(4,2,3,0,N=3,lags=(0,),own_lags=(0,))
        m.attach_reward(action_roles=True)
        # Draw long dwell with two possible continuation actions. A truly tonic
        # mark in the same view should still be selected.
        x=np.ones(4); y=x.copy()
        for i in range(200):
            a=int(i%25==0)
            y=x.copy(); y[1]=1-a
            m.learn(x,a,y);x=y
        self.assertTrue(m.slow[0]);self.assertFalse(m.slow[1])
        self.assertGreater(m._action_role_score[1],m._phasic_log[1])

    def test_process_continuation_uses_previous_not_next_action(self):
        from types import SimpleNamespace
        from models.factor_rollout import process_after

        class HMM:
            b=np.array([.8,.2])
            def T(self,a): return np.eye(2) if a==0 else np.array([[0.,1.],[1.,0.]])
            def E(self): return np.array([[.95],[.15]])
            def _lik(self,x,E): return E[:,0] if x[0] else 1.-E[:,0]

        h=HMM(); g=SimpleNamespace(started=True,hs=[h],chans=[0],logw=np.zeros(1))
        d=SimpleNamespace(A=2,n=1,groups=[g]); old=h.b.copy()
        for ap in range(2):
            p = (h.b@h.T(ap))*h._lik([1.],h.E()); p/=p.sum()
            want=np.array([float((p@h.T(a)@h.E())[0]) for a in range(2)])
            np.testing.assert_allclose(process_after(d,[1.],ap)[:,0],want)
        np.testing.assert_array_equal(h.b,old)

    def test_nonempty_condition_with_tiny_probabilities(self):
        from models.factor_reward import first_success
        self.assertEqual(first_success(np.array([1e-30,2e-30]),.1),0)
        self.assertEqual(first_success(np.array([1e-30,2e-30]),.5),1)
