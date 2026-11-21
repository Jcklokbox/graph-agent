"""Control-plan isolation, observational-law parity and exact tiny enumeration."""
import importlib
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from models.factor import CANONICAL, FactorModel, make_factor
from models.factor_projection import factor_current
from research.ar5_voi_fixture import Branch, exact_q


class FactorControlTests(unittest.TestCase):
    def test_joint_law_root_projection_matches_passive_path(self):
        f=make_factor(f'{CANONICAL}-N5',7,3,6,19)
        rng=np.random.default_rng(83)
        x=rng.integers(0,2,7)
        for i in range(18):
            x[0]=1.;x[1]=0.
            before=pickle.dumps(f)
            projected=list(factor_current(f,x))
            self.assertEqual(before,pickle.dumps(f))
            for a,(mass,p,ps,_) in enumerate(projected):
                r=f.predict(x,a)[f.ch]
                _,_,_,pp,_,w,local,_,_=f._memo
                np.testing.assert_array_equal(mass,w)
                np.testing.assert_array_equal(p,f.expert*local+(1.-f.expert)*pp)
                np.testing.assert_array_equal(ps,r[f.slow])
            a=i%3; y=rng.integers(0,2,7); f.predict(x,a);f.learn(x,a,y);x=y

    def test_exact_two_step_coupled_vs_anticoupled(self):
        f=FactorModel(3,2,2,0,N=2,lags=(0,),own_lags=(0,)); f.attach_reward()
        h=f.reward;h.slots[:]=[0,1];h.cov.fill(0.)
        h.mean[:]=[[.2,0.,0.,.65],[.7,0.,0.,-.65]]
        # Same marginal for each bit. Only the shared component changes.
        coupled=np.array([[[0.,0.],[1.,1.]]]); anticoupled=np.array([[[0.,1.],[1.,0.]]])
        w=np.array([[.5,.5]]);empty=np.array([])
        roots=[(w,coupled,empty,coupled),(w,anticoupled,empty,anticoupled)]
        u=np.array([[.25,.25,.25,.5,.5],[.75,.25,.25,.5,.5]])
        q0,evsi=exact_q((Branch(.5,.2,(.2,.7)),Branch(.5,.2,(.85,.05))),1.)
        q1,_=exact_q((Branch(1.,.7,(.2,.7)),),1.)
        self.assertGreater(evsi,0.)
        with patch('models.factor_projection.factor_current',return_value=iter(roots)):
            np.testing.assert_allclose(f.two_step_values([1.,0.,0.],u),[q0,q1],rtol=0,atol=1e-14)

    def test_virtual_rollouts_preserve_live_state_and_rng(self):
        for name in ('v21','v22','v23','v28'):
            cls=importlib.import_module(f'agents.{name}').Agent
            a=cls(6,2,5,seed=48,draws=2)
            rng=np.random.default_rng(81); x=rng.integers(0,2,6).astype(float)
            for i in range(9):
                u=rng.random((2,a.factor.J+3+int(a.factor.reward_control)))
                before=pickle.dumps(a)
                q=a.factor.two_step_values(x,u)
                self.assertEqual(before,pickle.dumps(a),name)
                np.testing.assert_array_equal(q,a.factor.two_step_values(x,u))
                a.step(x);y=rng.integers(0,2,6).astype(float);a.learn(y);x=y

    def test_legacy_modules_poisoned_before_import(self):
        code='''
import sys
import importlib
class Poison:
    def find_spec(self, fullname, *args):
        if fullname in {"agents.v12", "agents.v11", "models.context_tree"}:
            raise AssertionError("legacy controller dependency: " + fullname)
sys.meta_path.insert(0,Poison())
for name in ('v21','v22','v23','v28'):
    agent=importlib.import_module('agents.'+name).Agent(5,2,4,seed=2,draws=2)
    x=[1.,0.,0.,0.,.5]
    for i in range(8):
        a=agent.step(x)
        y=[float(bool(a)),1.-a,i%2,0.,.45]
        agent.learn(y);x=y
'''
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
