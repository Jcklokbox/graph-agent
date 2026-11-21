"""Control-plan isolation, observational-law parity and exact tiny enumeration."""
import importlib
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from models.factor import CANONICAL, FactorModel, make_factor
from models.factor_projection import factor_current, finite_marginal
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
        for name in ('v21','v22','v23','v28','v37','v38'):
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
for name in ('v21','v22','v23','v28','v37','v38'):
    agent=importlib.import_module('agents.'+name).Agent(5,2,4,seed=2,draws=2)
    x=[1.,0.,0.,0.,.5]
    for i in range(8):
        a=agent.step(x)
        y=[float(bool(a)),1.-a,i%2,0.,.45]
        agent.learn(y);x=y
'''
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)


    def test_exact_marginal_retains_hidden_idle_mass(self):
        import itertools
        q=np.array([[[.08,.19,.15,.10],[.69,.81,.34,.40]]]);w=np.array([[.4,.6]])
        slow=np.array([0,0,1,0],bool);sp=np.array([.33]);indices=[1,2]
        for quiet in (False,True):
            expected=np.zeros(4);idle=np.prod(1.-q[0][:,~slow],axis=1)
            shared_idle=float(w[0]@idle)
            for marks in itertools.product((0,1),repeat=4):
                b=np.asarray(marks,bool);lik=np.prod(np.where(b[~slow],q[0][:,~slow],1.-q[0][:,~slow]),axis=1)
                if quiet:
                    sensor=shared_idle if not b[~slow].any() else (1.-shared_idle)*float(w[0]@(lik/(1.-idle)))
                else:
                    sensor=float(w[0]@lik)
                expected[marks[1]+2*marks[2]]+=sensor*(sp[0] if marks[2] else 1.-sp[0])
            _,actual=finite_marginal(w,q,slow,sp,quiet,indices)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-15)
        p=np.array([[[1e-30,2e-30]]]);_,mass=finite_marginal(np.ones((1,1)),p,np.zeros(2,bool),np.empty(0),True,[0])
        self.assertAlmostEqual(mass[1]/1e-30,1.,places=13)

    def test_projected_control_tree_matches_explicit_atom_assimilation(self):
        from agents.v33 import DwellEventReward
        from models.factor_rollout import provisional_values
        f=FactorModel(3,2,2,0,N=2,lags=(0,),own_lags=(0,))
        h=f.reward=DwellEventReward(f.J,f.A,f.D)
        h.slots[:]=[0,-1];h.trans[:,:,0]=3.;h.trans[:,:,1]=2.
        h.trans[0,0,0]=20.;h.trans[1,0,1]=20.
        h.type_count[0,[1,15]]=[2.,2.];h.type_sum[0,[1,15]]=[.20,1.96]
        h.type_count[1,[5,12]]=[2.,2.];h.type_sum[1,[5,12]]=[.70,1.55]
        h.control_mass=h.type_count.sum(1)
        prior=h.type_count+1./h.B; moment=h.type_sum+h.center/h.B
        theta=moment/prior;prob=prior/prior.sum(1,keepdims=True)
        cur=moment.sum(1)/prior.sum(1);move=h.all_transitions();root=np.array([0.,0.,.5]); expected=[]
        for a in range(f.A):
            v=0.
            for c in h.ci:
                if move[a,0,c]==0.:continue
                v+=move[a,0,c]*cur[c]
                for r,p in zip(theta[c],prob[c]):
                    post=cur.copy();post[c]=(moment[c].sum()+r)/(prior[c].sum()+1.)
                    v+=move[a,0,c]*p*np.max(move[:,c]@post)
            expected.append(v)
        original=pickle.dumps(f)
        q=provisional_values(f,root,np.zeros((2,5)),optimistic=False)
        self.assertEqual(original,pickle.dumps(f))
        np.testing.assert_allclose(q,expected,rtol=0,atol=2e-15)


    def test_action_correlation_alone_does_not_make_a_pulse_actuator(self):
        from agents.v33 import DwellEventReward
        h=DwellEventReward(6,3,2)
        rng=np.random.default_rng(841)
        # The pulse predicts an action under the collected policy, but cannot
        # be sustained by any choice; the dwell witnesses are the actuators.
        x=np.zeros(6);x[1]=1.;hist=[]
        for t in range(550):
            a=0 if x[3] else int(rng.choice([1,2])) if rng.random()<.85 else 0
            p=int(np.argmax(x[:3]));q=int(np.clip(p+a-1,0,2))
            y=np.zeros(6);y[q]=1.;y[4]=x[3];y[3]=float(not x[3] and not x[4] and rng.random()<.23)
            y[5]=rng.integers(2)
            h.learn_joint(x,hist,a,.45,y);hist=(*hist[-3:],x);x=y
        self.assertGreater(h.evidence[4],np.log1p(h.J))
        selected=set(h.slots[h.slots>=0])
        self.assertEqual(selected,{0,1,2})

    def test_relabelled_actuation_switches_without_inheriting_value_hypothesis(self):
        from agents.v38 import RelabelledModeReward
        h=RelabelledModeReward(3,2,1);h.slots[0]=0
        h.trans[0,:,0]=240.;h.trans[1,:,1]=240.
        h.trans_total[:]=h.trans.sum(-1);h.control_mass[:2]=240.
        h.type_count[1,15]=240.;h.type_sum[1,15]=235.
        b=np.zeros(3);bn=b.copy();bn[0]=1.
        n=np.zeros(h.F-1);h._mode_steps=450
        for i in range(4):
            h._record(b,0,.34,bn,n,n)
            if h.switches:break
        self.assertEqual(h.switches,1)
        self.assertGreater(h.trans[0,0,1],h.trans[0,0,0])
        self.assertEqual(h.type_count[1,15],0.)
        self.assertEqual(h.archive_mode['type_count'][1,15],240.)
