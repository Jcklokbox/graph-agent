"""Teacher-data mechanism probe, not a DEV comparison or production controller."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from harness.journal import append
from envs.tasks import VARIANTS
from agents import agent_class, load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('variant');p.add_argument('--agent',default='v22');p.add_argument('--policy',choices=['uniform','demonstration'],default='uniform')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--train',type=int,default=600);p.add_argument('--test',type=int,default=200);a=p.parse_args()
    var=VARIANTS[a.variant]; env=var.make(100+a.seed);x=env.reset()
    brain=agent_class(load(a.agent))(var.n_inputs,var.n_actions,11,seed=a.seed)
    f=brain.factor; rng=np.random.default_rng(39099+a.seed)
    policy=var.oracle() if a.policy=='demonstration' and var.oracle is not None else None
    R=[]; rschool=[]; pp=[]; yy=[]; counts=np.zeros(var.n_actions,int)
    for i in range(a.train):
        action=int(rng.integers(var.n_actions)) if policy is None else int(policy(x))
        q=f.predict(x,action).copy(); y,r,*_=env.step(action)
        rschool.append(r); pp.append(q); yy.append(y);f.learn(x,action,y);x=y;counts[action]+=1
    pre=dict(features=f.reward.slots.tolist(), roles=np.flatnonzero(f.slow).tolist(),
             outcome=f.reward.mean.tolist(), counts=counts.tolist(),
             training_mean=float(np.mean(rschool)), training_positive=int(np.sum(np.array(rschool)==1.)))
    p,q=np.array(yy)[-200:],np.array(pp)[-200:]
    pre['channel_brier']=((p-q)**2).mean(0).tolist()
    rare=np.array(rschool)[-200:]==1.
    pre['rare_y']=p[rare].mean(0).tolist() if rare.any() else None
    pre['rare_forecast']=q[rare].mean(0).tolist() if rare.any() else None
    acts=[]
    for _ in range(a.test):
        action=brain.step(x);y,r,*_=env.step(action);brain.learn(y);x=y;R.append(r);acts.append(action)
    m=dict(**pre,test_mean=float(np.mean(R)), test_positive=int(np.sum(np.array(R)==1.)),
           test_shock=float(np.mean(np.array(R)==0.)), test_action_counts=np.bincount(acts,minlength=var.n_actions).tolist())
    append(a.out,dict(k=[os.environ['PG_TREE'],a.seed,f'{a.agent}|{a.variant}|{a.policy}|{a.train}|{a.test}'],m=m))
    print({k:v for k,v in m.items() if k not in ('outcome','channel_brier','rare_y','rare_forecast')},flush=True)


if __name__=='__main__':main()
