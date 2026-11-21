"""Exposed diagnosis of model values on actual late deviations; no oracle training."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from collections import Counter
import numpy as np
from agents import load,agent_class
from envs.tasks import VARIANTS,SPLITS
from harness.journal import append
from models.factor_rollout import provisional_values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('variant')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--model',default='v44');a=p.parse_args()
    var=VARIANTS[a.variant]; _,_,eb,bb=SPLITS['dev'];env=var.make(eb+a.seed);x=env.reset()
    brain=agent_class(load(a.model))(var.n_inputs,var.n_actions,11,seed=bb+a.seed)
    f=brain.factor; trial=[]; diag={}; previous=f.two_step_values; shocks=Counter(); event_miss=Counter()
    def observe(x,u,theta=None):
        diag['u']=u
        diag['q']=previous(x,u,theta)
        return diag['q']
    f.two_step_values=observe
    for i in range(var.steps):
        chosen=brain.step(x);y,r,*_=env.step(chosen)
        if i>=var.steps//3 and r<.5:
            key=f"{int(x[:3].argmax())}->{int(y[:3].argmax())}|cue={int(x[3])}|ctx={int(x[7])}"
            shocks[key]+=1
            if len(trial)<35:
                q=provisional_values(f,x,diag['u'],False)
                trial.append(dict(t=i,x=x.tolist(),choice=chosen,reward=r,q=diag['q'].tolist(),fixed_q=q.tolist(),next=y.tolist()))
        if i>=var.steps//3 and x[3] and not x[7] and r<1.:
            event_miss[str(int(x[:3].argmax()))]+=1
        brain.learn(y);x=y
    append(a.out,dict(k=[os.environ['PG_TREE'],a.seed,f'{a.model}|{a.variant}|deviations'],
                      m=dict(shocks=dict(shocks),event_miss=dict(event_miss),examples=trial)))

if __name__=='__main__':main()
