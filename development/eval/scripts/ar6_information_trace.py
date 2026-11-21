"""Replay exposed v48/v50 choices; separate opportunity representation from value."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np

from agents.v50 import Agent
from envs.tasks import SPLITS,VARIANTS
from harness.beacon import cli
from harness.journal import append
from models.factor_information import posterior_option
from models.factor_rollout import provisional_values


def run(variant, seed, enabled, steps):
    v=VARIANTS[variant]; _,_,eb,bb=SPLITS['dev'];e=v.make(eb+seed);x=e.reset()
    brain=Agent(v.n_inputs,v.n_actions,11,seed=bb+seed); f=brain.factor
    head=f.reward; internal=head.discovery_value; scratch={};samples=[];choices=[]
    def discover(q, op, rule, inp, phase):
        bonus=internal(q,op,rule,inp,phase)
        scratch['bonus']=bonus.tolist()
        scratch['cases']=[dict(freq=fr, paired=paired, cells=[dict(code=int(c),
                        reach=op[:,c].tolist(), n=float(n[c].sum()), s=float(a[c].sum()),
                        value=posterior_option(q,op[:,c],head.type_count[c],n[c],a[c],head.center,
                          -np.log(len(head.anchors)),whole_amounts=head.type_sum[c]))
                    for c in np.flatnonzero(head.control_mass>0.)])
            for n,a,fr,paired,_ in head._information_features(inp,phase)] if x[3] and x[1] else None
        return bonus if enabled else np.zeros_like(q)
    head.discovery_value=discover
    def predict(x,u,theta=None):
        q=provisional_values(f,x,u)
        if x[3] and x[1]:
            scratch['fixed']=provisional_values(f,x,u,False).tolist()
            scratch['anchors']=head.anchors.tolist()
            scratch['slots']=head.slots.tolist()
        return q
    f.two_step_values=predict
    for t in range(steps):
        a=brain.step(x);o,r,*_=e.step(a)
        if r!=.5:
            choices.append([t,int(x[3]),int(x[7]),a,r])
        if x[3] and x[1]:
            samples.append(dict(t=t,trap=int(x[7]),a=a,r=r,**scratch))
        scratch.clear();brain.learn(o);x=o
    return dict(samples=samples,outcomes=choices)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('variant')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--seed',type=int,nargs='+',default=[6,13])
    a=p.parse_args()
    for seed in a.seed:
        for enabled in (False,True):
            m=run(a.variant,seed,enabled,a.steps)
            append(a.out,dict(k=[os.environ['PG_TREE'],seed,f'{a.variant}|info={enabled}|{a.steps}'],m=m))
            events=[(t,r) for t,_,_,_,r in m['outcomes'] if r==1.]
            print(a.variant,seed,enabled,events[:3],len(events),flush=True)


if __name__=='__main__':
    with cli():
        main()
