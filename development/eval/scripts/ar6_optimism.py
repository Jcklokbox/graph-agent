"""Exposed counterfactual decompositions and surgical optimism-policy ablations."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from time import monotonic

import numpy as np

from agents.v48 import Agent
from envs.tasks import SPLITS, VARIANTS
from eval.benchmark import _oracle_return, _policy_return, _window
from harness.beacon import cli
from harness.journal import append, load
from models.factor_rollout import provisional_values

CHOICES={'both': (True,True), 'root': (True,False), 'future': (False,True)}


def episode(variant, seed, choice):
    v=VARIANTS[variant]; _,_,eb,bb=SPLITS['dev']; e=v.make(eb+seed)
    ag=Agent(v.n_inputs,v.n_actions,11,seed=bb+seed)
    f=ag.factor; x=e.reset(); notes=[]; results=[]; diag={}; t0=monotonic(); first=None; prefix=Counter()
    def run(x,u,theta=None):
        r,c=CHOICES[choice]
        lead=provisional_values(f,x,u,root_explore=r,continuation_explore=c)
        if variant.startswith('canc') and x[3] and x[1] and len(notes)<36 and choice=='both':
            diag.update({name:provisional_values(f,x,u,root_explore=a,continuation_explore=b).tolist()
                         for name,(a,b) in CHOICES.items() if name!=choice}, both=lead.tolist())
        return lead
    f.two_step_values=run
    for t in range(v.steps):
        a=ag.step(x); y,r,*_=e.step(a); results.append(r)
        if first is None and r==1.:
            first=t
        if t<240:
            prefix[str(r)]+=1
        if diag:
            notes.append(dict(t=t,trap=int(x[7]),a=a,r=r,qs=dict(diag)))
            diag.clear()
        ag.learn(y); x=y
    lo,hi=_window(variant,v.steps); val=float(np.mean(results[lo:hi])); stay=_policy_return(variant,eb+seed,lambda o:1)
    oracle=_oracle_return(variant,eb+seed)
    return dict(score=(val-stay)/(oracle-stay), first=first, prefix=dict(prefix), notes=notes,
                dt=monotonic()-t0, steps=v.steps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path);p.add_argument('--variant',nargs='+',required=True)
    p.add_argument('--seed',type=int,nargs='+',default=[0,6,13]);p.add_argument('--choice',nargs='+',choices=CHOICES,default=list(CHOICES))
    a=p.parse_args(); rec,dup,bad=load(a.out)
    if dup or bad:
        raise ValueError('bad journal')
    for i in a.seed:
        for v in a.variant:
            for k in a.choice:
                key=(os.environ['PG_TREE'],i,f'{v}|{k}')
                if key not in rec:
                    r=episode(v,i,k);append(a.out,dict(k=key,m=r));print(key,round(r['score'],3),r['first'],flush=True)


if __name__=='__main__':
    with cli():
        main()
