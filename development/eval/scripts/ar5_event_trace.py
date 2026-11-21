"""Exposed mechanism probe: ordinal cue-window support and factor role trace."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from collections import Counter
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class,load
from harness.journal import append


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('variant')
    p.add_argument('--model',default='v28');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=200); a=p.parse_args()
    v=VARIANTS[a.variant];_,_,eb,bb=SPLITS['dev'];e=v.make(eb+a.seed);x=e.reset()
    brain=agent_class(load(a.model))(v.n_inputs,v.n_actions,11,seed=bb+a.seed)
    f=brain.factor; head=f.reward; reward=Counter(); trials=Counter(); examples=[]; changes=[]
    for t in range(a.steps):
        c=int(head.code(f._bits(x))); slots=head.slots.copy()
        ctx=np.r_[x[3:5],*[b[3:5] for b in f.hist[-2:][::-1]]]
        chosen=brain.step(x);y,r,*_=e.step(chosen); brain.learn(y)
        trials[tuple(map(int,np.r_[x[:5],ctx[2:],chosen,y[:3]]))]+=1
        reward[str(r)]+=1
        if r==1. or (len(examples)<30 and ctx.any()):
            examples.append([t,c,list(map(int,x[:8])),list(map(int,ctx)),chosen,list(map(int,y[:8])),r])
        if (slots!=head.slots).any():changes.append([t,head.slots.tolist()])
        x=y
    data=dict(rewards=dict(reward),examples=examples,roles=changes,eg=head.evidence.tolist(),
              transitions=head.trans.tolist() if head.k<=3 else head.trans[:,:8,:8].tolist(),
              visits=head.control_mass.tolist(),trials=[[list(k),n] for k,n in sorted(trials.items())])
    if hasattr(head,'evidence_type'):
        idx=np.argsort(-head.evidence_type)[:,:6]
        data.update(top_features=[list(zip(j.tolist(),s.tolist())) for j,s in
                    zip(idx,np.take_along_axis(head.evidence_type,idx,1))],type_count=head.type_count.tolist())
    append(a.out,dict(k=[os.environ['PG_TREE'],a.seed,f'{a.model}|{a.variant}|{a.steps}'],m=data))


if __name__=='__main__': main()
