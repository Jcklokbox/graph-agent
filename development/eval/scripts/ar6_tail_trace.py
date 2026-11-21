"""Selected exposed trajectories: attribute late mistakes to exploration or mean."""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import numpy as np
from agents import agent_class,load
from envs.tasks import SPLITS,VARIANTS
from harness.beacon import cli
from harness.journal import append
from models.factor_rollout import provisional_values


def trace(variant,seed,model,steps):
    v=VARIANTS[variant];_,_,eb,bb=SPLITS['dev'];e=v.make(eb+seed);x=e.reset()
    a=agent_class(load(model))(v.n_inputs,v.n_actions,11,seed=bb+seed);f=a.factor;h=f.reward
    probe={};events=Counter();rows=[];r_sum=0.;first=None
    original=h.discovery_value
    def discovery(*args,**kw):
        bonus=original(*args,**kw);probe['bonus']=bonus.copy();return bonus
    h.discovery_value=discovery
    def plan(x,u,theta=None):
        q=provisional_values(f,x,u);probe['q']=q.copy();probe['u']=u
        return q
    f.two_step_values=plan
    for t in range(steps):
        action=a.step(x);y,r,*_=e.step(action)
        if r==1. and first is None:first=t
        if t>=steps-200:
            tag='cancelled' if x[3] and x[7] else ('valid' if x[3] else 'none')
            pos=int(np.argmax(x[:3]));land=int(np.argmax(y[:3]));events[f'{tag}|{pos}|{land}|{r}']+=1;r_sum+=r
            if r<.5 or (tag=='valid' and r<1.):
                c=int(h.code(f._bits(y)));rule=h.rule();b=probe.get('bonus',np.zeros(f.A));q=probe['q'];fixed=provisional_values(f,x,probe['u'],False)
                rows.append(dict(t=t,event=tag,pos=pos,land=land,r=r,a=action,positive=int(h.type_count[c,-1]),
                  q=q.tolist(),bonus=b.tolist(),fixed=fixed.tolist(),scope=h.slots.tolist(),parent=rule[0][c][rule[1][c]].tolist()))
        probe.clear();a.learn(y);x=y
    return dict(events=dict(events),rows=rows,first=first,reward=r_sum)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--model',default='v54')
    p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--variant',default='canc-noise3');p.add_argument('--steps',type=int,default=0)
    a=p.parse_args()
    for s in a.seeds:
        result=trace(a.variant,s,a.model,a.steps or VARIANTS[a.variant].steps)
        append(a.out,dict(k=[os.environ['PG_TREE'],s,a.model+'|'+a.variant],m=result))
        print(s,result['first'],result['reward'],len(result['rows']),flush=True)


if __name__=='__main__':
    with cli():main()
