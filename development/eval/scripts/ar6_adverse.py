"""Exposed matched adverse interventions after actual positive contact."""
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


def build(v,seed,n,calibration=0):
    _,_,eb,_=SPLITS['dev']; env=v.make(eb+seed);x=env.reset()
    pos,neg,probe,cal=[],[],[],{}; cycles=0
    for t in range(360):
        if x[3]:
            if cycles<calibration:
                a=0 if cycles%2==0 else 2
                cal.update({t+1:a,t+2:1,t+3:2-a})
                cycles+=1
            elif len(pos)<n:
                if not x[7]:pos.append(t)
            elif len(neg)<n:
                if x[7]:neg.append(t)
            else:
                probe.append((t,bool(x[7])))
                if sum(b for _,b in probe)>=2 and sum(not b for _,b in probe)>=2:
                    return pos,neg,probe,cal
        x=env.step(1)[0]
    raise ValueError('tape has insufficient reachable matched cue windows')


def run(v,seed,arm,table):
    _,_,eb,bb=SPLITS['dev'];env=v.make(eb+seed);x=env.reset()
    agent=agent_class(load('v58'))(v.n_inputs,v.n_actions,11,seed=bb+seed);f=agent.factor;h=f.reward
    pos,neg,probe,cal=table;wp=dict(probe);moves=dict(cal)
    for t in pos:moves.update({t:0,t+1:2})
    if arm in ('same','opposite'):
        action=0 if arm=='same' else 2
        for t in neg:moves.update({t:action,t+1:2-action})
    events=Counter();sensor=[];results=[];prefix_r=[]
    for t in range(probe[-1][0]+1):
        if t in wp:
            if x[1]!=1.:raise ValueError('probe at different initial position')
            u=np.random.default_rng(np.random.SeedSequence([seed,t,9001])).random((5,f.J+3+int(f.reward_control)))
            rule=h.rule();b=f._bits(x);c=int(h.code(f._bits(np.eye(v.n_inputs)[0])));right=int(h.code(f._bits(np.eye(v.n_inputs)[2])))
            feat=h.input(b,f.hist)[1:];phase=h.phase(b,f.hist)
            mu,prob,*_=h.likelihood(rule,feat[None,:]);pairs=h.signed_rules(rule)
            means,pr,atoms,*_=h.context_moment(pairs,feat[None,:],phase[None,:],mu,prob,rule[3])
            soft,_,_,used=h.refine_root(pr[0],atoms[0],rule,feat,phase)
            d=np.flatnonzero(h.anchors==3)
            observed=[] if not len(d) else h.ctx_n[c,d[0],0,7].tolist()
            reach=h.all_transitions()[:,int(h.code(b)),c].tolist()
            fixed=provisional_values(f,x,u,False);q=provisional_values(f,x,u)
            results.append(dict(t=t,cancelled=wp[t],fixed=fixed.tolist(),q=q.tolist(),
              left_mean=float(means[0,c]),left_base=float(mu[0,c]),left_soft=float(soft[c]),
              soft_used=bool(used[c]),left_cells=observed,reach=reach,ctrl=h.control_mass.tolist(),
              left=int(c),right=right,slots=h.slots.tolist(),parent=rule[0][c][rule[1][c]].tolist(),
              splits=[dict(j=int(j),m=int(m),lag=int(l),n=h.ctx_n[c,d,l,m].tolist(),u=h.ctx_s[c,d,l,m].tolist())
                      for j,m,l,s,g,d in pairs if s[c]]))
        a=moves.get(t,1);y,r,*_=env.step(a)
        if t in cal:prefix_r.append(float(r))
        if t in pos:events['positive' if r==1. else 'invalid']+=1
        if t in neg:events['adverse' if r==0. else 'neutral' if r==.5 else 'invalid']+=1
        sensor.append([int(x[3]),int(x[7]),*map(int,x[12:])])
        f.predict(x,a);f.learn(x,a,y);x=y
    if (events['invalid'] or events['positive']!=len(pos)
        or events['adverse']!=(len(neg) if arm!='neutral' else 0)
        or events['neutral']!=(len(neg) if arm=='neutral' else 0)):
        raise ValueError('actual intervention payoff differs from plan')
    return dict(events=dict(events),results=results,sensor=sensor,positive_ticks=pos,adverse_ticks=neg,calibration_rewards=prefix_r)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--variant',default='canc-noise3',choices=['canc','canc-noise3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[6,7,12,15]);p.add_argument('--dose',type=int,default=4)
    p.add_argument('--calibration',type=int,default=0,help='shared balanced noncue dwell cycles')
    a=p.parse_args();v=VARIANTS[a.variant];tree=os.environ['PG_TREE']
    for s in a.seeds:
        table=build(v,s,a.dose,a.calibration);sensor=None;rewards=None
        for arm in ('neutral','opposite','same'):
            trace=run(v,s,arm,table)
            if sensor is None:sensor=trace['sensor'];rewards=trace['calibration_rewards']
            elif sensor!=trace['sensor'] or rewards!=trace['calibration_rewards']:
                raise ValueError('unmatched exogenous tape or calibration payoff')
            append(a.out,dict(k=[tree,s,f'{a.variant}|{arm}|{a.dose}|cal{a.calibration}'],m=trace))
            print(s,arm,trace['events'],len(trace['results']),flush=True)


if __name__=='__main__':
    with cli():main()
