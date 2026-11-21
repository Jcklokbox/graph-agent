"""Cadenced prequential T1 evidence: partitions, candidate scores and member/channel loss."""
from __future__ import annotations
import argparse
import os
import numpy as np
from envs.family import WORLD_SETS, WorldEnv, Filter
from harness.journal import append
from models.factor import CANONICAL, make_factor


def trace(out, suite, world, seed, steps):
    model= WORLD_SETS[suite]()[world]
    rng=np.random.default_rng(2000+seed); env=WorldEnv(model,1000+seed)
    f=Filter(model); m=make_factor(CANONICAL,model.n_channels,model.n_actions,model.value_index,seed)
    x=env.reset(); f.observe(x)
    loss=np.zeros((4,m.J)); n=0; merges=0
    for t in range(steps):
        a=int(rng.integers(model.n_actions))
        o=f.predict(a)[m.ch]; q=m.predict(x,a)[m.ch]
        _,_,_, pp, _, w, pz, _, _=m._memo
        raw=np.einsum('nk,nkj->j',w,pz)/w.sum()
        y=np.asarray(env.step(a)); bits=y[m.ch]
        qs=np.clip(np.asarray((q,raw,pp,o)),1e-6,1-1e-6)
        loss-=bits*np.log(qs)+(1-bits)*np.log1p(-qs); n+=1
        m.learn(x,a,y); f.update(a,y);x=tuple(y)
        if (t+1) % max(1,steps//8)==0 or len(m.proc.merges)!=merges:
            merges=len(m.proc.merges)
            row=dict(step=t+1,groups=m.proc.partition(),scope=m.trace(),candidates=[
                {k:c[k] for k in ('g1','g2','d','bar') if k in c} for c in m.proc.cand],
                labels=['mixed','local-raw','process','oracle'], average_ll=(loss/n).tolist())
            append(out,dict(k=[os.environ['PG_TREE'],seed,f'{suite}|{world}|{t+1}'],m=row))
            print(world,seed,t+1, row['groups'],np.round(loss.mean(1)/n,4).tolist(),flush=True)
            loss.fill(0); n=0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out');p.add_argument('suite');p.add_argument('world');
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    trace(a.out,a.suite,a.world,a.seed,a.steps)


if __name__=='__main__':main()
