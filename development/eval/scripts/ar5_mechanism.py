"""Development-only accessors for opportunity, ancestry and realized forecast."""
from __future__ import annotations
import argparse
import os
import numpy as np
from harness.journal import append
from envs.phenomena import SETS, US, VALUE
from models.factor import CANONICAL, make_factor


def signature(m, x, a=0):
    b = m._bits(x)
    kk = min(m.K, int(m.used.max()) + 1)
    loc = m._signature(b)
    prior = m._regime_prior(loc, kk)
    bank = m._members(b, a, m.hist, kk)
    p, bg = m._conditional(bank, kk, loc, not b[~m.slow].any())
    w = np.exp(m.logw - m.logw.max()); w /= w.sum()
    w = w[:, None] * prior
    j = m.ch.index(US)
    z = w.sum(0)
    pred = (w * p[:, :, j]).sum(0)
    fresh = np.arange(kk)[None, :] >= m.used[:, None]
    return {'p':float(pred.sum()),'z': z.tolist(), 'contrib':pred.tolist(),
            'fresh':float((w*fresh).sum()), 'bg':float((w*bg[:,:,j]).sum()),
            'procedure_prior':float(m.expert[j]), 'roles':list(map(int,np.flatnonzero(m.slow))),
            'place':list(loc),'births':m.births, 'live':int(m.used.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    w, n, _ = SETS['v1']
    for group in w['pree'].groups:
        s = w['pree'].build(group, a.seed)
        m = make_factor(CANONICAL, n, 2, VALUE, a.seed)
        nc = 0
        keys = [0, 1, 2, 3, 9, 19, 29, 38, 39, 40, 41, 42, 43, 44, 45]
        for i,(x,y) in enumerate(zip(s.obs,s.obs[1:])):
            if x[0] > .5:
                if nc in keys:
                    append(a.out, {'k':[os.environ['PG_TREE'],a.seed,f'pree|{group}|cue{nc}'],
                                   'm':dict(pre=signature(m,x),y=y[US], t=i)})
                nc+=1
            m.learn(x,0,y)
        probe = s.probes[-1][-1]
        append(a.out,{'k':[os.environ['PG_TREE'],a.seed,f'pree|{group}|probe'],
                      'm':dict(pre=signature(m,probe),y=None,t=len(s.obs)-1)})


if __name__ == '__main__':
    main()
