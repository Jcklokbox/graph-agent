"""Matched Markov null and shared-latent structure positive control.

This is a development diagnostic for process discovery, not a replacement for
T1/d2/d3 gates. Truth lives only in this generator and scorer.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import os
import numpy as np
from harness.journal import append
from models.decompose import Decomposer


def loop(n, seed, steps, coupled):
    if n % 2:
        raise ValueError('n must be even')
    rng = np.random.default_rng(seed + 41000)
    ids = rng.permutation(n).reshape(-1,2)
    truth = {tuple(sorted(k)) for k in ids} if coupled else set()
    h = rng.random(n // (2 if coupled else 1)) < .5
    g = Decomposer(n+1,2,n,seed,windows=True)
    x=np.zeros(n+1)
    for _ in range(steps):
        # Equal marginal reset law, emissions and initialization; only
        # simultaneous hidden sharing changes between comparison arms.
        changed = rng.random(len(h)) < .08
        h[changed] = rng.random(len(h))[changed] < .5
        state = np.repeat(h, 2) if coupled else h
        z = np.logical_xor(state, rng.random(n) < .09)
        y=np.zeros(n+1); y[ids.ravel()]=z
        a=int(rng.integers(2)); g.predict(x,a);g.learn(x,a,y);x=y
    found = {tuple(sorted(p)) for grp in g.partition() for p in combinations(grp,2)}
    return dict(found=len(found), hit=len(found & truth), false=len(found-truth),
                possible=len(truth), groups=g.partition(), checked=g.proposal_count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out');p.add_argument('--n',type=int,default=12);p.add_argument('--n-seeds',type=int,default=4)
    p.add_argument('--steps',type=int,default=1800);a=p.parse_args()
    h=os.environ['PG_TREE']
    for seed in range(a.n_seeds):
        for c in (False,True):
            m=loop(a.n,seed,a.steps,c)
            append(a.out,dict(k=[h,seed,f'structure-control|{a.n}|{int(c)}|{a.steps}'],m=m))
            print(seed,c,{k:v for k,v in m.items() if k!='groups'},flush=True)


if __name__=='__main__':main()
