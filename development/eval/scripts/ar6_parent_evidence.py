"""Exposed evaluator: compare exact Dirichlet likelihoods across parent families."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build, run
from harness.beacon import cli
from harness.journal import append
from models.factor_information import log_marginal


def structure_scores(h,c):
    total=h.type_count[c];prior=1./h.B;baseline=log_marginal(total,prior)
    permitted=np.ones(h.J,bool);permitted[h.slots[h.slots>=0]]=False
    u=[];mult=[];n_unary=int(permitted.sum())*h.L
    for j in np.flatnonzero(np.tile(permitted,h.L)):
        on=h.on[c,:,j];off=total-on
        if min(on.sum(),off.sum())>0.:
            u.append((float(log_marginal(on,prior)+log_marginal(off,prior)-baseline-np.log(n_unary)),int(j)))
    base=(h.context_x+.5)/(1.+h.context_t);n_pair=0
    for d,j in enumerate(h.anchors):
        if j<0 or not permitted[j]:continue
        rate=(h.pair_on[d]+.5)/(1.+h.anchor_n[d]);eligible=permitted&(rate>base)&(h.assoc[d]>np.log1p(h.J));eligible[j]=False
        for m in np.flatnonzero(eligible):
            n_pair+=h.L
            for l in range(h.L):
                bins=h.ctx_atoms[c,d,l,m];outside=total-bins.sum(0)
                if np.any(outside<0.):raise ValueError('paired context outlives reward base')
                if min(bins[0].sum(),bins[1].sum(),outside.sum())>0.:
                    gain=sum(log_marginal(t,prior) for t in (*bins,outside))-baseline
                    mult.append((float(gain),int(j),int(m),l,bins.sum(1).tolist()))
    pair=[(float(ell-np.log(n_pair)),j,m,l,n) for ell,j,m,l,n in mult]
    return dict(best_unary=sorted(u,reverse=True)[:4],best_compound=sorted(pair,reverse=True)[:4],
                unary_candidates=n_unary,compound_candidates=n_pair)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    a=p.parse_args()
    for v in ('canc','canc-noise3'):
        for seed in (6,7,12,15):
            for period in (5,7):
                variant=VARIANTS[v];table=build(variant,seed,4,16,period)
                result=run(variant,seed,'same',table,period,lambda f,c,x,u: structure_scores(f.reward,c))
                key=[os.environ['PG_TREE'],seed,f'{v}|adverse|period{period}']
                append(a.out,dict(k=key,m=result));print(seed,key[-1],result['results'][0]['structure'],flush=True)


if __name__=='__main__':
    with cli():main()
