"""Development interventions on conditional credit (not a registered holdout).

Context background arm uses sub-saturating cue outcomes. Signed arm maintains
anchor probes during compound acquisition to distinguish credit from reversal.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from envs.phenomena import A, B, X, Y, US, CTX1, CTX2, HOME, VALUE, N_CH_V1, Schedule
from harness.journal import append
from models.factor import CANONICAL, make_factor


def trace(m, present, ctx):
    x = tuple(float(i in (*present, ctx)) for i in range(N_CH_V1))
    b = m._bits(x); loc = m._signature(b); j = m.ch.index(US)
    k = min(m.K, int(m.used.max()) + 1)
    p = m._regime_prior(loc, k); pr = m._cue_prior(p, b, k)
    w = np.exp(m.logw - m.logw.max()); w /= w.sum()
    mem = m._members(b, 0, m.hist, k)
    local, bg = m._conditional(mem, k, loc, not b[~m.slow].any())
    cols = {m.ch[s[0]]:i+len(m.own_lags)+m.A for i,s in enumerate(m.slots[j]) if s and s[1]==0}
    def weighted(table): return float(np.sum(w[:, None] * pr * table))
    return dict(pred=float(m.predict(x,0,hypothetical=True)[US]), local=weighted(local[...,j]),
                no_route=float(np.sum(w[:,None]*p*local[...,j])), bg=weighted(bg[...,j]),
                mix=float(m.expert[j]), place=list(loc),
                shared={s:float(m.shared_mu[j,u]) for s,u in cols.items()},
                diag={s:float(m.shared_cov[j,u,u]) for s,u in cols.items()},
                add_weight=weighted(m._E[:,:k,j,-1]/m._E[:,:k,j].sum(-1)),
                scope=np.sum(w[:, None]*pr,axis=0).tolist())


def episode(kind, treated, seed, particles, controlled=False):
    s = Schedule(seed,N_CH_V1)
    arms = []
    for place in (CTX1, CTX2, HOME): s.gap(35, place)
    if kind == 'background' and controlled:
        high, low = .34, .04
        p1, p2 = (high,low) if treated else (low,high)
        s.gap(80, CTX1, p1); s.gap(80, CTX2, p2)
        arms.append(len(s.obs)-1)
        for _ in range(4):
            s.gap(30, CTX1, p1)
            for _ in range(12):
                # Control the number of free-outcome opportunities exactly.
                s.gap(5, CTX1, p1); s._o([A],CTX1); s._o(ctx=CTX1,us=s.rng.random() < .62)
            s.gap(90, CTX2, p2)
            arms.append(len(s.obs)-1)
        panels = {'train':([A],CTX1), 'background':([],CTX1), 'neutral':([A],HOME),'base':([],HOME)}
    elif kind == 'background':
        # The known, nonsaturating outcome risk .6 fixes different cue excesses
        # under noisy OR, as long as a 0.30 background actually persists.
        src = CTX1 if treated else CTX2
        s.gap(100, src, .30)
        s.gap(100, CTX2 if treated else CTX1)
        arms.append(len(s.obs)-1)
        for _ in range(4):
            s.gap(20, src, .30)
            s.trials([A], .60, 5, ctx=CTX1)
            arms.append(len(s.obs)-1)
        panels = {'train':([A],CTX1), 'background':([],CTX1), 'neutral':([A],HOME),'base':([],HOME)}
    else:
        xs = X if treated else Y
        # All training arms have the same number and order of outcomes.
        hi, lo, transfer = (.88,.10,.62) if controlled else (1.,0.,.9)
        s.mixed([([A],hi),([A,X],lo),([Y],0.),([A,Y],hi)],22)
        arms.append(len(s.obs)-1)
        for _ in range(4):
            s.mixed([([A],hi),([A,X],lo),([A,Y],hi),([B,xs],transfer)],6 if controlled else 4)
            arms.append(len(s.obs)-1)
        panels = {name:(cues,CTX1) for name,cues in [('AX',[A,X]),('AY',[A,Y]),('B',[B]),('BX',[B,X]),('BY',[B,Y])]}
    f = make_factor(f'{CANONICAL}-N{particles}',N_CH_V1,2,VALUE,seed)
    measures = {}
    for i,(x,y) in enumerate(zip(s.obs,s.obs[1:]),1):
        f.predict(x,0);f.learn(x,0,y)
        if i in arms:
            measures[str(arms.index(i))] = {name:trace(f,cues,ctx) for name,(cues,ctx) in panels.items()}
    return measures


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('out');p.add_argument('kind',choices=['background','signed'])
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--particles', type=int,default=200)
    p.add_argument('--controlled',action='store_true');a=p.parse_args()
    start = 74700 if a.controlled else 74100
    for seed in range(start,start+a.seeds):
        for arm in (False,True):
            m=episode(a.kind,arm,seed,a.particles,a.controlled)
            tag = a.kind + ('-controlled' if a.controlled else '')
            append(a.out,dict(k=[os.environ['PG_TREE'],seed,f'{tag}|{int(arm)}|{a.particles}'],m=m))
            print(a.kind,seed,arm,{k:round(v['pred'],3) for k,v in m['4'].items()},flush=True)


if __name__=='__main__':
    from harness.beacon import cli

    with cli():
        main()
