"""Read-only forecast of the factor's current latent state, for trial branches.

Parameter arrays are shared if current; dirty bounded HMM parameters are
reconstructed into local arrays. This never prepares a real learning cache.
"""
from __future__ import annotations

import numpy as np
from models.decompose import Window
from models.kernels import hmm_params, hmm_forward


def scratch_params(h):
    if not h._dirty:
        return h._trans, h._emit
    t,e=np.empty_like(h.ntr),np.empty_like(h.non)
    hmm_params(h.ntr,h.non,h.ntot,h.alpha,h.sticky,h.shrink,t,e)
    return t,e


def scratch_lik(x,e):
    return np.exp(np.log(np.maximum(np.where(np.asarray(x)>.5,e,1.-e),1e-300)).sum(-1))


def window_marg(h,a,hist):
    best=None
    for j in range(1,min(len(hist),h.WIN)+1):
        c=h.t.get((tuple(hist[-j:]),a))
        if c is not None and (best is None or c[1]>=h.MIN):
            best=c
    return np.full(h.m,.5) if best is None else (best[0]+1.)/(best[1]+2.)


def process_current(d,x):
    """Return all current-action projections of the process expert."""
    x=np.asarray(x,float); out=np.full((d.A,d.n),.5)
    beliefs={}
    for g in d.groups:
        obs=x[g.chans]; parts=[]
        w=np.exp(g.logw-g.logw.max());w/=w.sum()
        for h in g.hs:
            if isinstance(h,Window):
                hist=h.hist if g.started else [tuple(obs>.5)]
                parts.append([window_marg(h,a,hist) for a in range(d.A)])
                continue
            t,e=scratch_params(h)
            b=h.b
            if not g.started:
                ll=scratch_lik(obs,e);b=ll/ll.sum()
            results=[hmm_forward(b,t[a],e,True) for a in range(d.A)]
            beliefs[h]=[p for p,_ in results]
            parts.append([q for _,q in results])
        out[:,g.chans]=sum(wk*np.asarray(q) for wk,q in zip(w,parts))
    if d.conj:
        for j,cs in d.cj.items():
            if not cs:
                continue
            from models.decompose import CONJ_PEN
            lw=np.array([0.]+[c.adv-CONJ_PEN for c in cs]);w=np.exp(lw-lw.max());w/=w.sum()
            def marginal(c,a):
                if c.ho not in beliefs or c.hg not in beliefs:
                    return .5
                e=(c.n1+.5)/(c.n+1.)
                return float(beliefs[c.ho][a]@e@beliefs[c.hg][a])
            for a in range(d.A):
                out[a,j]=w[0]*out[a,j]+sum(wk*marginal(c,a) for wk,c in zip(w[1:],cs))
    return out


def factor_current(f,x):
    """Current joint product mixture and slow marginals, for every action."""
    b=f._bits(x);fast=~f.slow;quiet=not b[fast].any()
    pp=None if f.proc is None else process_current(f.proc,x)[:,f.ch]
    kk=min(f.K,int(f.used.max())+1); place=f._signature(b)
    prior=f._cue_prior(f._regime_prior(place,kk),b,kk)
    w=np.exp(f.logw-f.logw.max())[:,None]*prior
    js=np.flatnonzero(f.slow);c=f.slow_counts[js,b[js].astype(int)]
    slow_p=(c[:,1]+.5)/(c.sum(1)+1.)
    for a in range(f.A):
        mem=None if quiet and len(js) else f._members(b,a,f.hist,kk)
        local,_=f._conditional(mem,kk,place,quiet)
        mix=local if pp is None else f.expert*local+(1.-f.expert)*pp[a]
        yield w,mix,slow_p,local


def finite_marginal(w, q, slow, slow_p, quiet, indices):
    """Exact low-dimensional marginal of the deployed joint clock law.

    Caller bounds `indices`. Omitting fast channels changes the zero-observation
    branch: subtract the full fast-null component from the restricted fast-null
    pattern before putting back the shared clock's zero mass.
    """
    indices=np.asarray(indices,int)
    n=len(indices)
    patterns=(np.arange(2**n)[:,None]&(1<<np.arange(n)))!=0
    weights=w.reshape(-1);weights=weights/weights.sum()
    flat=q.reshape(-1,len(slow));fast=~slow
    retained_fast=~slow[indices]
    fast_idx=indices[retained_fast]
    slow_idx=indices[~retained_fast]
    fpattern=patterns[:,retained_fast]
    p=flat[:,fast_idx]
    z=np.prod(np.where(fpattern[:,None,:],p,1.-p),axis=2)
    if quiet and fast.any() and len(fast_idx):
        omitted=fast.copy();omitted[fast_idx]=False
        with np.errstate(divide='ignore'):
            log_idle=np.log1p(-flat[:,fast]).sum(1)
            log_idle_omitted=np.log1p(-flat[:,omitted]).sum(1)
        event=-np.expm1(log_idle)
        quiet_mass=weights@np.exp(log_idle)
        zero=~fpattern.any(1)
        z[zero]*=-np.expm1(log_idle_omitted)
        # Per-member conditioning happens before mixing, just as in sampler.
        condition=np.divide(z,event,out=np.zeros_like(z),where=event>0.)
        mass=condition@weights*(weights@event)+zero*quiet_mass
    else:
        mass=z@weights
    if len(slow_idx):
        r=np.zeros(len(slow));r[slow]=slow_p
        mass*=np.prod(np.where(patterns[:,~retained_fast],r[slow_idx],1.-r[slow_idx]),axis=1)
    return patterns,mass/mass.sum()
