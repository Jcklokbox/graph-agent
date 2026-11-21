"""Read-only counterfactual of joint structure selection on actual replay."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build,run
from eval.scripts.ar6_parent_evidence import structure_scores
from harness.beacon import cli
from harness.journal import append
from models.factor_rollout import provisional_values


def real_atoms(h,c,j,m,l):
    """All three exhaustive branches use the very same actual tape window."""
    n=np.zeros((3,h.B));s=np.zeros_like(n)
    for _,_,value,by,features,onset in h.replay:
        if h.code(by)!=c:continue
        g=int(bool(features[l*h.J+m])) if onset[l*h.J+j] else 2
        a=min(h.B-1,int(value*h.B));n[g,a]+=1.;s[g,a]+=value
    return n,s


def projected(f,c,x,u,common_window=False):
    h=f.reward; choices={};rank={};sample={}
    for k in np.flatnonzero(h.control_mass>0.):
        score=structure_scores(h,k,common_window);b=score['best_compound'];a=score['best_unary']
        if b and b[0][0]>max(0.,a[0][0] if a else 0.):
            bf,j,m,l,_=b[0];d=int(np.flatnonzero(h.anchors==j)[0]);key=int(k)
            choices[key]=(j,m,l,d,bf);rank[str(key)]=[j,m,l,bf]
            if common_window:sample[key]=real_atoms(h,k,j,m,l)
    if not choices:return dict(choices=rank,q=provisional_values(f,x,u,False).tolist())
    original_rule=h.rule;original_splits=h.signed_rules;original_context=h.context_moment
    def rule():
        idx,w,p,*rest=original_rule();idx=idx.copy();w=w.copy();p=p.copy()
        for c,(j,_,l,_,_) in choices.items():
            mark=l*h.J+j;idx[c,0]=mark;w[c,0]=True
            p[c,:,0]=(h.on[c,:,mark]+.5)/(1.+h.type_count[c])
        return idx,w,p,*rest
    def splits(rule):
        ans=[]
        for j,m,l,s,g,d in original_splits(rule):
            s=s.copy();s[list(choices)]=False
            if s.any():ans.append((j,m,l,s,g,d))
        for c,(j,m,l,d,bf) in choices.items():
            s=np.zeros(h.C,bool);g=np.zeros(h.C);s[c]=True;g[c]=bf
            ans.append((j,m,l,s,g,d))
        return ans
    def moment(s,x,phase,mean,p,atoms):
        mu,p,r,count,group=original_context(s,x,phase,mean,p,atoms)
        for c,(j,m,l,d,_) in choices.items():
            outside=~phase[:,l*h.J+j]
            if c in sample:
                n,amount=sample[c];k=np.where(outside,2,x[:,l*h.J+m].astype(int))
                alpha=n[k]+1./h.B;mass=alpha/alpha.sum(1,keepdims=True);values=(amount[k]+h.center/h.B)/alpha
                p[:,c]=mass;r[:,c]=values;mu[:,c]=(mass*values).sum(1)
                count[:,c]=alpha.sum(1);group[:,c]=1+3*(d*h.L+l)+k
                continue
            if not outside.any():continue
            n=h.type_count[c]-h.ctx_atoms[c,d,l,m].sum(0)
            amount=h.type_sum[c]-h.ctx_amount[c,d,l,m].sum(0)
            alpha=n+1./h.B; mass=alpha/alpha.sum();value=(amount+h.center/h.B)/alpha
            p[outside,c]=mass;r[outside,c]=value;mu[outside,c]=mass@value
            count[outside,c]=alpha.sum();group[outside,c]=1+2*len(h.anchors)*h.L+d*h.L+l
        return mu,p,r,count,group
    h.rule=rule;h.signed_rules=splits;h.context_moment=moment
    try:q=provisional_values(f,x,u,False)
    finally:del h.rule,h.signed_rules,h.context_moment
    return dict(choices=rank,q=q.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--common-window',action='store_true');a=p.parse_args()
    for v in ('canc','canc-noise3'):
        for seed in (6,15):
            for period in (5,7):
                variant=VARIANTS[v];table=build(variant,seed,4,16,period)
                for arm in ('same','opposite','neutral'):
                    z=run(variant,seed,arm,table,period,lambda f,c,x,u: projected(f,c,x,u,a.common_window))
                    append(a.out,dict(k=[os.environ['PG_TREE'],seed,f'{v}|{arm}|period{period}'],m=z))
                    print(v,seed,period,arm,[(r['cancelled'],round(r['structure']['q'][0]-r['structure']['q'][1],3)) for r in z['results']],flush=True)


if __name__=='__main__':
    with cli():main()
