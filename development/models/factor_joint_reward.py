"""Common-support joint reward structure selection for actual learned states."""
from math import lgamma

import numpy as np
from numba import njit


@njit(cache=True)
def _select(feat, phase, y, code, unary, pairs, C, B, gamma, norm):
    """Compare complete partitions on one support window, grouped by arrival."""
    pooled = np.zeros((C, B), np.int64)
    uon = np.zeros((C, len(unary), B), np.int64)
    pn = np.zeros((C, len(pairs), 3, B), np.int64)
    ps = np.zeros((C, len(pairs), 3, B))
    for i in range(len(y)):
        c = code[i]
        r = min(B - 1, int(y[i] * B))
        pooled[c,r] += 1
        for u in range(len(unary)):
            if feat[i,unary[u]]:
                uon[c,u,r] += 1
        for k in range(len(pairs)):
            g = int(feat[i,pairs[k,1]]) if phase[i,pairs[k,0]] else 2
            pn[c,k,g,r] += 1
            ps[c,k,g,r] += y[i]
    choice = np.full(C, -1, np.int64)
    bn = np.zeros((C,3,B), np.int64)
    bs = np.zeros((C,3,B))
    edge = np.zeros(C)
    ucost = np.log(max(1,len(unary)))
    pcost = np.log(max(1,len(pairs)))
    for c in range(C):
        N = pooled[c].sum()
        if N < 3:
            continue
        base = gamma[pooled[c]].sum() - norm[N]
        threshold = 0.
        for u in range(len(unary)):
            on = uon[c,u]
            n = on.sum()
            if n and n < N:
                score = gamma[on].sum() + gamma[pooled[c]-on].sum() - norm[n] - norm[N-n] - base - ucost
                threshold = max(threshold, score)
        best = threshold
        for p in range(len(pairs)):
            score = -base - pcost
            for g in range(3):
                n = pn[c,p,g].sum()
                if not n:
                    score = -np.inf
                    break
                score += gamma[pn[c,p,g]].sum() - norm[n]
            if score > best:
                best = score
                choice[c] = p
        if choice[c] >= 0:
            bn[c] = pn[c,choice[c]]
            bs[c] = ps[c,choice[c]]
            edge[c] = best - threshold
    return choice, bn, bs, edge


class JointEvidence:
    """Pick a signed family before unary admission without changing sensory inference."""
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.joint_key=np.full((self.C,4),-1,dtype=int)
        self.joint_count=np.zeros((self.C,3,self.B))
        self.joint_prob=np.zeros_like(self.joint_count)
        self.joint_amount=np.zeros_like(self.joint_count)
        self.joint_edge=np.zeros(self.C)
        a=1./self.B
        self._joint_gamma=np.array([lgamma(n+a)-lgamma(a) for n in range(1+self.replay.maxlen)])
        self._joint_norm=np.array([lgamma(n+1.) for n in range(1+self.replay.maxlen)])

    def joint_basis(self):
        keep=np.ones(self.J,bool);keep[self.slots[self.slots>=0]]=False
        unary=np.flatnonzero(np.tile(keep,self.L))
        pairs=[];keys=[];baseline=(self.context_x+.5)/(1.+self.context_t)
        for d,j in enumerate(self.anchors):
            if j<0 or not keep[j]:continue
            rate=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            valid=keep&(rate>baseline)&(self.assoc[d]>np.log1p(self.J));valid[j]=False
            for m in np.flatnonzero(valid):
                for l in range(self.L):
                    pairs.append((l*self.J+j,l*self.J+m));keys.append((j,m,l,d))
        return unary, pairs, keys

    def refresh_joint(self):
        self.joint_key.fill(-1);self.joint_edge.fill(0.)
        if not (self.slots>=0).any() or not self.replay:return
        unary,pairs,keys=self.joint_basis()
        if not pairs:return
        _,_,y,by,feat,phase=zip(*self.replay)
        mark,n,amount,margin=_select(np.asarray(feat,dtype=bool),np.asarray(phase,dtype=bool),
                np.asarray(y),self.code(np.asarray(by)),unary,np.asarray(pairs),self.C,self.B,
                self._joint_gamma,self._joint_norm)
        active=mark>=0
        if not active.any():return
        self.joint_key[active]=np.asarray(keys)[mark[active]]
        alpha=n[active]+1./self.B
        self.joint_count[active]=alpha
        self.joint_prob[active]=alpha/alpha.sum(-1,keepdims=True)
        self.joint_amount[active]=(amount[active]+self.center/self.B)/alpha
        self.joint_edge=margin

    def learn_joint(self,b,hist,a,y,bn):
        super().learn_joint(b,hist,a,y,bn)
        self.refresh_joint()

    def rule(self):
        idx,w,p,atoms,p0,base,novel=super().rule()
        rows=np.flatnonzero(self.joint_edge>0.)
        if len(rows):
            idx=idx.copy();w=w.copy();p=p.copy();novel=novel.copy()
            j,_,l,_=self.joint_key[rows].T
            mark=l*self.J+j
            idx[rows,0]=mark;w[rows,0]=True;novel[rows]=0.
            p[rows,:,0]=(self.on[rows,:,mark]+.5)/(1.+self.type_count[rows])
        return idx,w,p,atoms,p0,base,novel

    def signed_rules(self,rule):
        chosen=self.joint_edge>0.
        old=super().signed_rules(rule)
        if not chosen.any():return old
        out=[]
        for j,m,l,s,g,d in old:
            s=s&~chosen
            if s.any():out.append((j,m,l,s,g,d))
        for c in np.flatnonzero(chosen):
            j,m,l,d=self.joint_key[c]
            support=np.arange(self.C)==c
            out.append((int(j),int(m),int(l),support,self.joint_edge,int(d)))
        return out

    def context_moment(self,splits,x,phase,mean,prob,atoms):
        chosen=self.joint_edge>0.
        if not chosen.any():return super().context_moment(splits,x,phase,mean,prob,atoms)
        old=[z for z in splits if not np.any(z[3]&chosen)]
        mu,p,v,n,g=super().context_moment(old,x,phase,mean,prob,atoms)
        if not old:p=p.copy();v=v.copy();mu=mu.copy()
        for c in np.flatnonzero(chosen):
            j,m,l,d=self.joint_key[c]
            k=np.where(phase[:,l*self.J+j],x[:,l*self.J+m].astype(int),2)
            p[:,c]=self.joint_prob[c,k];v[:,c]=self.joint_amount[c,k]
            mu[:,c]=(p[:,c]*v[:,c]).sum(1);n[:,c]=self.joint_count[c,k].sum(1)
            g[:,c]=1+2*len(self.anchors)*self.L+3*(d*self.L+l)+k
        return mu,p,v,n,g
