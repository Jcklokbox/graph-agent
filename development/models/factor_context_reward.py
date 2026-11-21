"""Bounded configural extensions to the controlled event reward law."""
from __future__ import annotations
import numpy as np


class ContextualOpportunities:
    """Attach a small rotating set of onset×moderator opportunity factors.

    Concurrent moderator evidence comes from the full observation stream.
    Outcome sufficient statistics are conditional on a realized controlled
    successor. An adverse outcome in one branch does not extinguish its sibling.
    """
    max_anchors=4

    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        K=min(J,self.max_anchors)
        self.anchors=np.full(K,-1,int)
        self.ctx_n=np.zeros((self.C,K,self.L,J,2));self.ctx_s=np.zeros_like(self.ctx_n)
        self.assoc=np.zeros((K,J));self.pair_on=np.zeros((K,J));self.anchor_n=np.zeros(K)
        self.context_x=np.zeros(J);self.context_t=0
        self._j=np.arange(J)

    def _update_context_structure(self,b):
        intensity=self.onset_total/(.5+self.past_on)
        strength=intensity*(1.-(self.past_on+.5)/(1.+self.total))*np.sqrt(self.onset_total)
        strength[self.slots[self.slots>=0]]=0.
        desired=np.argpartition(-strength,len(self.anchors)-1)[:len(self.anchors)]
        desired=desired[strength[desired]>0.]
        abandoned=np.flatnonzero(~np.isin(self.anchors,desired))
        maintained=set(self.anchors)
        replacement=[v for v in desired if v not in maintained]
        revised=False
        for d in abandoned:
            j=replacement.pop(0) if replacement else -1
            if self.anchors[d]==j:continue
            self.anchors[d]=j;self.ctx_n[:,d]=0.;self.ctx_s[:,d]=0.
            self.assoc[d]=0.;self.pair_on[d]=0.;self.anchor_n[d]=0.;revised=True
        base=(self.context_x+.5)/(1.+self.context_t)
        for d,j in enumerate(self.anchors):
            if j<0 or not b[j]:continue
            alt=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            self.assoc[d]+=np.log(np.where(b,alt,1.-alt)/np.where(b,base,1.-base))
            self.pair_on[d]+=b;self.anchor_n[d]+=1.
        self.context_x+=b;self.context_t+=1
        return revised

    def _pair_record(self,bn,y,x,phase):
        c=int(self.code(bn));ix=x.reshape(self.L,self.J);on=phase.reshape(self.L,self.J)
        for d,j in enumerate(self.anchors):
            if j<0:continue
            for l in np.flatnonzero(on[:,j]):
                z=ix[l].astype(int)
                self.ctx_n[c,d,l,self._j,z]+=1.;self.ctx_s[c,d,l,self._j,z]+=y

    def learn_joint(self,b,hist,a,y,bn):
        # Scope changes or switching to another regimen can invalidate old
        # conditionals. Higher-level sensory recurrence is still learned.
        old=self.slots.copy();modes=self.switches
        super().learn_joint(b,hist,a,y,bn)
        change=self._update_context_structure(b)
        if change or modes!=self.switches or (old!=self.slots).any():
            self._reset_context_counts()
            for bx,ax,rx,by,x,phase in self.replay:
                self._pair_record(by,rx,x,phase)
        elif self.replay:
            _,_,rx,by,x,phase=self.replay[-1]
            self._pair_record(by,rx,x,phase)

    def _reset_context_counts(self):
        self.ctx_n.fill(0.);self.ctx_s.fill(0.)

    def pair_context(self,rule,x,phase,shift=False):
        """Model-aware optimism for already observed onset configurations."""
        out=np.zeros(self.C)
        f=np.asarray(x).reshape(self.L,self.J);on=np.asarray(phase).reshape(self.L,self.J)
        allowed=np.ones(self.J,bool);allowed[self.slots[self.slots>=0]]=False
        base_rate=(self.context_x+.5)/(1.+self.context_t)
        for d,j in enumerate(self.anchors):
            if j<0:continue
            am=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            # Seek co-handler moderators, not ubiquitous absences; do not
            # duplicate causally controlled coordinates in the sensory prior.
            good=allowed&(am>base_rate)&(am<1.-.5/(1.+self.anchor_n[d]));good[j]=False
            score=np.where(good,self.assoc[d],-np.inf)
            m=int(np.argmax(score))
            if score[m]<=np.log1p(self.J):continue
            rarity=(1.-base_rate[j])*self.onset_total[j]/(.5+self.past_on[j])
            for l in np.flatnonzero(on[:,j]):
                if shift and l==0:continue
                c=int(f[l,m]);n=self.ctx_n[:,d,l,m,c]
                if getattr(self,'context_refines_known',False):
                    total=self.ctx_n[:,d,l,m].sum(1)
                    prior=(self.ctx_s[:,d,l,m].sum(1)+.5)/(1.+total)
                    u=(self.ctx_s[:,d,l,m,c]+prior)/(1.+n)
                    if c==0:
                        weight=rarity*(am[m]-base_rate[m])/max(am[m],1e-12)
                    else:
                        weight=rarity*(am[m]-base_rate[m])/max(1.-base_rate[m],1e-12)
                    u+=weight*(1.-u)/np.sqrt(1.+n)
                    # An admitted stationary causal law still leaves its
                    # two correlated configurations uncertain.
                    same=((rule[0]==l*self.J+j)&rule[1]).any(1)
                    qualified=same if getattr(self,'context_requires_parent',False) else same|~rule[1].any(1)
                    out=np.maximum(out,u*qualified)
                else:
                    u=(self.ctx_s[:,d,l,m,c]+.5)/(1.+n)
                    u+=rarity*(1.-u)/np.sqrt(1.+n)
                    out=np.maximum(out,u)
        return out if getattr(self,'context_refines_known',False) else out*~rule[1].any(1)


class SuccessorOpportunities:
    """Use forecastable, off-state successor pulses to propose novel reward tests."""
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.succ_n=np.zeros((self.C,J));self.succ_s=np.zeros_like(self.succ_n)
        self.succ_on=np.zeros(J);self.succ_repeat=np.zeros(J);self.succ_total=0

    def _bridge_record(self,y,bn):
        c=int(self.code(bn));self.succ_n[c]+=bn;self.succ_s[c]+=y*bn

    def learn_joint(self,b,hist,a,y,bn):
        old=self.slots.copy();mode=self.switches
        super().learn_joint(b,hist,a,y,bn)
        self.succ_on+=bn;self.succ_repeat+=b*bn;self.succ_total+=1
        if mode!=self.switches or (old!=self.slots).any():
            self.succ_n.fill(0.);self.succ_s.fill(0.)
            for _,_,ry,by,_,_ in self.replay:self._bridge_record(ry,by)
        else:
            self._bridge_record(y,bn)

    def bridge_table(self,rule):
        known=rule[1].any(1)
        allowed=np.ones(self.J,bool);allowed[self.slots[self.slots>=0]]=False
        outside=self.succ_n.sum(0)-self.succ_n
        base=(self.succ_on+.5)/(1.+self.succ_total)
        transience=1.-self.succ_repeat/(.5+self.succ_on)
        n=self.succ_n;m=(self.succ_s+.5)/(1.+n)
        upper=m+transience*(1.-m)/np.sqrt(1.+n)
        possible=(outside>0)&allowed[None,:]&~known[:,None]
        # Fleet opportunity only where departure from this belief is possible.
        return np.where(possible,np.maximum(0.,upper-rule[5][:,None]),0.),base


class SignedContextualOpportunities(ContextualOpportunities):
    """Outcome probability for a supported signed split inside an onset scope."""
    context_refines_known=True

    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.ctx_atoms=np.zeros((*self.ctx_n.shape,self.B))
        self.ctx_amount=np.zeros_like(self.ctx_atoms)

    def _update_context_structure(self,b):
        before=self.anchors.copy()
        revise=super()._update_context_structure(b)
        if revise:
            changed=before!=self.anchors
            self.ctx_atoms[:,changed]=0.;self.ctx_amount[:,changed]=0.
        return revise

    def _pair_record(self,bn,y,x,phase):
        super()._pair_record(bn,y,x,phase)
        c=int(self.code(bn));r=min(self.B-1,int(y*self.B));ix=x.reshape(self.L,self.J)
        on=phase.reshape(self.L,self.J)
        for d,j in enumerate(self.anchors):
            if j<0:continue
            for l in np.flatnonzero(on[:,j]):
                z=ix[l].astype(int)
                self.ctx_atoms[c,d,l,self._j,z,r]+=1.;self.ctx_amount[c,d,l,self._j,z,r]+=y

    def learn_joint(self,b,hist,a,y,bn):
        # Reprojection is one bounded operation on scope change. The callback
        # inside the contextual basis owns its reset, including subtype arrays.
        return super().learn_joint(b,hist,a,y,bn)

    def _reset_context_counts(self):
        super()._reset_context_counts()
        self.ctx_atoms.fill(0.);self.ctx_amount.fill(0.)

    def signed_rules(self,rule):
        """Select signed splits by local explained-energy gain and model cost."""
        result=[];baseline=(self.context_x+.5)/(1.+self.context_t)
        possible=np.ones(self.J,bool);possible[self.slots[self.slots>=0]]=False
        for d,j in enumerate(self.anchors):
            if j<0:continue
            rate=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            is_pair=possible&(rate>baseline)&(rate<1.-.5/(1.+self.anchor_n[d]));is_pair[j]=False
            e=np.where(is_pair,self.assoc[d],-np.inf);m=int(e.argmax())
            if e[m]<=np.log1p(self.J):continue
            for l in range(self.L):
                parent=((rule[0]==l*self.J+j)&rule[1]).any(1)
                if not parent.any():continue
                n=self.ctx_n[:,d,l,m];s=self.ctx_s[:,d,l,m];N=n.sum(1)
                mu=np.divide(s,n,out=np.zeros_like(s),where=n>0.)
                gain=np.prod(n,axis=1)/np.maximum(1.,N)*np.square(mu[:,0]-mu[:,1])
                support=parent&(n>0).all(1)&(gain>.25*np.log1p(N))
                if support.any():
                    result.append((j,m,l,support,gain,d))
        return result

    def context_moment(self,splits,x,phase,mean,prob,atoms):
        """Return learned real local laws, with identity for all non-split cells."""
        if not splits:
            return mean,prob,np.broadcast_to(atoms,prob.shape),np.zeros_like(mean),np.zeros_like(mean,dtype=int)
        x=np.asarray(x);phase=np.asarray(phase)
        mass=np.array(prob,copy=True);values=np.broadcast_to(atoms,prob.shape).copy()
        selected=np.zeros_like(mean,dtype=int);count=np.zeros_like(mean);score=np.zeros_like(mean)
        for j,m,l,support,gain,d in splits:
            on=phase[:,l*self.J+j];cfg=x[:,l*self.J+m].astype(int)
            use=on[:,None]&support[None,:]&(gain>score)
            if not use.any():continue
            bn=self.ctx_atoms[:,d,l,m][:,cfg,:].transpose(1,0,2)
            bs=self.ctx_amount[:,d,l,m][:,cfg,:].transpose(1,0,2)
            alpha=bn+1./self.B
            p=alpha/alpha.sum(-1,keepdims=True)
            a=(bs+self.center/self.B)/alpha
            mass[use]=p[use];values[use]=a[use]; score[use]=np.broadcast_to(gain,use.shape)[use]
            idx=np.broadcast_to(1+2*(d*self.L+l)+cfg[:,None],use.shape)
            selected[use]=idx[use];count[use]=alpha.sum(-1)[use]
        return (mass*values).sum(-1),mass,values,count,selected
