"""AR6-08: reward evidence selects a moderator within an earned parent."""
import numpy as np

from agents.v54 import Agent as Controller, ScopedEventInformation


class SignedSelection(ScopedEventInformation):
    def signed_rules(self,rule):
        result=[]
        allowed=np.ones(self.J,bool);allowed[self.slots[self.slots>=0]]=False
        base=(self.context_x+.5)/(1.+self.context_t)
        for d,j in enumerate(self.anchors):
            if j<0:continue
            rate=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            eligible=allowed&(rate>base)&(self.assoc[d]>np.log1p(self.J));eligible[j]=False
            if not eligible.any():continue
            for l in range(self.L):
                parent=((rule[0]==l*self.J+j)&rule[1]).any(1)
                if not parent.any():continue
                n=self.ctx_n[:,d,l];s=self.ctx_s[:,d,l];count=n.sum(-1)
                mu=np.divide(s,n,out=np.zeros_like(s),where=n>0.)
                gain=n.prod(-1)/np.maximum(1.,count)*np.square(mu[...,0]-mu[...,1])
                excess=gain-.25*np.log1p(count)-.25*np.log1p(eligible.sum()-1)
                good=parent[:,None]&eligible[None,:]&(n>0).all(-1)&(excess>0.)
                best=np.argmax(np.where(good,excess,-np.inf),axis=1)
                for m in np.unique(best[good.any(1)]):
                    support=good[:,m]&(best==m)
                    result.append((j,int(m),l,support,gain[:,m],d))
        return result


class Agent(Controller):
    reward_class=SignedSelection
