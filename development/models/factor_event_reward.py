"""Controlled event reward with sparse signed refinements and decision information."""
import numpy as np

from models.factor_context_reward import SignedContextualOpportunities
from models.factor_information import SoftEventInformation
from models.factor_regime_reward import RecurrentRegimes
from models.factor_reward import ProvisionalEventReward


class DwellEventReward(ProvisionalEventReward):
    """Require witnessed action-sensitive dwell and project reward parents."""

    factored_projection = True
    elapsed_only = True
    controlled_dwell = True


class ModeEventReward(RecurrentRegimes, DwellEventReward):
    motion_optimism = True


class SparseSignedReward(SignedContextualOpportunities, ModeEventReward):
    P = 1
    sparse_outcomes = True


class SignedSelection(SoftEventInformation, SparseSignedReward):
    """Select a supported moderator within an admitted local payoff parent."""

    context_requires_parent = True
    scoped_pair_only = True
    restrict_info_parent = True
    intrinsic_opportunity = True

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
