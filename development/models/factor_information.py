"""Finite one-event decision information, not optimism about realized payoffs."""
from __future__ import annotations
from math import lgamma

import numpy as np


def log_marginal(count, prior):
    """Likelihood of one specified sequence under a symmetric Dirichlet."""
    return sum(lgamma(float(x)+prior)-lgamma(prior) for x in count) - (
        lgamma(float(np.sum(count))+len(count)*prior)-lgamma(len(count)*prior))


def posterior_option(q, reach, whole, part, amounts, centers, log_odds, *, whole_amounts=None):
    """Single future decision, with H0 pooled and H1 onset-specific.

    q includes the current pooled mean of this successor. A sample from this
    successor updates H1, including its posterior weight. No claiming the
    mean shift itself as information. Units are value per repeated occasion.
    """
    if np.ptp(reach) < 1e-12:
        return 0.
    k=len(centers); prior=1./k
    outside=np.maximum(0.,whole-part)
    odds=log_odds+log_marginal(part,prior)+log_marginal(outside,prior)-log_marginal(whole,prior)
    odds=float(np.clip(odds,-30,30)); lam=1./(1.+np.exp(-odds))
    p0=(whole+prior)/(1.+whole.sum()); p1=(part+prior)/(1.+part.sum())
    # H0 only uses the existing q and stays pooled; H1 shifts this single
    # destination's return by the change in immediate reward mean.
    mu0=p0@centers if whole_amounts is None else float((whole_amounts+prior*centers).sum()/(1.+whole.sum()))
    reps=(amounts+prior*centers)/(part+prior)
    arg=lam*p1[:,None]
    mass=(1.-lam)*p0+lam*p1
    mu1=(p1@reps*(1.+part.sum())+reps)/(2.+part.sum())
    joint=mass[:,None]*q[None,:]+arg*(mu1-mu0)[:,None]*reach[None,:]
    # Compare the same fully marginal future opportunity, with and without
    # observing its atom; arbitrary posterior mean changes cancel.
    return max(0.,float(joint.max(axis=1).sum()-joint.sum(axis=0).max()))


class RepeatableInformation:
    """Only the finite current cue bank supplies repeated-test proposals."""
    information_mode='overlay'

    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.info_age=np.arange(1,2*self.replay.maxlen+1,dtype=float)

    def _information_features(self, x, phase):
        baseline=(self.context_x+.5)/(1.+self.context_t)
        allowed=np.ones(self.J,bool)
        allowed[self.slots[self.slots>=0]]=False
        for d,j in enumerate(self.anchors):
            if j<0 or not phase[j] or self.anchor_n[d]<2:
                continue
            p=(self.pair_on[d]+.5)/(1.+self.anchor_n[d])
            good=allowed&(p>baseline)&(p<1.-.5/(1.+self.anchor_n[d]));good[j]=False
            bf=np.where(good,self.assoc[d],-np.inf);m=int(np.argmax(bf))
            freq=(self.onset_total[j]+.5)/(1.+self.total)
            if bf[m]>np.log1p(self.J):
                selected=int(x[m]); freq*=p[m] if selected else 1.-p[m]
                yield self.ctx_atoms[:,d,0,m,selected], self.ctx_amount[:,d,0,m,selected], freq, True
            else:
                yield self.ctx_atoms[:,d,0,j,1], self.ctx_amount[:,d,0,j,1], freq, False

    def discovery_value(self,q,operator,rule,x,phase):
        # Beta(1, 1) one-step regime hazard: E[(1-h)^k | s unchanged steps]
        # = (s+1)/(s+1+k). Truncate to the bounded local replay horizon.
        scale=1.+self._mode_steps
        horizon=float(np.sum(scale/(scale+self.info_age)))
        out=np.zeros(self.A)
        for counts,amounts,freq,paired in self._information_features(x,phase):
            for c in np.flatnonzero(self.control_mass>0.):
                if rule[1][c].any() and not paired:
                    continue
                reach=operator[:,c]
                if np.ptp(reach)<1e-12:
                    continue
                opt=posterior_option(q,reach,self.type_count[c],counts[c],amounts[c],
                                     self.center,-np.log(len(self.anchors)),whole_amounts=self.type_sum[c])
                # Reaching the same c is what gathers the evidence. Information
                # available under all actions cannot by itself drive taking risk.
                out=np.maximum(out,(reach-reach.min())*(horizon*freq*opt))
        return out
