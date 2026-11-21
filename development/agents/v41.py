"""D-98: multiplicity-aware bounded-real confidence for onset opportunities."""
from __future__ import annotations
import numpy as np
from agents.v36 import Agent as Controller, ModeEventReward


class ConfidenceEventReward(ModeEventReward):
    def provisional(self,rule):
        n=self.onset;m=(self.onset_reward+.5)/(1.+n)
        var=np.maximum(0.,(self.onset_square+1./3)/(1.+n)-m*m)
        eligible=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        transience=np.tile(self.onset_total/(.5+self.past_on),self.L)
        prevalence=(.5+self.present)/(1.+self.total)
        likely=transience*(1.-prevalence)*eligible
        # One-sided Hoeffding confidence radius with a family-size charge;
        # pseudo-count provides feasibility for a truly untested condition.
        upper=np.minimum(1.,m+np.sqrt(np.log1p(n.shape[-1])/(2.*(1.+n))))
        base=rule[5][:,None]
        support=base+likely*np.maximum(0.,upper-base)
        support*=eligible;support*=~rule[1].any(1)[:,None]
        return support,var


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=ConfidenceEventReward(f.J,f.A,f.D)
