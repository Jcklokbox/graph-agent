"""D-107: posterior opportunity in one unresolved phase, not twice."""
from __future__ import annotations
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class CredibleEventReward(SparseSignedReward):
    one_window=True

    def provisional(self,rule):
        _,var,_=super().provisional(rule)
        n=self.onset; m=(self.onset_reward+.5)/(1.+n)
        eligible=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        strength=np.tile(self.onset_total/(.5+self.past_on),self.L)
        prevalence=(.5+self.present)/(1.+self.total)
        relevance=strength*(1.-prevalence)*eligible
        base=rule[5][:,None]
        # Mixture between a persistent payoff-state law and one alternative
        # depending on a candidate demarcated perceptual event.
        upper=m+(1.-m)/np.sqrt(1.+n)
        guess=base+relevance*(upper-base)
        return guess*eligible*~rule[1].any(1)[:,None],var


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=CredibleEventReward(f.J,f.A,f.D)
