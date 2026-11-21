"""D-105: compete for unsupported causal payoff hypotheses of the same cue."""
from __future__ import annotations
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class CompetitiveReward(SparseSignedReward):
    def provisional(self,rule):
        optimistic,var,_=super().provisional(rule)
        base=rule[5][:,None];n=self.onset
        mean=(self.onset_reward+.5)/(1.+n)
        strength=n*np.square(np.maximum(mean-base,0.))
        structure=np.zeros_like(strength)
        c,p=np.nonzero(rule[1]); structure[c,rule[0][c,p]]=1.
        own=(strength*structure).reshape(self.C,self.L,self.J).sum(1)
        other=own.sum(0)-own
        opposing=np.tile(other,(1,self.L))
        # Competition acts only on residual, unearned payoff uncertainty.
        witnessed=np.maximum(0.,self.onset_reward-n*base)
        shrink=(1.+witnessed)/(1.+witnessed+opposing)
        return np.where(optimistic>base,base+(optimistic-base)*shrink,optimistic),var


class Agent(Controller):
    def __init__(self,n_inputs,n_actions,value_index,seed=0,*,draws=5,posterior=False):
        super().__init__(n_inputs,n_actions,value_index,seed=seed,draws=draws,posterior=posterior)
        f=self.factor
        f.reward=CompetitiveReward(f.J,f.A,f.D)
