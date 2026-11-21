"""Reference streaming likelihood on shared support (not used by an agent)."""
from collections import deque
import numpy as np


class WindowEvidence:
    """One state; fixed structure assignments on one common real-outcome tape.

    Every hypothesis partitions all retained rows, including its complement.
    Structure rotation is a caller concern: rebuild a changed basis from actual
    retained rows. No learning from imagined utility or sparse deletion masks.
    """
    def __init__(self, models, capacity, atoms=16, groups=3):
        if min(models,capacity,atoms,groups)<1:raise ValueError('positive dimensions required')
        self.capacity=capacity;self.atoms=atoms;self.groups=groups
        self.history=deque();self.model=np.arange(models)
        self.n=np.zeros((models,groups,atoms),dtype=np.int64)
        self.amount=np.zeros_like(self.n,dtype=float)
        self.group_n=np.zeros((models,groups),dtype=np.int64)
        self.total=np.zeros(atoms,dtype=np.int64)
        self.score=np.zeros(models)

    def _increment(self,k,g):
        a=1./self.atoms;keys=self.model,g
        return (np.log(self.n[self.model,g,k]+a)-np.log(self.group_n[keys]+1.)
                -np.log(self.total[k]+a)+np.log(self.total.sum()+1.))

    def append(self,value,group):
        if not np.isfinite(value) or not 0.<=value<=1.:raise ValueError('bounded value required')
        g=np.asarray(group)
        if (g.shape!=self.model.shape or not np.issubdtype(g.dtype,np.integer)
            or (g<0).any() or (g>=self.groups).any()):
            raise ValueError('one valid group per hypothesis required')
        g=g.copy();k=min(self.atoms-1,int(value*self.atoms))
        if len(self.history)==self.capacity:
            v,o,h=self.history.popleft();keys=self.model,h
            self.n[self.model,h,o]-=1;self.amount[self.model,h,o]-=v;self.group_n[keys]-=1;self.total[o]-=1
            self.score-=self._increment(o,h)
        self.score+=self._increment(k,g)
        keys=self.model,g
        self.n[self.model,g,k]+=1;self.amount[self.model,g,k]+=value;self.group_n[keys]+=1;self.total[k]+=1
        self.history.append((value,k,g))
