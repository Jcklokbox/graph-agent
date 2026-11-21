"""Finite recurrent control and payoff regimes for the factor controller."""
from __future__ import annotations
import numpy as np


class RecurrentRegimes:
    """Mixin: sequential predictive test, one archived regime, finite work.

    The auxiliary short-run predictor uses discounted sufficient statistics.
    Mode switches are driven only by realized controlled-state transitions.
    Either all mode-dependent sufficient statistics are recalled or none are;
    no sensitivity to an episode index or to sensor/action identities.
    """
    _banks=('trans','trans_total','control_mass','type_count','type_sum','on',
            'evidence_type','onset','onset_reward','onset_square')

    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.recent_trans=np.zeros_like(self.trans)
        self.archive_mode=None
        self.known_mode_roles=np.zeros(J,bool)
        from itertools import combinations
        self.permutations=[]
        if getattr(self,'relabel_actions',False):
            for a,b in combinations(range(A),2):
                perm=np.arange(A);perm[a],perm[b]=perm[b],perm[a]
                self.permutations.append(perm)
        self._log_change=np.zeros(2+len(self.permutations))
        self._mode_steps=0
        self.switches=0

    def _observe_mark(self,b,a,bn):
        revised=super()._observe_mark(b,a,bn)
        if revised:
            self.archive_mode=None
            self._mode_steps=0
            self.recent_trans.fill(0.)
            self._log_change.fill(0.)
        return revised

    def _predict_row(self,row,base):
        return (row+base)/(1.+row.sum())

    def _switch(self,recalled,permutation=None):
        active={k:getattr(self,k).copy() for k in self._banks}
        if recalled:
            for k in self._banks:
                getattr(self,k)[...] = self.archive_mode[k]
        else:
            for k in self._banks:
                getattr(self,k).fill(0.)
        if permutation is not None:
            rows=active['trans'][permutation]
            self.trans[...] = rows/np.sqrt(1.+rows.sum(-1,keepdims=True))
            self.trans_total[...] = self.trans.sum(-1)
        self.archive_mode=active
        self.known_mode_roles[self.slots[self.slots>=0]]=True
        self.replay.clear()
        self._log_change.fill(0.)
        self._mode_steps=0
        self.switches+=1

    def _record(self,b,a,y,bn,x,phase):
        i,c=int(self.code(b)),int(self.code(bn))
        valid=(self.ci & int(self.binary[self.slots<0].sum()))==0
        base=(self.control_mass+.5)*valid;base/=base.sum()
        curr=self._predict_row(self.trans[a,i],base)[c]
        recent=self._predict_row(self.recent_trans[a,i],base)[c]
        known=None if self.archive_mode is None else self._predict_row(self.archive_mode['trans'][a,i],base)[c]
        if not getattr(self,'_replaying',False) and self.trans_total[a,i]>0 and (self.slots>=0).any():
            q=np.array([recent,known if known is not None else curr,
                        *[self._predict_row(self.trans[perm[a],i],base)[c] for perm in self.permutations]])
            self._log_change=np.maximum(0.,self._log_change+np.log(q/curr))
            threshold=np.log1p(self._mode_steps)+np.log1p(valid.sum())
            advantage=self._log_change.copy()
            advantage[2:]-=np.log1p(len(self.permutations))
            winner=int(np.argmax(advantage))
            if advantage[winner]>threshold:
                self._switch(winner==1 and self.archive_mode is not None,
                             self.permutations[winner-2] if winner>=2 else None)
                self.replay.append((b.copy(),a,y,bn.copy(),x.copy(),phase.copy()))
        # Decay only the bounded row being tested, so unchosen interventions
        # preserve their evidence. This is a likelihood witness, not a timer.
        self.recent_trans[a,i]*=.5
        self.recent_trans[a,i,c]+=1.
        self._mode_steps+=1
        super()._record(b,a,y,bn,x,phase)
