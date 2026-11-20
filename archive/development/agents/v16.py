"""v16 (slice 3, cost step on top of C0 = v15): bounded context per target after maturation.

v15 mixes v12 with the decomposition witness; v12's part still keys every target's context on
the full set of active channels, so its cost and its fragmentation grow with the product of
the processes (D-5: v12 sec x6.8 for x3 processes). v16 prunes, per target j, the channels in
v12's context keys to

    N(j) = group of j in the witness's partition  U  {value channel}

once the witness has had MATURE steps to find the partition (before: N(j) = all channels, v12
exactly, so fast association in short experiments is untouched). Biological reading: broad
early connectivity pruned to what the slowly learned structure supports (synaptic pruning;
a critical period). Timing windows (delayed cue -> outcome) keep their own evidence gate and
are not pruned. v11's attended set is NOT added: on sticky hidden processes its dependence
test marks almost every channel (s-4x3: 11 of 12 after 2000 steps — Yule's spurious
dependence again, D-10), which would undo the pruning (mechanism check, D-12).
"""
from __future__ import annotations

import agents.v15 as v15

MATURE = 1000
USE_ATTENTION = False


class Agent(v15.Agent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._gmap, self._gver = {}, None

    def _nbhd(self, target):
        d = self.dec
        if d.t < MATURE:
            return None
        ver = (len(d.merges), len(self.attended))
        if ver != self._gver:
            att = (set(self.attended) if USE_ATTENTION else set()) | {self.value_index}
            self._gmap = {}
            for g in d.groups:
                s = frozenset(g.chans) | att
                for c in g.chans:
                    self._gmap[c] = s
            self._gver = ver
        return self._gmap.get(target, frozenset({self.value_index, target}))

    def _restrict(self, act, target):
        N = self._nbhd(target)
        return tuple(act) if N is None else tuple(i for i in act if i in N)

    def _predict(self, x, action, target, depth, act=None):
        full = tuple(self._active(x)) if x is not None else act
        return super()._predict(x, action, target, depth, self._restrict(full, target))

    def _learn_target(self, active, full_key, att_key, action, target, outcome):
        r = self._restrict(full_key, target)
        super()._learn_target(list(r), r, self._att(r), action, target, outcome)
