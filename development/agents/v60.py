"""AR6-N-1 (D-114): provisional payoff = room left in the (state, configuration) cell.

Measured defect (AR-6 alt-N, exposed probes on `canc`). The optimistic payoff of
the unified object is

    bound(c, f) = m(c, f) + (1 - m(c, f)) * plausibility(f) / sqrt(1 + n_onset(c))
    incumbent(c) = max(r(c), max_f bound(c, f) * onset(f))

so the headroom claimed at an outcome control state `c` is driven by how *rarely
the cue fires there*, and the payoff mean `m` it inflates is the state's onset
payoff. Two consequences, both observed: the maximum-reward decision (left at a
cue with the cancellation lamp off) keeps its learned mid payoff while the state
that persistence reaches carries the onset, so the optimism rewards staying; and
an onset that the reward law has already explained still claims
`(1-m)*plausibility` of headroom, because `m` belongs to the state and not to the
feature it multiplies.

One coherent change. Keep the marginal statistic and its gate exactly, and let
the provisional payoff be the room that is actually left in the cell the
hypothesis is about — the state together with the observed configuration of the
payoff's own screened parents. The room is the Jeffreys width of that cell's
realized payoff,

    room(c, pattern) = sqrt(m (1 - m) / (2 + n)),

so it is positive only while the cell's payoff is mid and poorly determined, it
shrinks with the cell's own exposure (no exploration rate), and it is zero where
the payoff is already pinned. A nuisance channel is not a payoff parent, so it
does not multiply the cells; a parent the law already screened does, and its
untried value is exactly the configuration that was never tested.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from agents.v44 import Agent as Controller, SparseSignedReward


class CellRoomProvision(SparseSignedReward):
    cell_cap = 256

    def __init__(self, J, A, lag):
        super().__init__(J, A, lag)
        self.cells = OrderedDict()
        self._sig, self._idx = None, np.empty(0, int)
        self._b = None

    def _pattern(self, b):
        active = self.slots[self.slots >= 0]
        sig = tuple(int(s) for s in active)
        if sig != self._sig:
            self._idx = np.unique(active % self.J).astype(int) if len(active) else np.empty(0, int)
            self._sig = sig
        if not len(self._idx):
            return ()
        return tuple(np.flatnonzero(b[self._idx] > .5).tolist())

    def learn_joint(self, b, hist, a, y, bn):
        super().learn_joint(b, hist, a, y, bn)
        key = (int(self.code(bn)), self._pattern(b))
        rec = self.cells.get(key)
        if rec is None:
            if len(self.cells) >= self.cell_cap:
                self.cells.popitem(last=False)
            rec = [0., 0.]
            self.cells[key] = rec
        else:
            self.cells.move_to_end(key)
        rec[0] += 1.
        rec[1] += y

    def cell_room(self, b):
        pattern = self._pattern(b)
        room = np.zeros(self.C)
        for c in range(self.C):
            rec = self.cells.get((c, pattern))
            if rec is None:
                room[c] = .5               # untried cell: maximal Jeffreys width
            else:
                n, s = rec
                m = (s + .5) / (1. + n)
                room[c] = np.sqrt(max(0., m * (1. - m) / (2. + n)))
        return room


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = CellRoomProvision(f.J, f.A, f.D)
