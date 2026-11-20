"""v12d: v12 with an event-driven attention test (slice 3b, first bounded piece; D-15).

v11's attention (Granger-style split Bayes factors, stratified by the target's own state)
re-evaluates every counted pair (i, j) at every lag on every step: O(max_age n^2) Bayes factors
per step — the largest single term of v12's inference cost (38 % on s-4x3, inference-only
profile, 06:18 UTC). v12d evaluates a pair only on the steps where its evidence changed, i.e.
when the cue i was active at that lag (Hebbian locality: only co-active units update their
connection). The counts are unchanged; a dependence can only be detected later, never
differently. Cost O(max_age |active| n): linear in n when activity is sparse (as in sensory
coding), still quadratic for dense channels.
"""
from __future__ import annotations

from agents.v11 import split_log_bf
from agents.v12 import PosteriorGraph12


class EventDrivenAttention:
    def _update_attention(self, active, y, action) -> None:
        n = self.n_inputs
        cues = [i for i in active if i < n]
        prev = set(cues)
        on = [y[j] > .5 for j in range(n)]
        hist = self._dep_hist
        hist.insert(0, cues)
        del hist[self.max_age:]
        for k, old in enumerate(hist, 1):
            self._dep_count((action, k), old, prev, on)
        if len(self.attended) == n:
            return
        for k, old in enumerate(hist, 1):
            self._dep_test_cues((action, k), old)

    def _dep_test_cues(self, a, cues) -> None:
        A = self.attended
        N, Y, C = self._dep[a]
        for i in cues:
            for j in range(self.n_inputs):
                if j == i or (i in A and j in A):
                    continue
                cell = C.get((i, j))
                if cell is None:
                    continue
                lbf = 0.
                for sj in (0, 1):
                    Ns, y1 = N[sj][j], Y[sj][j]
                    n1, c = cell[2 * sj], cell[2 * sj + 1]
                    if n1 < 1. or n1 >= Ns or y1 < 1. or y1 >= Ns:
                        continue
                    lbf += split_log_bf(((c, n1), (y1 - c, Ns - n1)))
                if lbf > self._dep_threshold:
                    A.add(i)
                    A.add(j)


class Agent(EventDrivenAttention, PosteriorGraph12):
    pass
