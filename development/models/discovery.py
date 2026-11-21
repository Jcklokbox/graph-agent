"""Bounded candidate exposure, independent of channel labels in distribution.

A random cyclic order covers every source, with O(n) construction and no all-
pairs table. Each row has a stable local bank and one replaceable unpinned
entry per traversal. Scores are meaningful only over that entry's exposure:
callers clear its local statistics on replacement. A fixed bank does not
provide fixed-latency detection of arbitrarily distant or rare interactions.
"""
from __future__ import annotations

import numpy as np


class NeighborBank:
    def __init__(self, n, capacity, seed, include_self=True):
        self.n = n
        self.width = min(capacity, n if include_self else max(0, n - 1))
        self.include_self = include_self
        self.order = np.random.default_rng(seed).permutation(n)
        self.rank = np.empty(n, int)
        self.rank[self.order] = np.arange(n)
        self.cursor = self.width if include_self else self.width + 1
        self.replace = 0
        if self.complete:
            # Label order is only a tie break when all hypotheses have equal score.
            base = np.broadcast_to(np.arange(n), (n, n))
            self.source = (base.copy() if include_self else
                           base[base != np.arange(n)[:, None]].reshape(n, self.width).copy())
        else:
            start = 0 if include_self else 1
            self.source = self.order[(self.rank[:, None] + np.arange(start, start + self.width)) % n]

    @property
    def complete(self):
        return self.width >= (self.n if self.include_self else self.n - 1)

    def rotate(self, pinned=None):
        """Replace one explored source per row, without touching installed parents."""
        if self.complete or not self.width:
            return np.empty(0, int), np.empty(0, int)
        nxt = self.order[(self.rank + self.cursor) % self.n]
        self.cursor = (self.cursor + 1) % self.n
        col = np.full(self.n, -1, int)
        for d in range(self.width):
            b = (self.replace + d) % self.width
            eligible = col < 0
            if pinned is not None:
                eligible &= ~np.any(pinned == self.source[:, b, None], axis=1)
            if not self.include_self:
                eligible &= nxt != np.arange(self.n)
            col[eligible] = b
        rr = np.flatnonzero((col >= 0) & ~np.any(self.source == nxt[:, None], axis=1))
        cc = col[rr]
        self.source[rr, cc] = nxt[rr]
        self.replace = (self.replace + 1) % self.width
        return rr, cc
