"""v20 (slice 3b-2 candidate, AR-4): v12's controller reading ONE predictor, `models/factor.py`.

No mixture (decision 13:07, v19 cancelled): every non-value channel of the depth-0 prediction
the planner sees comes from the 3b-1 predictor (`factor-L4-P-V`: place view x regime HCRP,
per-channel structure average with lagged slots, additive and process members). The value
channel and the reward / continuation model stay v12's (the controller is reused as an
interface; whether its predictive tables are needed at all is what 3b-2 measures against v12).
"""
from __future__ import annotations

import numpy as np

from agents.v12 import PosteriorGraph12
from models.factor import FactorModel


class Agent(PosteriorGraph12):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, **kw):
        super().__init__(n_inputs, n_actions, value_index, seed=seed, **kw)
        self.f = FactorModel(n_inputs, n_actions, value_index, seed, lags=tuple(range(5)),
                             own_lags=(0, 1, 2), k=5, process=True, place="view")

    def _predict_next(self, x, action, depth):
        own = PosteriorGraph12._predict_next(self, x, action, depth)
        if depth != 0:
            return own
        p = self.f.predict(x, action)
        out = [float(u) for u in p]
        out[self.value_index] = own[self.value_index]
        return tuple(out)

    def learn(self, next_obs, **kw):
        x, action = self._pending
        self.f.learn(x, action, next_obs)
        super().learn(next_obs, **kw)


Agent.__name__ = "Agent"
