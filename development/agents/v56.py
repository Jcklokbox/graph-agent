"""CONTROL probe: room bound on the unmodified onset gate. D-114 isolation: room-bounded optimism on the original onset gate only."""
from __future__ import annotations
from agents.v44 import Agent as Controller, SparseSignedReward


class RoomOnsetProvision(SparseSignedReward):
    pass


class Agent(Controller):
    def __init__(self, n_inputs, n_actions, value_index, seed=0, *, draws=5, posterior=False):
        super().__init__(n_inputs, n_actions, value_index, seed, draws=draws, posterior=posterior)
        f = self.factor
        f.reward = RoomOnsetProvision(f.J, f.A, f.D)
