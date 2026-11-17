"""Small task variants for theory-level behavioral experiments."""
from __future__ import annotations

import numpy as np


class CancellationGateEnv:
    """A cue predicts food unless a red cancellation lamp is present."""

    def __init__(self, seed: int = 0, period: int = 5) -> None:
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 1
        self.wellbeing = 0.5
        self.precursor: int | None = None
        self.trap = False
        self.food: int | None = None
        self.next_cue = 3
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.0
        if self.precursor == 0:
            obs[3] = 1.0
        if self.food == 0:
            obs[5] = 1.0
        obs[7] = float(self.trap)
        obs[11] = self.wellbeing
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, bool]]:
        if action not in range(3):
            raise ValueError(action)
        cue, cancellation = self.precursor, self.trap
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1
        self.food = 0 if cue == 0 and not cancellation else None
        if cue == 0 and not cancellation and self.position == 0:
            self.wellbeing = 1.0
        elif self.position == 1:
            self.wellbeing = 0.5
        else:
            self.wellbeing = 0.0
        self.precursor, self.trap = None, False
        if self.t >= self.next_cue:
            self.precursor = 0
            self.trap = bool(self.rng.random() < 0.5)
            self.next_cue += self.period
        return self.observation(), self.wellbeing, False, {
            "received_shock": self.wellbeing == 0.0,
            "got_food": self.wellbeing == 1.0,
        }


class VariableDelayEnv:
    """A cue is followed by a food event after a randomly chosen 2–4 steps."""

    def __init__(self, seed: int = 0, period: int = 8) -> None:
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.next_cue = 2
        self.food_due: int | None = None
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=float)
        obs[1] = 1.0
        if self.t == self.next_cue - self.period:
            obs[3] = 1.0
        if self.food_due == self.t:
            obs[5] = 1.0
        obs[11] = 0.5
        return obs

    def step(self, action: int = 1) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        del action
        if self.t == self.next_cue - self.period:
            self.food_due = self.t + int(self.rng.integers(2, 5))
        self.t += 1
        if self.t >= self.next_cue:
            self.next_cue += self.period
        return self.observation(), 0.5, False, {}


class DistantGateEnv:
    """A delayed one-step food window after a cue on the far side."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 2
        self.next_cue = 5
        self.food_time: int | None = None
        self.last_cue = -100
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.0
        if self.t == self.last_cue:
            obs[3] = 1.0
        if self.food_time == self.t and self.position == 0:
            obs[5] = 1.0
        obs[11] = 0.50 if self.position == 2 else 0.45 if self.position == 1 else (
            1.0 if self.food_time == self.t else 0.0
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, bool]]:
        if action not in range(3):
            raise ValueError(action)
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1
        reward = 0.50 if self.position == 2 else 0.45 if self.position == 1 else 0.0
        if self.position == 0 and self.food_time == self.t:
            reward = 1.0
        if self.t >= self.next_cue and self.position == 2:
            self.last_cue = self.t
            self.food_time = self.t + 2
            self.next_cue = self.t + int(self.rng.integers(5, 7))
        obs = self.observation()
        obs[11] = reward
        return obs, reward, False, {
            "got_food": reward == 1.0,
            "received_shock": reward == 0.0 and self.position == 0,
        }
