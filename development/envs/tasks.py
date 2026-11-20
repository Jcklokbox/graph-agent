"""Environments, task variations and reference policies of the benchmark.

Three base tasks (12 channels: positions 0-2, cues 3-7, value channel 11):

* aut  (BlindGateEnv)        a precursor on one side announces food there on the
                               next step; standing on a side without food is a shock.
* canc (CancellationGateEnv)  a cue announces food on the left unless a
                               cancellation lamp is lit at the same time.
* dist (DistantGateEnv)       a cue, visible only from the right, announces a
                               one-step food window on the far left two steps later;
                               the middle costs a little, the right is safe.

A *variant* is a base task with wrappers: extra noise channels (iid or sticky),
longer delays, a monotone re-mapping of the reward, serial reversals of the
action/sensor map, a duplicated action. Every variant is one entry of VARIANTS;
SPLITS lists the DEV variants (model selection) and the held-out sets (each run
once, when it was fresh).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- base tasks
@dataclass(frozen=True)
class StepInfo:
    step: int
    position: int
    action: int
    precursor_side: int | None
    food_side: int | None
    got_food: bool
    received_shock: bool


class BlindGateEnv:
    """Precursor on side s at t -> food on side s at t+1 (seen only there)."""

    def __init__(self, seed: int = 0, event_interval: tuple[int, int] = (4, 5),
                 noise_probability: float = .5) -> None:
        self.seed, self.event_interval = int(seed), event_interval
        self.noise_probability = noise_probability
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_count, self.position, self.wellbeing = 0, 1, .5
        self._precursor_side = self._food_side = None
        self._next_event_step = int(self.rng.integers(self.event_interval[0],
                                                      self.event_interval[1] + 1))
        return self._observation()

    def _observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float32)
        obs[self.position] = 1
        if self._precursor_side is not None:
            obs[3 + self._precursor_side] = 1
        if self._food_side is not None:
            obs[5 + self._food_side] = 1
        obs[7] = float(self.rng.random() < self.noise_probability)
        obs[11] = self.wellbeing
        return obs

    def _move(self, action: int) -> int:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.step_count += 1
        return action

    def _settle(self, chosen: int):
        food = self._food_side
        got = (self.position == 0 and food == 0) or (self.position == 2 and food == 1)
        shock = self.position != 1 and not got
        self.wellbeing = 1. if got else (0. if shock else .5)
        info = StepInfo(self.step_count, self.position, chosen, self._precursor_side, food,
                        bool(got), bool(shock))
        return self._observation(), self.wellbeing, False, info

    def step(self, action: int):
        chosen = self._move(int(action))
        self._food_side, self._precursor_side = self._precursor_side, None
        if self.step_count == self._next_event_step:
            self._precursor_side = int(self.rng.integers(0, 2))
            gap = int(self.rng.integers(*self.event_interval, endpoint=True))
            self._next_event_step = self.step_count + gap
        return self._settle(chosen)


class BlindGateFoodDelay(BlindGateEnv):
    """BlindGate where food appears `delay` steps after its precursor (base: 1)."""

    def __init__(self, seed: int = 0, delay: int = 2) -> None:
        self.delay = delay
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._hist = deque([None] * self.delay, maxlen=self.delay)
        return super().reset(seed)

    def step(self, action: int):
        chosen = self._move(int(action))
        self._hist.append(self._precursor_side)
        self._food_side, self._precursor_side = self._hist[0], None
        if self.step_count >= self._next_event_step:
            self._precursor_side = int(self.rng.integers(0, 2))
            gap = int(self.rng.integers(*self.event_interval, endpoint=True)) + self.delay - 1
            self._next_event_step = self.step_count + gap
        return self._settle(chosen)


class CancellationGateEnv:
    """A cue predicts food on the left unless a cancellation lamp is present."""

    def __init__(self, seed: int = 0, period: int = 5) -> None:
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t, self.position, self.wellbeing = 0, 1, .5
        self.precursor, self.trap, self.food = None, False, None
        self.next_cue = 3
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.
        obs[3] = float(self.precursor == 0)
        obs[5] = float(self.food == 0)
        obs[7] = float(self.trap)
        obs[11] = self.wellbeing
        return obs

    def step(self, action: int):
        if action not in range(3):
            raise ValueError(action)
        delivered = self.precursor == 0 and not self.trap
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1
        self.food = 0 if delivered else None
        self.wellbeing = (1. if delivered and self.position == 0
                          else .5 if self.position == 1 else 0.)
        self.precursor, self.trap = None, False
        if self.t >= self.next_cue:
            self.precursor = 0
            self.trap = bool(self.rng.random() < .5)
            self.next_cue += self.period
        return self.observation(), self.wellbeing, False, {
            "received_shock": self.wellbeing == 0., "got_food": self.wellbeing == 1.}


class DistantGateEnv:
    """A cue seen on the right opens a one-step food window on the far left."""

    def __init__(self, seed: int = 0, delay: int = 2) -> None:
        self.rng = np.random.default_rng(seed)
        self.delay = delay
        self.reset()

    def reset(self) -> np.ndarray:
        self.t, self.position, self.next_cue = 0, 2, 5
        self.food_time, self.last_cue = None, -100
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.
        obs[3] = float(self.t == self.last_cue)
        obs[5] = float(self.food_time == self.t and self.position == 0)
        obs[11] = .50 if self.position == 2 else .45 if self.position == 1 else (
            1. if self.food_time == self.t else 0.)
        return obs

    def step(self, action: int):
        if action not in range(3):
            raise ValueError(action)
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1
        reward = .50 if self.position == 2 else .45 if self.position == 1 else 0.
        if self.position == 0 and self.food_time == self.t:
            reward = 1.
        if self.t >= self.next_cue and self.position == 2:
            self.last_cue, self.food_time = self.t, self.t + self.delay
            self.next_cue = self.t + int(self.rng.integers(5, 7))
        obs = self.observation()
        obs[11] = reward
        return obs, reward, False, {"got_food": reward == 1.,
                                    "received_shock": reward == 0. and self.position == 0}


# ------------------------------------------------------------------ wrappers
class RewardMap:
    """Order-preserving re-mapping of the value channel and the reward."""

    def __init__(self, env, fn):
        self.env, self.fn = env, fn

    def _m(self, o):
        o = np.asarray(o, float).copy()
        o[11] = self.fn(o[11])
        return o

    def reset(self):
        return self._m(self.env.reset())

    def step(self, a):
        o, r, d, i = self.env.step(a)
        return self._m(o), self.fn(r), d, i


class Distractors:
    """k task-irrelevant noise channels appended to the observation: iid
    Bernoulli(p) (sticky = 0) or a two-state Markov chain that keeps its value
    with probability `sticky` (stationary mean p)."""

    def __init__(self, env, k, seed, p=.5, sticky=0.):
        self.env, self.k, self.p, self.sticky = env, k, p, sticky
        self.rng = np.random.default_rng(seed)
        self.state = (self.rng.random(k) < p).astype(float)

    def _m(self, o):
        if self.sticky > 0.:
            keep = self.rng.random(self.k) < self.sticky
            fresh = (self.rng.random(self.k) < self.p).astype(float)
            self.state = np.where(keep, self.state, fresh)
        else:
            self.state = (self.rng.random(self.k) < self.p).astype(float)
        return np.concatenate([np.asarray(o, float), self.state])

    def reset(self):
        return self._m(self.env.reset())

    def step(self, a):
        o, r, d, i = self.env.step(a)
        return self._m(o), r, d, i


class DuplicateAction:
    """A fourth action that is physically identical to 'stay'."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(1 if a == 3 else a)


class Relabel:
    """Permutes the sensor channels 0-7 and the actions of an environment."""

    def __init__(self, env, sensors, actions):
        self.env, self.sensors, self.actions = env, tuple(sensors), tuple(actions)

    def map(self, obs):
        out = np.asarray(obs, dtype=float).copy()
        out[:8] = np.asarray(obs, dtype=float)[list(self.sensors)]
        return out

    def reset(self):
        return self.map(self.env.reset())

    def step(self, action):
        obs, r, d, info = self.env.step(self.actions[action])
        return self.map(obs), r, d, info

    def reverse(self):
        """Mirror left/right: swap the outer actions and the two precursors."""
        self.actions = (self.actions[2], self.actions[1], self.actions[0])
        p = list(self.sensors)
        a, b = p.index(3), p.index(4)
        p[a], p[b] = p[b], p[a]
        self.sensors = tuple(p)


class SerialReversal:
    """BlindGate whose action map and precursor sides flip every `period` steps."""

    def __init__(self, env_seed, period):
        self.env = Relabel(BlindGateEnv(seed=env_seed), (4, 0, 2, 3, 1, 5, 6, 7), (2, 1, 0))
        self.period, self.t = period, 0

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, a):
        if self.t > 0 and self.t % self.period == 0:
            self.env.reverse()
        self.t += 1
        return self.env.step(a)


# ------------------------------------------------------------------- oracles
def blind_gate_oracle(obs) -> int:
    if obs[0] > .5:
        return 2
    if obs[2] > .5:
        return 0
    if obs[3] > .5:
        return 0
    if obs[4] > .5:
        return 2
    return 1


def cancellation_oracle(obs) -> int:
    if obs[0] > .5:
        return 2
    if obs[2] > .5:
        return 0
    if obs[3] > .5 and obs[7] < .5:
        return 0
    return 1


def distant_oracle(delay):
    state = {"plan": []}

    def policy(obs):
        if obs[2] > .5 and obs[3] > .5:
            state["plan"] = [1] * (delay - 2) + [0, 0]
        if state["plan"]:
            return state["plan"].pop(0)
        if obs[0] > .5 or obs[1] > .5:
            return 2
        return 1
    return policy


def food_delay_oracle(delay):
    state = {"plan": []}

    def policy(obs):
        if obs[3] > .5 or obs[4] > .5:
            side = 0 if obs[3] > .5 else 2
            state["plan"] = [1] * (delay - 1) + [side]
        if obs[0] > .5:
            state["plan"] = []
            return 2
        if obs[2] > .5:
            state["plan"] = []
            return 0
        if state["plan"]:
            return state["plan"].pop(0)
        return 1
    return policy


# ------------------------------------------------------------------ variants
@dataclass(frozen=True)
class Variant:
    make: object          # env_seed -> environment
    n_inputs: int
    n_actions: int
    steps: int
    oracle: object        # () -> policy, or None (serial reversal: oracle on the raw world)


def variant(base: str, *, noise: int = 0, p: float = .5, sticky: float = 0., delay: int = 0,
            reward=None, dup: bool = False, serial: int = 0, steps: int = 0) -> Variant:
    """Base task + wrappers. delay: food delay (aut) or cue-to-window delay (dist)."""
    def make(s):
        if serial:
            env = SerialReversal(s, serial)
        elif base == "aut":
            env = BlindGateFoodDelay(seed=s, delay=delay) if delay else BlindGateEnv(seed=s)
        elif base == "canc":
            env = CancellationGateEnv(seed=s)
        else:
            env = DistantGateEnv(seed=s, delay=delay or 2)
        if reward is not None:
            env = RewardMap(env, reward)
        if dup:
            env = DuplicateAction(env)
        if noise:
            env = Distractors(env, noise, s, p=p, sticky=sticky)
        return env

    if serial:
        oracle = None
    elif base == "aut":
        oracle = (lambda: food_delay_oracle(delay)) if delay else (lambda: blind_gate_oracle)
    elif base == "canc":
        oracle = lambda: cancellation_oracle
    else:
        oracle = lambda: distant_oracle(delay or 2)
    n = steps or (5 * serial if serial else 800 if base == "dist" else 600)
    return Variant(make, 12 + noise, 3 + int(dup), n, oracle)


def _square(w):
    return w * w


def _cube(w):
    return w * w * w


def _sqrt(w):
    return w ** .5


def _half(w):
    return .5 * w


def _lin(a, b):
    return lambda w: a + b * w


VARIANTS = {
    # ---- DEV (Rounds 7, 8, 10, 11)
    "aut": variant("aut"),
    "canc": variant("canc"),
    "dist": variant("dist"),
    "aut-half": variant("aut", reward=_half),
    "aut-noise4": variant("aut", noise=4),
    "dist-delay3": variant("dist", delay=3),
    "aut-serial300": variant("aut", serial=300),
    "aut-noise4-sticky": variant("aut", noise=4, sticky=.9),
    "canc-noise3": variant("canc", noise=3),
    "dist-noise2-sticky": variant("dist", noise=2, p=.3, sticky=.8),
    "dist-noise2": variant("dist", noise=2),
    "aut-fooddelay2": variant("aut", delay=2),
    "dist-delay3-noise2": variant("dist", delay=3, noise=2),
    "dist-noise3-p4": variant("dist", noise=3, p=.4),
    "aut-fooddelay3-noise1": variant("aut", delay=3, noise=1),
    # ---- HELD (Round 7)
    "aut-high": variant("aut", reward=_lin(.5, .5)),
    "aut-square": variant("aut", reward=_square),
    "aut-noise8-sticky": variant("aut", noise=8, sticky=.8),
    "canc-noise6": variant("canc", noise=6, p=.3),
    "dist-delay4": variant("dist", delay=4),
    "dist-shift": variant("dist", reward=_lin(.25, .5)),
    "dist-noise4": variant("dist", noise=4),
    "aut-dupaction": variant("aut", dup=True),
    "aut-serial200": variant("aut", serial=200, steps=1400),
    # ---- HELD2 (Round 8)
    "aut-noise6-sticky95": variant("aut", noise=6, p=.3, sticky=.95),
    "canc-noise4-sticky": variant("canc", noise=4, sticky=.8),
    "dist-delay5": variant("dist", delay=5),
    "dist-noise3": variant("dist", noise=3, p=.3),
    "dist-delay4-noise2": variant("dist", delay=4, noise=2),
    "aut-fooddelay3": variant("aut", delay=3),
    "aut-affine": variant("aut", reward=_lin(.2, .3)),
    "aut-sqrt": variant("aut", reward=_sqrt),
    "aut-dupaction-noise3": variant("aut", dup=True, noise=3),
    "aut-serial250": variant("aut", serial=250, steps=1500),
    # ---- HELD3 (Round 10)
    "dist-noise4-p4": variant("dist", noise=4, p=.4),
    "aut-fooddelay4": variant("aut", delay=4),
    "dist-delay5-noise1": variant("dist", delay=5, noise=1),
    "aut-serial350": variant("aut", serial=350),
    "aut-fooddelay2-noise2": variant("aut", delay=2, noise=2),
    "dist-delay3-sticky2": variant("dist", delay=3, noise=2, sticky=.85),
    "aut-noise5": variant("aut", noise=5),
    "canc-noise5-p3": variant("canc", noise=5, p=.3),
    "aut-sqrt-noise2": variant("aut", reward=_sqrt, noise=2),
    "dist-affine": variant("dist", reward=_lin(.1, .8)),
    # ---- HELD4 (Round 11)
    "dist-noise5-p2": variant("dist", noise=5, p=.2),
    "aut-fooddelay3-noise2": variant("aut", delay=3, noise=2),
    "dist-delay4-sticky1": variant("dist", delay=4, noise=1, sticky=.9),
    "aut-serial400": variant("aut", serial=400),
    "aut-fooddelay5": variant("aut", delay=5),
    "canc-noise2-sticky9": variant("canc", noise=2, sticky=.9),
    "aut-noise3-p2": variant("aut", noise=3, p=.2),
    "aut-fooddelay2-sticky2": variant("aut", delay=2, noise=2, sticky=.85),
    "aut-square-noise2": variant("aut", reward=_square, noise=2),
    "canc-affine": variant("canc", reward=_lin(.1, .8)),
    # ---- HELD5 (Round 12)
    "dist-noise4-p3": variant("dist", noise=4, p=.3),
    "dist-delay3-noise3": variant("dist", delay=3, noise=3),
    "aut-fooddelay4-noise1": variant("aut", delay=4, noise=1),
    "aut-fooddelay3-sticky3": variant("aut", delay=3, noise=3, sticky=.9),
    "aut-serial450": variant("aut", serial=450),
    "canc-noise4-p2": variant("canc", noise=4, p=.2),
    "aut-noise6": variant("aut", noise=6),
    "dist-sqrt": variant("dist", reward=_sqrt),
    "aut-fooddelay2-noise3-p3": variant("aut", delay=2, noise=3, p=.3),
    "dist-delay4-noise2-sticky": variant("dist", delay=4, noise=2, sticky=.8),
    # ---- HELD6 (Round 13, consolidation): defined before any run on it, run once
    "aut-noise7-p3": variant("aut", noise=7, p=.3),
    "canc-noise5-sticky": variant("canc", noise=5, sticky=.85),
    "dist-delay4-noise1": variant("dist", delay=4, noise=1),
    "aut-fooddelay3-noise2-p4": variant("aut", delay=3, noise=2, p=.4),
    "aut-serial325": variant("aut", serial=325),
    "dist-noise3-sticky": variant("dist", noise=3, sticky=.9),
    "aut-cube": variant("aut", reward=_cube),
    "canc-half": variant("canc", reward=_half),
    "aut-dupaction-noise2-sticky": variant("aut", dup=True, noise=2, sticky=.8),
    "dist-delay5-sticky1": variant("dist", delay=5, noise=1, sticky=.9),
}

# split -> (variant names, seeds, env-seed base, brain-seed base)
SPLITS = {
    "dev": (["aut", "canc", "dist", "aut-half", "aut-noise4", "dist-delay3", "aut-serial300",
             "aut-noise4-sticky", "canc-noise3", "dist-noise2-sticky", "dist-noise2",
             "aut-fooddelay2", "dist-delay3-noise2", "dist-noise3-p4", "aut-fooddelay3-noise1"],
            range(8), 100, 0),
    "held": (["aut-high", "aut-square", "aut-noise8-sticky", "canc-noise6", "dist-delay4",
              "dist-shift", "dist-noise4", "aut-fooddelay2", "aut-dupaction", "aut-serial200"],
             range(12), 7000, 700),
    "held2": (["aut-noise6-sticky95", "canc-noise4-sticky", "dist-delay5", "dist-noise3",
               "dist-delay4-noise2", "aut-fooddelay3", "aut-affine", "aut-sqrt",
               "aut-dupaction-noise3", "aut-serial250"], range(12), 9000, 900),
    "held3": (["dist-noise4-p4", "aut-fooddelay4", "dist-delay5-noise1", "aut-serial350",
               "aut-fooddelay2-noise2", "dist-delay3-sticky2", "aut-noise5", "canc-noise5-p3",
               "aut-sqrt-noise2", "dist-affine"], range(12), 11000, 1100),
    "held4": (["dist-noise5-p2", "aut-fooddelay3-noise2", "dist-delay4-sticky1", "aut-serial400",
               "aut-fooddelay5", "canc-noise2-sticky9", "aut-noise3-p2", "aut-fooddelay2-sticky2",
               "aut-square-noise2", "canc-affine"], range(12), 13000, 1300),
    "held5": (["dist-noise4-p3", "dist-delay3-noise3", "aut-fooddelay4-noise1",
               "aut-fooddelay3-sticky3", "aut-serial450", "canc-noise4-p2", "aut-noise6",
               "dist-sqrt", "aut-fooddelay2-noise3-p3", "dist-delay4-noise2-sticky"],
              range(12), 15000, 1500),
    "held6": (["aut-noise7-p3", "canc-noise5-sticky", "dist-delay4-noise1",
               "aut-fooddelay3-noise2-p4", "aut-serial325", "dist-noise3-sticky", "aut-cube",
               "canc-half", "aut-dupaction-noise2-sticky", "dist-delay5-sticky1"],
              range(12), 17000, 1700),
}
DEV = SPLITS["dev"][0]
