"""Transfer benchmark: PosteriorGraph on *variations* of the base tasks.

The main quality criterion is transfer to task variants that were not used
while designing the model, not new seeds of the same task.

* DEV variants (8 seeds, env 100+, brain 0+) may be used for model selection.
* HELD-OUT variants (12 seeds, env 7000+, brain 700+) are run once, after
  every design choice is frozen. They include new parameter values of the DEV
  families and families absent from DEV (non-linear reward map, delayed food
  in BlindGate, a duplicated action, combinations).

Score per variant: normalised tail return (brain - stay) / (oracle - stay),
stay = always action 1 (centre). Tail = last 200 steps; for serial reversal
every step after the first phase (adaptation is part of the task).

    python transfer_suite.py --dev  [--kw '{"explore": 2}']
    python transfer_suite.py --held [--kw ...]
    python transfer_suite.py --dev --seeds 4   # SCREENING ONLY (first 4 seeds)

`--seeds K` runs only the first K seeds of the split. It is a cheap screen
for discarding clearly worse ideas; configuration choices are still made on
all 8 DEV seeds and held-out splits are always reported on all 12.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque

import numpy as np

from benchmark_envs import CancellationGateEnv, DistantGateEnv
from blind_gate import BlindGateEnv, oracle_policy
from posterior_graph import PosteriorGraph
from run_posterior_scenarios import Relabel, cancellation_oracle, pmap

BRAIN_KW: dict = {}


# ------------------------------------------------------------------ wrappers
def _shock(info):
    return bool(info.get("received_shock", False) if isinstance(info, dict)
                else getattr(info, "received_shock", False))


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
    """k extra channels of task-irrelevant noise appended to the observation.

    sticky = 0: i.i.d. Bernoulli(p); sticky > 0: two-state Markov chain that
    keeps its value with probability `sticky` (stationary mean p)."""

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


class DistantDelay(DistantGateEnv):
    """DistantGate with the food window `delay` steps after the cue (base: 2)."""

    def __init__(self, seed=0, delay=3):
        self.delay = delay
        super().__init__(seed)

    def step(self, action):
        o, r, d, i = super().step(action)
        if self.last_cue == self.t:
            self.food_time = self.t + self.delay
        return o, r, d, i


class BlindGateFoodDelay(BlindGateEnv):
    """BlindGate where food appears `delay` steps after its precursor (base: 1)."""

    def __init__(self, seed=0, delay=2):
        self.delay = delay
        super().__init__(seed=seed)

    def reset(self, seed=None):
        self._hist = deque([None] * self.delay, maxlen=self.delay)
        return super().reset(seed)

    def step(self, action):
        chosen = self._decode_action(action)
        self._hist.append(self._precursor_side)
        food = self._hist[0]
        self.position = int(np.clip(self.position + chosen - 1, 0, 2))
        self.step_count += 1
        self._food_side = food
        self._precursor_side = None
        if self.step_count >= self._next_event_step:
            self._precursor_side = int(self.rng.integers(0, 2))
            gap = int(self.rng.integers(*self.event_interval, endpoint=True)) + self.delay - 1
            self._next_event_step = self.step_count + gap
        got = (self.position == 0 and food == 0) or (self.position == 2 and food == 1)
        shock = self.position != 1 and not got
        self.wellbeing = 1.0 if got else (0.0 if shock else .5)
        from blind_gate import StepInfo
        info = StepInfo(self.step_count, self.position, chosen, self._precursor_side,
                        self._food_side, bool(got), bool(shock))
        return self._observation(), self.wellbeing, False, info


# ------------------------------------------------------------------- oracles
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
# name -> (env factory(seed), n_inputs, n_actions, steps, oracle factory, raw-env factory)
def _variants():
    sq = lambda w: w * w
    V = {
        # ---- DEV
        "aut": (lambda s: BlindGateEnv(seed=s), 12, 3, 600, lambda: oracle_policy),
        "canc": (lambda s: CancellationGateEnv(seed=s), 12, 3, 600, lambda: cancellation_oracle),
        "dist": (lambda s: DistantGateEnv(seed=s), 12, 3, 800, lambda: distant_oracle(2)),
        "aut-half": (lambda s: RewardMap(BlindGateEnv(seed=s), lambda w: .5 * w), 12, 3, 600,
                     lambda: oracle_policy),
        "aut-noise4": (lambda s: Distractors(BlindGateEnv(seed=s), 4, s), 16, 3, 600,
                       lambda: oracle_policy),
        "dist-delay3": (lambda s: DistantDelay(seed=s, delay=3), 12, 3, 800,
                        lambda: distant_oracle(3)),
        "aut-serial300": (lambda s: SerialReversal(s, 300), 12, 3, 1500, None),
        # ---- DEV, added in Round 8 (noise forms that failed on HELD, other parameters)
        "aut-noise4-sticky": (lambda s: Distractors(BlindGateEnv(seed=s), 4, s, sticky=.9), 16, 3,
                              600, lambda: oracle_policy),
        "canc-noise3": (lambda s: Distractors(CancellationGateEnv(seed=s), 3, s), 15, 3, 600,
                        lambda: cancellation_oracle),
        "dist-noise2-sticky": (lambda s: Distractors(DistantGateEnv(seed=s), 2, s, p=.3, sticky=.8),
                               14, 3, 800, lambda: distant_oracle(2)),
        # ---- HELD-OUT
        "aut-high": (lambda s: RewardMap(BlindGateEnv(seed=s), lambda w: .5 + .5 * w), 12, 3, 600,
                     lambda: oracle_policy),
        "aut-square": (lambda s: RewardMap(BlindGateEnv(seed=s), sq), 12, 3, 600,
                       lambda: oracle_policy),
        "aut-noise8-sticky": (lambda s: Distractors(BlindGateEnv(seed=s), 8, s, sticky=.8), 20, 3,
                              600, lambda: oracle_policy),
        "canc-noise6": (lambda s: Distractors(CancellationGateEnv(seed=s), 6, s, p=.3), 18, 3, 600,
                        lambda: cancellation_oracle),
        "dist-delay4": (lambda s: DistantDelay(seed=s, delay=4), 12, 3, 800,
                        lambda: distant_oracle(4)),
        "dist-shift": (lambda s: RewardMap(DistantGateEnv(seed=s), lambda w: .25 + .5 * w), 12, 3,
                       800, lambda: distant_oracle(2)),
        "dist-noise4": (lambda s: Distractors(DistantGateEnv(seed=s), 4, s), 16, 3, 800,
                        lambda: distant_oracle(2)),
        "aut-fooddelay2": (lambda s: BlindGateFoodDelay(seed=s, delay=2), 12, 3, 600,
                           lambda: food_delay_oracle(2)),
        "aut-dupaction": (lambda s: DuplicateAction(BlindGateEnv(seed=s)), 12, 4, 600,
                          lambda: oracle_policy),
        "aut-serial200": (lambda s: SerialReversal(s, 200), 12, 3, 1400, None),
        # ---- HELD2 (Round 8): fresh variations, defined before any run on them
        "aut-noise6-sticky95": (lambda s: Distractors(BlindGateEnv(seed=s), 6, s, p=.3, sticky=.95),
                                18, 3, 600, lambda: oracle_policy),
        "canc-noise4-sticky": (lambda s: Distractors(CancellationGateEnv(seed=s), 4, s, sticky=.8),
                               16, 3, 600, lambda: cancellation_oracle),
        "dist-delay5": (lambda s: DistantDelay(seed=s, delay=5), 12, 3, 800,
                        lambda: distant_oracle(5)),
        "dist-noise3": (lambda s: Distractors(DistantGateEnv(seed=s), 3, s, p=.3), 15, 3, 800,
                        lambda: distant_oracle(2)),
        "dist-delay4-noise2": (lambda s: Distractors(DistantDelay(seed=s, delay=4), 2, s), 14, 3,
                               800, lambda: distant_oracle(4)),
        "aut-fooddelay3": (lambda s: BlindGateFoodDelay(seed=s, delay=3), 12, 3, 600,
                           lambda: food_delay_oracle(3)),
        "aut-affine": (lambda s: RewardMap(BlindGateEnv(seed=s), lambda w: .2 + .3 * w), 12, 3, 600,
                       lambda: oracle_policy),
        "aut-sqrt": (lambda s: RewardMap(BlindGateEnv(seed=s), lambda w: w ** .5), 12, 3, 600,
                     lambda: oracle_policy),
        "aut-dupaction-noise3": (lambda s: Distractors(DuplicateAction(BlindGateEnv(seed=s)), 3, s),
                                 15, 4, 600, lambda: oracle_policy),
        "aut-serial250": (lambda s: SerialReversal(s, 250), 12, 3, 1500, None),
        # ---- DEV, added in Round 10: the failure families of HELD2 with other
        # parameters (white noise in dist, delayed food, long delay with noise)
        "dist-noise2": (lambda s: Distractors(DistantGateEnv(seed=s), 2, s), 14, 3, 800,
                        lambda: distant_oracle(2)),
        "dist-delay3-noise2": (lambda s: Distractors(DistantDelay(seed=s, delay=3), 2, s), 14, 3,
                               800, lambda: distant_oracle(3)),
        # ---- DEV, added in Round 11: the HELD3 failure families with other
        # parameters (more iid noise in dist, delayed food with noise)
        "dist-noise3-p4": (lambda s: Distractors(DistantGateEnv(seed=s), 3, s, p=.4), 15, 3, 800,
                           lambda: distant_oracle(2)),
        "aut-fooddelay3-noise1": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=3), 1, s),
                                  13, 3, 600, lambda: food_delay_oracle(3)),
        # ---- HELD3 (Round 10): fresh variations, defined before any run on them
        "dist-noise4-p4": (lambda s: Distractors(DistantGateEnv(seed=s), 4, s, p=.4), 16, 3, 800,
                           lambda: distant_oracle(2)),
        "aut-fooddelay4": (lambda s: BlindGateFoodDelay(seed=s, delay=4), 12, 3, 600,
                           lambda: food_delay_oracle(4)),
        "dist-delay5-noise1": (lambda s: Distractors(DistantDelay(seed=s, delay=5), 1, s), 13, 3,
                               800, lambda: distant_oracle(5)),
        "aut-serial350": (lambda s: SerialReversal(s, 350), 12, 3, 1750, None),
        "aut-fooddelay2-noise2": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=2), 2, s),
                                  14, 3, 600, lambda: food_delay_oracle(2)),
        "dist-delay3-sticky2": (lambda s: Distractors(DistantDelay(seed=s, delay=3), 2, s,
                                                      sticky=.85), 14, 3, 800,
                                lambda: distant_oracle(3)),
        "aut-noise5": (lambda s: Distractors(BlindGateEnv(seed=s), 5, s), 17, 3, 600,
                       lambda: oracle_policy),
        "canc-noise5-p3": (lambda s: Distractors(CancellationGateEnv(seed=s), 5, s, p=.3), 17, 3,
                           600, lambda: cancellation_oracle),
        "aut-sqrt-noise2": (lambda s: Distractors(RewardMap(BlindGateEnv(seed=s), lambda w: w ** .5),
                                                  2, s), 14, 3, 600, lambda: oracle_policy),
        "dist-affine": (lambda s: RewardMap(DistantGateEnv(seed=s), lambda w: .1 + .8 * w), 12, 3,
                        800, lambda: distant_oracle(2)),
        # ---- HELD4 (Round 11): fresh variations, defined before any run on them
        "dist-noise5-p2": (lambda s: Distractors(DistantGateEnv(seed=s), 5, s, p=.2), 17, 3, 800,
                           lambda: distant_oracle(2)),
        "aut-fooddelay3-noise2": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=3), 2, s),
                                  14, 3, 600, lambda: food_delay_oracle(3)),
        "dist-delay4-sticky1": (lambda s: Distractors(DistantDelay(seed=s, delay=4), 1, s,
                                                      sticky=.9), 13, 3, 800,
                                lambda: distant_oracle(4)),
        "aut-serial400": (lambda s: SerialReversal(s, 400), 12, 3, 2000, None),
        "aut-fooddelay5": (lambda s: BlindGateFoodDelay(seed=s, delay=5), 12, 3, 600,
                           lambda: food_delay_oracle(5)),
        "canc-noise2-sticky9": (lambda s: Distractors(CancellationGateEnv(seed=s), 2, s, sticky=.9),
                                14, 3, 600, lambda: cancellation_oracle),
        "aut-noise3-p2": (lambda s: Distractors(BlindGateEnv(seed=s), 3, s, p=.2), 15, 3, 600,
                          lambda: oracle_policy),
        "aut-fooddelay2-sticky2": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=2), 2, s,
                                                         sticky=.85), 14, 3, 600,
                                   lambda: food_delay_oracle(2)),
        "aut-square-noise2": (lambda s: Distractors(RewardMap(BlindGateEnv(seed=s), sq), 2, s), 14,
                              3, 600, lambda: oracle_policy),
        "canc-affine": (lambda s: RewardMap(CancellationGateEnv(seed=s), lambda w: .1 + .8 * w), 12,
                        3, 600, lambda: cancellation_oracle),
        # ---- HELD5 (Round 12): fresh variations, defined before any run on them
        "dist-noise4-p3": (lambda s: Distractors(DistantGateEnv(seed=s), 4, s, p=.3), 16, 3, 800,
                           lambda: distant_oracle(2)),
        "dist-delay3-noise3": (lambda s: Distractors(DistantDelay(seed=s, delay=3), 3, s), 15, 3,
                               800, lambda: distant_oracle(3)),
        "aut-fooddelay4-noise1": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=4), 1, s),
                                  13, 3, 600, lambda: food_delay_oracle(4)),
        "aut-fooddelay3-sticky3": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=3), 3, s,
                                                         sticky=.9), 15, 3, 600,
                                   lambda: food_delay_oracle(3)),
        "aut-serial450": (lambda s: SerialReversal(s, 450), 12, 3, 2250, None),
        "canc-noise4-p2": (lambda s: Distractors(CancellationGateEnv(seed=s), 4, s, p=.2), 16, 3,
                           600, lambda: cancellation_oracle),
        "aut-noise6": (lambda s: Distractors(BlindGateEnv(seed=s), 6, s), 18, 3, 600,
                       lambda: oracle_policy),
        "dist-sqrt": (lambda s: RewardMap(DistantGateEnv(seed=s), lambda w: w ** .5), 12, 3, 800,
                      lambda: distant_oracle(2)),
        "aut-fooddelay2-noise3-p3": (lambda s: Distractors(BlindGateFoodDelay(seed=s, delay=2), 3, s,
                                                           p=.3), 15, 3, 600,
                                     lambda: food_delay_oracle(2)),
        "dist-delay4-noise2-sticky": (lambda s: Distractors(DistantDelay(seed=s, delay=4), 2, s,
                                                            sticky=.8), 14, 3, 800,
                                      lambda: distant_oracle(4)),
    }
    return V


VARIANTS = _variants()
DEV = ["aut", "canc", "dist", "aut-half", "aut-noise4", "dist-delay3", "aut-serial300",
       "aut-noise4-sticky", "canc-noise3", "dist-noise2-sticky",
       "dist-noise2", "aut-fooddelay2", "dist-delay3-noise2",
       "dist-noise3-p4", "aut-fooddelay3-noise1"]
HELD = ["aut-high", "aut-square", "aut-noise8-sticky", "canc-noise6", "dist-delay4",
        "dist-shift", "dist-noise4", "aut-fooddelay2", "aut-dupaction", "aut-serial200"]
HELD2 = ["aut-noise6-sticky95", "canc-noise4-sticky", "dist-delay5", "dist-noise3",
         "dist-delay4-noise2", "aut-fooddelay3", "aut-affine", "aut-sqrt", "aut-dupaction-noise3",
         "aut-serial250"]
HELD3 = ["dist-noise4-p4", "aut-fooddelay4", "dist-delay5-noise1", "aut-serial350",
         "aut-fooddelay2-noise2", "dist-delay3-sticky2", "aut-noise5", "canc-noise5-p3",
         "aut-sqrt-noise2", "dist-affine"]
HELD4 = ["dist-noise5-p2", "aut-fooddelay3-noise2", "dist-delay4-sticky1", "aut-serial400",
         "aut-fooddelay5", "canc-noise2-sticky9", "aut-noise3-p2", "aut-fooddelay2-sticky2",
         "aut-square-noise2", "canc-affine"]
HELD5 = ["dist-noise4-p3", "dist-delay3-noise3", "aut-fooddelay4-noise1", "aut-fooddelay3-sticky3",
         "aut-serial450", "canc-noise4-p2", "aut-noise6", "dist-sqrt", "aut-fooddelay2-noise3-p3",
         "dist-delay4-noise2-sticky"]
SEEDS = {"dev": (range(8), 100, 0), "held": (range(12), 7000, 700),
         "held2": (range(12), 9000, 900), "held3": (range(12), 11000, 1100),
         "held4": (range(12), 13000, 1300), "held5": (range(12), 15000, 1500)}


def _window(name, n):
    if name.startswith("aut-serial"):
        period = int(name[len("aut-serial"):])
        return period, n
    return n - 200, n


def _policy_run(name, env_seed, policy):
    make, _, _, n, _ = VARIANTS[name]
    env = make(env_seed)
    obs = env.reset()
    R = []
    for _ in range(n):
        obs, r, _, _ = env.step(policy(obs))
        R.append(float(r))
    lo, hi = _window(name, n)
    return float(np.mean(R[lo:hi]))


def _oracle_run(name, env_seed):
    make, _, _, n, ofac = VARIANTS[name]
    if ofac is not None:
        return _policy_run(name, env_seed, ofac())
    # Serial reversal: the oracle acts on the un-relabelled world.
    env = make(env_seed)
    env.reset()
    R = []
    for _ in range(n):
        if env.t > 0 and env.t % env.period == 0:
            env.env.reverse()
        env.t += 1
        inner = env.env.env
        o = inner._observation()
        _, r, _, _ = inner.step(oracle_policy(o))
        R.append(float(r))
    lo, hi = _window(name, n)
    return float(np.mean(R[lo:hi]))


def brain_run(name, env_seed, brain_seed):
    make, n_in, n_act, n, _ = VARIANTS[name]
    env = make(env_seed)
    b = PosteriorGraph(n_in, n_act, 11, seed=brain_seed, **BRAIN_KW)
    obs = env.reset()
    R, S = [], []
    t0 = time.time()
    for _ in range(n):
        a = b.step(obs)
        obs, r, _, info = env.step(a)
        b.learn(obs)
        R.append(float(r))
        S.append(_shock(info))
    lo, hi = _window(name, n)
    stay = _policy_run(name, env_seed, lambda o: 1)
    orc = _oracle_run(name, env_seed)
    return dict(brain=float(np.mean(R[lo:hi])), stay=stay, oracle=orc,
                shocks=float(np.mean(S)), late_shocks=float(np.mean(S[lo:hi])),
                sec=time.time() - t0)


def _job(name, env_seed, brain_seed):
    return brain_run(name, env_seed, brain_seed)


def run(split, names=None, verbose=True, n_seeds=None):
    seeds, eb, bb = SEEDS[split]
    if n_seeds is not None:
        seeds = seeds[:n_seeds]
    names = names or {"dev": DEV, "held": HELD, "held2": HELD2, "held3": HELD3,
                      "held4": HELD4, "held5": HELD5}[split]
    jobs = [(nm, eb + s, bb + s) for nm in names for s in seeds]
    # longest first
    jobs.sort(key=lambda j: -VARIANTS[j[0]][3] * VARIANTS[j[0]][1] ** 2)
    res = dict(zip(jobs, pmap(_job, jobs)))
    rows = {}
    for nm in names:
        rs = [res[(nm, eb + s, bb + s)] for s in seeds]
        gap = [r["oracle"] - r["stay"] for r in rs]
        norm = [(r["brain"] - r["stay"]) / g if g > 1e-9 else float("nan") for r, g in zip(rs, gap)]
        rows[nm] = dict(score=float(np.mean(norm)),
                        wins=f"{sum(r['brain'] > r['stay'] + 1e-9 for r in rs)}/{len(rs)}",
                        brain=float(np.mean([r["brain"] for r in rs])),
                        stay=float(np.mean([r["stay"] for r in rs])),
                        oracle=float(np.mean([r["oracle"] for r in rs])),
                        shocks=float(np.mean([r["shocks"] for r in rs])),
                        late_shocks=float(np.mean([r["late_shocks"] for r in rs])),
                        sec=float(np.mean([r["sec"] for r in rs])))
    if verbose:
        print(f"{'variant':<20}{'score':>7}{'wins':>7}{'brain':>7}{'stay':>7}{'oracle':>7}"
              f"{'shock':>7}{'late':>7}{'s/ep':>7}")
        for nm, r in rows.items():
            print(f"{nm:<20}{r['score']:7.3f}{r['wins']:>7}{r['brain']:7.3f}{r['stay']:7.3f}"
                  f"{r['oracle']:7.3f}{r['shocks']:7.3f}{r['late_shocks']:7.3f}{r['sec']:7.1f}")
        tr = [r["score"] for nm, r in rows.items() if nm not in ("aut", "canc", "dist")]
        print(f"mean score: all {np.mean([r['score'] for r in rows.values()]):.3f}"
              + (f" | variants only {np.mean(tr):.3f}" if tr else ""))
        r7 = DEV[:7]
        if split == "dev" and all(nm in rows for nm in r7):
            print(f"Round-7 DEV set: all {np.mean([rows[nm]['score'] for nm in r7]):.3f}"
                  f" | variants only {np.mean([rows[nm]['score'] for nm in r7[3:]]):.3f}")
    return rows


if __name__ == "__main__":
    if "--kw" in sys.argv:
        BRAIN_KW.update(json.loads(sys.argv[sys.argv.index("--kw") + 1]))
    import posterior_graph as _pg
    for i, a in enumerate(sys.argv):
        if a == "--set":   # experiment switches: --set NAME=json (module attribute)
            k, v = sys.argv[i + 1].split("=", 1)
            setattr(_pg, k, json.loads(v))
    split = ("held5" if "--held5" in sys.argv else "held4" if "--held4" in sys.argv else "held3" if "--held3" in sys.argv else "held2" if "--held2" in sys.argv
             else "held" if "--held" in sys.argv else "dev")
    only = [a for a in sys.argv[1:] if a in VARIANTS]
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else None
    if n_seeds is not None:
        if split != "dev":
            sys.exit("--seeds is a DEV screening option; held-out splits use all seeds")
        print(f"SCREENING ONLY: first {n_seeds} of {len(SEEDS[split][0])} seeds")
    print("settings:", BRAIN_KW, {a: getattr(_pg, a) for a in dir(_pg)
                                  if a.isupper() and isinstance(getattr(_pg, a), (bool, int, float))})
    run(split, only or None, n_seeds=n_seeds)
