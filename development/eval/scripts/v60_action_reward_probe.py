"""Short paired closed-loop check for the v60 action reward split.

Run from development/ with ``python -m eval.scripts.v60_action_reward_probe``.
The control disables only the new split. No qualification or saved evidence.
"""
from contextlib import nullcontext
import json
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent, JointReward


def episode(seed, period, *, action_split):
    agent = Agent(6, 2, 5, seed)
    x = np.array([0., 0., 0., 0., 0., .5])
    steps = 4 * period if period else 120
    rewards = np.empty(steps)
    scope = nullcontext() if action_split else patch.object(JointReward, 'constant_context_action_values',
                                                            return_value=None)
    with scope:
        for t in range(steps):
            best = t // period % 2 if period else 0
            a = agent.step(x)
            y = x.copy()
            y[-1] = .9 if a == best else .1
            agent.learn(y)
            x = y
            rewards[t] = y[-1]
    return dict(mean=float(rewards.mean()), after_switch=[float(rewards[t:t + 50].mean())
                                                         for t in range(period, steps, period)] if period else [])


def main():
    for period in (0, 200):
        for seed in (0, 1, 2):
            print(json.dumps(dict(seed=seed, period=period,
                                  control=episode(seed, period, action_split=False),
                                  updated=episode(seed, period, action_split=True))), flush=True)


if __name__ == '__main__':
    main()
