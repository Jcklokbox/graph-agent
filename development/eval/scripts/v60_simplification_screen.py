"""Whole-component and particle-budget screens on short, exposed dev paths.

No action overrides, no held-out cases and no qualification harness. Candidates
are cold starts with the same public online controller and seed offsets.
"""
import argparse
import hashlib
import json
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agents.v60 import Agent
from envs.tasks import SPLITS, VARIANTS
from models.factor import CANONICAL, FactorModel, make_factor


SCREEN = (
    ('dist-delay3', 1905953054, 160),
    ('dist-delay3-noise2', 91, 160),
    ('canc-noise3', 15, 160),
    ('aut-noise4-sticky', 26, 160),
    ('aut-serial300', 6, 380),
)
CHECK = (
    ('aut-fooddelay3-noise1', 37, 192),
    ('dist-noise3-p4', 59, 192),
    ('canc', 37, 192),
    ('aut-serial300', 1822923777, 380),
)
STRESS = (('aut-serial300', 6, 680), ('aut-noise4-sticky', 26, 400))
ARMS = ('reference', 'no-process', '32-particles')
BANKS = ('q1', 'q0', 'ev', '_E', 'mu', 'Sg', 'rate0', 'rate1', 'cue', 'mz')


def build(variant, seed, arm):
    if arm == 'reference':
        return Agent(variant.n_inputs, variant.n_actions, 11, seed=seed)

    def factor(name, n, actions, value, seed):
        if name != CANONICAL:
            raise ValueError(name)
        if arm == '32-particles':
            return make_factor(f'{name}-N32', n, actions, value, seed)
        if arm == 'no-process':
            return FactorModel(n, actions, value, seed, N=200, lags=range(5),
                               own_lags=(0, 1, 2), k=5, process=False, cond=True)
        raise ValueError(arm)

    with patch('agents.factor_controller.make_factor', factor):
        return Agent(variant.n_inputs, variant.n_actions, 11, seed=seed)


def run(task, seed, steps, arm):
    _, _, eb, bb = SPLITS['dev']
    variant = VARIANTS[task]
    env, agent = variant.make(eb + seed), build(variant, bb + seed, arm)
    f = agent.factor
    obs = env.reset()
    rewards, elapsed, actions = [], [], []
    for _ in range(steps):
        start = perf_counter()
        action = agent.step(obs)
        obs, reward, _, _ = env.step(action)
        agent.learn(obs)
        elapsed.append(perf_counter() - start)
        actions.append(action)
        rewards.append(float(reward))
    return dict(task=task, seed=seed, steps=steps, arm=arm,
                total=float(np.sum(rewards)), mean=float(np.mean(rewards)),
                first30=float(np.mean(rewards[:30])), last80=float(np.mean(rewards[-80:])),
                action_sha256=hashlib.sha256(np.asarray(actions, dtype='<i4').tobytes()).hexdigest(),
                bank_bytes=sum(getattr(f, k).nbytes for k in BANKS),
                median_ms=float(np.median(elapsed[30:]) * 1000), switches=f.reward.switches)


def fixed_tape(seed):
    """Same real transitions and uniforms, to distinguish score from action parity."""
    spec = SimpleNamespace(n_inputs=12, n_actions=3)
    agents = [build(spec, seed, arm) for arm in ARMS]
    uniforms = np.random.default_rng(19).random((5, 15))
    scores = []
    pos = 0
    for t in range(96):
        b = np.zeros(12)
        b[pos], b[3], b[4], b[-1] = 1., t % 7 < 2, t % 11 < 3, .45
        a = t % 3
        scores.append([agent.factor.two_step_values(b, uniforms) for agent in agents])
        pos = (pos + a) % 3
        y = np.zeros_like(b)
        y[pos], y[3], y[4] = 1., (t + 1) % 7 < 2, (t + 1) % 11 < 3
        y[-1] = .5 if pos == 1 else .45 if pos == 2 else b[3] * (1. - b[4])
        for agent in agents:
            agent.factor.predict(b, a)
            agent.factor.learn(b, a, y)
    q = np.asarray(scores)
    for i, arm in enumerate(ARMS[1:], 1):
        delta = q[:, i] - q[:, 0]
        yield dict(seed=seed, arm=arm, differing_rows=int(np.any(delta != 0., axis=1).sum()),
                   max_abs_delta=float(np.abs(delta).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=int, nargs='?')
    panels = parser.add_mutually_exclusive_group()
    panels.add_argument('--check', action='store_true')
    panels.add_argument('--stress', action='store_true')
    panels.add_argument('--numeric', action='store_true')
    args = parser.parse_args()
    if args.numeric:
        if args.case is not None:
            parser.error('numeric tape has no case index')
        for seed in (7, 29):
            for record in fixed_tape(seed):
                print(json.dumps(record), flush=True)
        return
    panel = STRESS if args.stress else CHECK if args.check else SCREEN
    arms = ('reference', '32-particles') if args.stress else ARMS
    if args.case is not None and not 0 <= args.case < len(panel):
        parser.error('case index outside the selected panel')
    for case in (panel if args.case is None else (panel[args.case],)):
        for arm in arms:
            print(json.dumps(run(*case, arm)), flush=True)


if __name__ == '__main__':
    main()
