"""Exposed diagnostic: separate absent-payoff discovery from post-exposure reuse.

Runs one episode on a named variant with a chosen agent, logging per-step the
observation, action, reward, shocks and the reward member's internal belief that
a food delivery is available. No teacher flag is fed to the learner: the script
only observes the episode from the outside.

Usage (development/ on PYTHONPATH):
    python arena6/trace_canc.py --agent v48 --variant canc --seed 6 --steps 420
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def build(model, v, seed, dev_seed):
    return agent_class(load(model))(v.n_inputs, v.n_actions, 11, seed=seed)


def head_belief(head, b):
    """Best current bounded-reward belief for the current observation, no lookahead."""
    try:
        rule = head.rule()
    except Exception:
        return None
    x = head.input(b, np.zeros_like(b)[None][:0])[1:] if False else None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48')
    p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--steps', type=int, default=420)
    p.add_argument('--out', default=None)
    p.add_argument('--slot-prefix', action='store_true')
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed)
    x = env.reset()
    brain = build(a.agent, v, bb + a.seed, a.seed)
    f = brain.factor
    rows = []
    exposure = None
    for t in range(a.steps):
        b = f._bits(x)
        slots_before = f.reward.slots.copy()
        cuetrap = (x[3] > .5, x[7] > .5, x[0] > .5, x[2] > .5)
        if a.slot_prefix and exposure is None and not cuetrap[0]:
            # scripted free exploration prefix: choose the action the oracle would
            pass
        chosen = brain.step(x)
        y, r, _, info = env.step(chosen)
        brain.learn(y)
        shocked = bool(info['received_shock'])
        if r == 1.0 and exposure is None:
            exposure = t
        rows.append(dict(t=t, pos=[int(x[0] > .5), int(x[1] > .5), int(x[2] > .5)],
                         cue=int(x[3] > .5), trap=int(x[7] > .5),
                         act=int(chosen), r=float(r), shock=int(shocked),
                         post=[int(y[0] > .5), int(y[1] > .5), int(y[2] > .5)],
                         slots=[int(s) for s in f.reward.slots],
                         an=list(map(int, getattr(f.reward, 'anchors', [])))))
        x = y
    rec = dict(k=[a.agent, a.variant, a.seed, a.steps], m=dict(rows=rows, exposure=exposure))
    if a.out:
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    else:
        print(json.dumps(rec))


if __name__ == '__main__':
    main()
