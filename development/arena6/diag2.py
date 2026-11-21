"""Exposed diagnostic: read the controller's action scores and reward head at every step.

Wraps the factor's `two_step_values` (read-only) to capture the action scores the
controller maximises. Also records reward-head state. Nothing is fed to the learner.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48'); p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6); p.add_argument('--steps', type=int, default=420)
    p.add_argument('--out', default='/tmp/d2.jsonl')
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed); x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor; head = f.reward
    cap = {}
    orig = f.two_step_values

    def wrapped(xx, u, theta=None):
        q = orig(xx, u, theta)
        cap['q'] = np.round(np.asarray(q, float), 6).tolist()
        return q

    f.two_step_values = wrapped
    rows = []
    for t in range(a.steps):
        slot = head.slots.copy()
        rows_pre = dict(t=t, pos=int(np.argmax(x[:3])), cue=int(x[3] > .5), trap=int(x[7] > .5))
        a_ch = int(brain.step(x))
        q = cap.get('q')
        y, r, _, inf = env.step(a_ch); brain.learn(y)
        rows.append(dict(**rows_pre, act=a_ch, q=q, rew=float(r),
                         shock=int(inf['received_shock']),
                         slots=[int(s) for s in slot],
                         anchors=list(map(int, getattr(head, 'anchors', []))),
                         tslots=[int(s) for s in head.slots]))
        x = y
    with open(a.out, 'a') as fh:
        fh.write(json.dumps(dict(k=[a.agent, a.variant, a.seed, a.steps], m=dict(rows=rows))) + '\n')


if __name__ == '__main__':
    main()
