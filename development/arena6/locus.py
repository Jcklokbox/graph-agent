"""Read the provisional-payoff quantities at cancellation opportunities."""
from __future__ import annotations
import argparse, json
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def snap(head, f, b):
    rule = head.rule()
    bound, var = head.provisional(rule)
    seen = head.phase(b, f.hist)
    inc = np.maximum(rule[5], (bound * seen).max(1))
    trans = head.all_transitions()
    code = int(head.code(b))
    states = np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0)) if head.sparse_outcomes else head.ci
    d = dict(code=code, states=states.tolist(), base=np.round(rule[5], 4).tolist(),
             inc=np.round(inc, 4).tolist(),
             bound_max=np.round(bound.max(1), 4).tolist(),
             feat=np.round(bound.max(0), 4).tolist(),
             n=np.round(head.onset.max(1), 1).tolist(),
             ons_total=np.round(head.onset_total, 1).tolist(),
             past_on=np.round(head.past_on, 1).tolist(),
             immediate=[round(float(trans[a, code, states] @ inc[states]), 4) for a in range(3)],
             trans=np.round(trans[:, code, :], 4).tolist(),
             slots=head.slots.tolist(), seen=[int(s) for s in np.flatnonzero(seen)],
             )
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48'); p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6); p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed); x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor
    for t in range(a.steps):
        cue, lamp = x[3] > .5, x[7] > .5
        pos = int(np.argmax(x[:3]))
        if cue and not lamp:
            s = snap(f.reward, f, f._bits(x))
            print('t', t, 'pos', pos, 'slots', s['slots'], 'code', s['code'],
                  'seen', s['seen'])
            print('   immediate', s['immediate'])
            print('   base  ', [s['base'][i] for i in s['states']])
            print('   inc   ', [s['inc'][i] for i in s['states']])
            print('   bmax  ', [s['bound_max'][i] for i in s['states']])
            print('   onsetn', [s['n'][i] for i in s['states']])
            print('   trans ', s['trans'])
            break
        act = int(brain.step(x)); y, r, _, _ = env.step(act); brain.learn(y); x = y


if __name__ == '__main__':
    main()
