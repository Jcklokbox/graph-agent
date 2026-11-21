"""Dump the provisional (optimistic payoff) cells that drive a cue decision."""
from __future__ import annotations
import argparse
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48'); p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6); p.add_argument('--steps', type=int, default=140)
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed); x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor; head = f.reward
    for t in range(a.steps):
        cue, lamp = x[3] > .5, x[7] > .5
        if cue and not lamp:
            b = f._bits(x); rule = head.rule(); bound, _ = head.provisional(rule)
            seen = head.phase(b, f.hist)
            trans = head.all_transitions(); code = int(head.code(b))
            states = np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0))
            print('t', t, 'slots', head.slots.tolist(), 'code', code, 'seen_feats',
                  np.flatnonzero(seen).tolist())
            n = head.onset; m = (head.onset_reward + .5) / (1. + n)
            strength = (head.onset_total / (.5 + head.past_on))
            prev = (.5 + head.present) / (1. + head.total)
            known = rule[1].any(1)
            for j in (3, 7):
                print('  feature', j, 'cue' if j == 3 else 'lamp',
                      'strength %.3f prev %.3f onset_total %.0f past_on %.0f' %
                      (strength[j], prev[j], head.onset_total[j], head.past_on[j]))
                for c in states:
                    print('     state %d n[%d,%d]=%.0f m=%.3f u=%.3f known=%d' %
                          (c, c, j, n[c, j], m[c, j], bound[c, j], known[c]))
            inc = np.maximum(rule[5], (bound * seen).max(1))
            print('   imm', [round(float(trans[aa, code, states] @ inc[states]), 4) for aa in range(3)],
                  'argmax', int(np.argmax([float(trans[aa, code, states] @ inc[states]) for aa in range(3)])))
            print('   base', [round(rule[5][c], 3) for c in states],
                  'inc', [round(inc[c], 3) for c in states])
            break
        act = int(brain.step(x)); y, r, _, _ = env.step(act); brain.learn(y); x = y


if __name__ == '__main__':
    main()
