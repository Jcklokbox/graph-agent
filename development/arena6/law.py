"""Is the cancellation payoff already representable in the learned law?

At each cue step, print the config-conditioned predicted payoff `r[c]` for every
control state, the marginal `base[c]`, and the configuration evidence, so we can
separate "the law cannot express the conjunction" from "exploration misses it".
"""
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
        if cue:
            b = f._bits(x); rule = head.rule()
            in0 = head.input(b, f.hist)[1:]
            r, pr, law, ev = head.likelihood(rule, in0[None, :])
            r = r[0]; law = law[0]; ev = ev[0]
            trans = head.all_transitions(); code = int(head.code(b))
            states = np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0))
            print('t', t, 'pos', int(np.argmax(x[:3])), 'lamp', int(lamp),
                  'slots', head.slots.tolist(), 'code', code)
            print('   states', states.tolist())
            print('   r     ', [round(float(r[c]), 4) for c in states])
            print('   base  ', [round(float(rule[5][c]), 4) for c in states])
            print('   ev    ', [round(float(ev[c]), 6) for c in states])
            print('   cmass ', [int(head.control_mass[c]) for c in states])
            print('   parents', {int(c): [int(j) for j in rule[0][c][rule[1][c]]] for c in states})
            print('   imm   ', [round(float(trans[aa, code, states] @ r[states]), 4) for aa in range(3)])
            if t > 60:
                break
        act = int(brain.step(x)); y, rw, _, _ = env.step(act); brain.learn(y); x = y


if __name__ == '__main__':
    main()
