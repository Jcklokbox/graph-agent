"""Why does a variance-gated optimism destroy a real payoff law it already has?"""
from __future__ import annotations
import argparse
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v53'); p.add_argument('--variant', default='aut')
    p.add_argument('--seed', type=int, default=6); p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed); x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor; head = f.reward
    for t in range(a.steps):
        b = f._bits(x)
        rule = head.rule()
        bound, _ = head.provisional(rule)
        gate = head.provisional_gate(b, f.hist)
        r, *_ = head.likelihood(rule, head.input(b, f.hist)[1:][None, :])
        inc = np.maximum(r[0], (bound * gate).max(1))
        trans = head.all_transitions(); code = int(head.code(b))
        states = np.flatnonzero(np.any(trans[:, code, :] > 0, axis=0))
        imm = np.array([float(trans[aa, code, states] @ inc[states]) for aa in range(3)])
        prov = np.array([float(trans[aa, code, states] @ ((bound * gate).max(1))[states]) for aa in range(3)])
        real = np.array([float(trans[aa, code, states] @ r[0][states]) for aa in range(3)])
        act = int(brain.step(x))
        if t < 24 or t % 20 == 0:
            print('t', t, 'pos', int(np.argmax(x[:3])), 'prec', int(x[3] > .5), int(x[4] > .5),
                  'slots', head.slots.tolist(), 'code', code)
            print('   real ', np.round(real, 3), 'prov', np.round(prov, 3), 'imm', np.round(imm, 3),
                  'argmax', int(imm.argmax()), 'chosen', act)
            print('   obs', np.round(x[:8], 2), 'r', np.round(r[0][states], 3))
        y, rw, _, _ = env.step(act); brain.learn(y); x = y


if __name__ == '__main__':
    main()
