"""Decompose the controller's action scores at cancellation opportunities.

At each step, reconstruct (read-only) the terms of `provisional_values` and
record the per-action score plus the optimistic upper bound at every successor
state, so the ranking defect can be seen directly.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load
from models.factor_projection import factor_current
from models.factor_reward import conditional_draws
import models.factor_rollout as R


def capture(f, x, uniforms):
    head = f.reward
    b = f._bits(x)
    rule = head.rule()
    in0 = head.input(b, f.hist)[1:]
    r, prob, _, _ = head.likelihood(rule, in0[None, :])
    r = r[0]
    splits = head.signed_rules(rule) if hasattr(head, 'signed_rules') else []
    seen = head.phase(b, f.hist)
    if splits:
        _, _, out0, _, g0 = head.context_moment(splits, in0[None, :], seen[None, :],
                                                r[None, :], prob[None, :], rule[3])
        atoms0 = out0[0]
    else:
        atoms0 = rule[3]
    bound, _, _ = head.provisional(rule)
    inc = np.maximum(r, (bound * seen).max(1))
    code = int(head.code(b))
    trans = head.all_transitions()
    states = np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0)) if getattr(head, 'sparse_outcomes', False) else head.ci
    return dict(code=code, r=np.round(r, 5).tolist(), inc=np.round(inc, 5).tolist(),
                bound=np.round(bound, 4).tolist(),
                bound_seen=np.round((bound * seen).max(1), 5).tolist(),
                base=np.round(rule[5], 5).tolist(),
                parents=rule[1].any(1).astype(int).tolist(),
                seen=[int(s) for s in np.flatnonzero(seen)],
                states=states.tolist(), slots=head.slots.tolist(),
                trans=np.round(trans[:, code, :], 4).tolist(),
                ttotal=np.round(head.trans_total[:, code], 2).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48'); p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6); p.add_argument('--steps', type=int, default=420)
    p.add_argument('--out', default='/tmp/decomp.jsonl')
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed); x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor
    rows = []
    for t in range(a.steps):
        cap = capture(f, x, None)
        act = int(brain.step(x))
        y, r, _, inf = env.step(act); brain.learn(y)
        rows.append(dict(t=t, pos=int(np.argmax(x[:3])), cue=int(x[3] > .5), lamp=int(x[7] > .5),
                         act=act, rew=float(r), shock=int(inf['received_shock']), **cap))
        x = y
    with open(a.out, 'a') as fh:
        fh.write(json.dumps(dict(k=[a.agent, a.variant, a.seed, a.steps], m=dict(rows=rows))) + '\n')


if __name__ == '__main__':
    main()
