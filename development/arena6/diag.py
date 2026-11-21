"""Exposed diagnostic: why does v48 miss the cancellation action on `canc`?

For one episode, at every step record the agent's decision-time quantities from
the unified factor+controller object (`two_step_values` internals), plus the
environment-side opportunity. Nothing here reaches the learner.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load
from models.factor_projection import factor_current
from models.factor_reward import conditional_draws


def internals(f, x, head):
    """Reproduce the pieces provisional_values uses, without mutating state."""
    b = f._bits(x)
    hist = (*f.hist, b)
    rule = head.rule()
    idx, weight, p, atoms, p0, base, novelty = rule
    in0 = head.input(b, f.hist)[1:]
    r, prob, _, _ = head.likelihood(rule, in0[None, :])
    r = r[0]
    splits = head.signed_rules(rule) if hasattr(head, 'signed_rules') else []
    seen = head.phase(b, f.hist)
    if splits:
        _, _, out0, _, g0 = head.context_moment(splits, in0[None, :], seen[None, :], r[None, :], prob[None, :], atoms)
        atoms0 = out0[0]
    else:
        atoms0 = atoms
    bound, _, _ = head.provisional(rule)
    inc = np.maximum(r, (bound * seen).max(1))
    code = int(head.code(b))
    trans = head.all_transitions()
    states = np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0))
    return dict(code=code, known=weight.any(1).astype(int).tolist(),
                nov=np.round(novelty, 4).tolist() if hasattr(novelty, 'tolist') else [],
                r=np.round(r, 4).tolist(), bound_max=np.round(bound.max(1), 4).tolist(),
                seen=[int(s) for s in np.flatnonzero(seen)],
                inc_max=float(inc.max()), states=states.tolist(),
                trans=np.round(trans[:, code, :], 3).tolist(),
                slots=head.slots.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48')
    p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--steps', type=int, default=420)
    p.add_argument('--out', default=None)
    p.add_argument('--every', type=int, default=1)
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed)
    x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor
    head = f.reward
    rows = []
    for t in range(a.steps):
        pre = env.__dict__.copy() if hasattr(env, '__dict__') else {}
        info = internals(f, x, head) if (t % a.every == 0) else {}
        cue = float(x[3]) > .5
        trap = float(x[7]) > .5
        pos = int(np.argmax(x[:3]))
        q = None
        # counterfactual: reward of each action at this state (env is deterministic given action)
        act = int(brain.step(x))
        y, r, _, inf = env.step(act)
        brain.learn(y)
        rows.append(dict(t=t, pos=pos, cue=int(cue), trap=int(trap), act=act, rew=float(r),
                         shock=int(inf['received_shock']),
                         post=int(np.argmax(y[:3])), **info))
        x = y
    rec = dict(k=[a.agent, a.variant, a.seed, a.steps], m=dict(rows=rows))
    with open(a.out, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
