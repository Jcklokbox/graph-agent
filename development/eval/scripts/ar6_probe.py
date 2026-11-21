"""AR-6 exposed-seed probe: separate payoff discovery from post-exposure reuse.

Arms
----
free     : the learner acts from step 0 (discovery timing as deployed).
oracle   : a scripted *task-oracle* prefix of ``--prefix`` steps, then free.
stay     : a scripted always-stay prefix of the same length (null yoke).
matched  : a scripted prefix that reproduces the oracle's *control-state and
           cue/lamp exposure* (position 0 exactly on the same steps) but draws
           its action on the rewarded steps from a coin flip, so the payoff is
           decorrelated from the cue (adverse control).
offgoal  : the oracle's cue/lamp entry timing but always moves to position 2
           (a matched-opportunity, wrong-actuator yoke).

Nothing here is injected into the learner beyond ordinary observations and the
reward the environment returns for the action actually taken. Scripted arms are
diagnostic fixtures, never qualification.

Per-episode summary measures
    payoffs        : number of steps with reward 1.0 (real payoff reached)
    shocks         : number of steps with reward 0.0
    tail           : mean reward over the last 200 steps
    score          : (tail - stay)/(oracle - stay)
    left_ok        : fraction of steps with cue & no lamp where action 0
    left_bad       : fraction of steps with cue & lamp where action 0
    left_nocue     : fraction of steps with no cue where action 0
    act            : action counts
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from agents import agent_class, load
from envs.tasks import SPLITS, VARIANTS
from harness.journal import append


def scripted(kind, x, rng):
    cue = x[3] > .5
    lamp = x[7] > .5
    pos = int(np.argmax(x[:3]))
    if kind == 'oracle':
        if pos == 0:
            return 2
        if pos == 2:
            return 0
        return 0 if (cue and not lamp) else 1
    if kind == 'stay':
        return 1
    if kind == 'matched':
        # Reproduce the oracle's entry into position 0 on exactly the same
        # steps, but on the rewarded step substitute the stay action with
        # probability 1/2 so the payoff becomes cue-independent.
        if pos == 0:
            return 2
        if pos == 2:
            return 0
        if cue and not lamp and rng.random() < .5:
            return 1
        return 0 if (cue and not lamp) else 1
    if kind == 'offgoal':
        if pos == 2:
            return 1
        return 2
    raise ValueError(kind)


def episode(model, variant, env_seed, brain_seed, prefix, kind, steps, trace=False):
    v = VARIANTS[variant]
    env = v.make(env_seed)
    brain = agent_class(load(model))(v.n_inputs, v.n_actions, 11, seed=brain_seed)
    f = brain.factor
    rng = np.random.default_rng(90210 + brain_seed)
    x = env.reset()
    R, A = [], []
    left_ok = left_bad = left_nocue = 0
    n_ok = n_bad = n_nocue = 0
    row = []
    qinfo = []
    for t in range(steps):
        cue, lamp = x[3] > .5, x[7] > .5
        pos = int(np.argmax(x[:3]))
        if t < prefix:
            a = scripted(kind, x, rng)
        else:
            a = brain.step(x)
        y, r, _, _ = env.step(a)
        if t >= prefix:
            brain.learn(y)
        else:
            # The learner still observes the transition; it just does not choose.
            f.learn(x, a, y)
        if cue and not lamp:
            n_ok += 1
            left_ok += int(a == 0)
        elif cue and lamp:
            n_bad += 1
            left_bad += int(a == 0)
        else:
            n_nocue += 1
            left_nocue += int(a == 0)
        R.append(float(r))
        A.append(int(a))
        if trace:
            row.append([t, [int(z) for z in x[:8]], int(a), float(r),
                        int(pos), [int(z) for z in y[:8]]])
        x = y
    tail = float(np.mean(R[-200:]))
    stay = .5
    orc = v.oracle()
    oenv = v.make(env_seed)
    ox = oenv.reset()
    OR = []
    for _ in range(steps):
        ox, orr, _, _ = oenv.step(int(orc(ox)))
        OR.append(float(orr))
    otail = float(np.mean(OR[-200:]))
    out = dict(prefix=prefix, arm=kind, variant=variant, seed=brain_seed,
               payoffs=int(np.sum(np.array(R) == 1.)), shocks=int(np.sum(np.array(R) == 0.)),
               tail=tail, oracle_tail=otail, stay_tail=stay,
               score=float((tail - stay) / (otail - stay)) if otail != stay else float('nan'),
               score_all=float((np.mean(R) - stay) / (otail - stay)) if otail != stay else float('nan'),
               left_ok=left_ok / max(1, n_ok), left_bad=left_bad / max(1, n_bad),
               left_nocue=left_nocue / max(1, n_nocue),
               n_ok=n_ok, n_bad=n_bad, n_nocue=n_nocue,
               act=np.bincount(A, minlength=v.n_actions).tolist(),
               sec=None)
    if trace:
        head = f.reward
        out['slots'] = head.slots.tolist()
        out['visits'] = head.control_mass.tolist()
        out['rule_idx'] = head.rule()[0].tolist() if hasattr(head, 'rule') else None
        out['rule_w'] = head.rule()[1].astype(int).tolist() if hasattr(head, 'rule') else None
        out['anchors'] = head.anchors.tolist() if hasattr(head, 'anchors') else None
        out['assoc'] = np.round(head.assoc, 3).tolist() if hasattr(head, 'anchors') else None
        splits = head.signed_rules(head.rule()) if hasattr(head, 'signed_rules') else []
        out['splits'] = [[int(j), int(m), int(l), int(s.sum()), float(g)]
                         for j, m, l, s, g, d in splits]
        out['ctx_n_sum'] = (head.ctx_n.sum(axis=(0, 1, 2, 4)).tolist()
                            if hasattr(head, 'ctx_n') else None)
        out['onset_total'] = head.onset_total.tolist()
        out['present'] = head.present.tolist()
        out['trace'] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('variant', nargs='+')
    p.add_argument('--model', default='v48')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--prefix', type=int, nargs='+', default=[0])
    p.add_argument('--arm', nargs='+', default=['free'])
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--trace', action='store_true')
    a = p.parse_args()
    tree = os.environ.get('PG_TREE', 'local')
    eb, bb = SPLITS['dev'][2], SPLITS['dev'][3]
    import time
    for variant in a.variant:
        for seed in a.seeds:
            for prefix in a.prefix:
                for arm in a.arm:
                    if prefix == 0 and arm != 'free':
                        continue
                    if prefix > 0 and arm == 'free':
                        continue
                    t0 = time.time()
                    rec = episode(a.model, variant, eb + seed, bb + seed, prefix, arm,
                                  a.steps, trace=a.trace)
                    rec['sec'] = round(time.time() - t0, 2)
                    key = [tree, seed, f'{a.model}|{variant}|{arm}|{prefix}|{a.steps}']
                    append(a.out, dict(k=key, m=rec))
                    short = {k: rec[k] for k in ('payoffs', 'shocks', 'tail', 'score',
                                                 'left_ok', 'left_bad', 'left_nocue', 'act', 'sec')}
                    print(variant, seed, arm, prefix, short, flush=True)


if __name__ == '__main__':
    from harness.beacon import cli

    with cli():
        main()
