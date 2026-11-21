"""AR6-01 exposed intervention: payoffs and controls on the same exogenous tape."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from agents import agent_class, load
from envs.tasks import SPLITS, VARIANTS
from harness.beacon import cli
from harness.journal import append, load as load_journal
from models.factor_rollout import provisional_values

ARMS = ('free', 'reward', 'bad', 'right')


def tapes(make, seed, exposures):
    """Intervention schedules computed by the evaluator, never given to the learner."""
    env = make(seed)
    x = env.reset()
    good, bad = [], []
    for t in range(180):
        if x[3]:
            (bad if x[7] else good).append(t)
        if len(good) >= exposures and len(bad) >= exposures:
            break
        x, *_ = env.step(1)
    else:
        raise ValueError('failed to find matched exposed events')
    good, bad = good[:exposures], bad[:exposures]
    prefix = max(good[-1], bad[-1])+2  # allow the final return to centre
    return int(prefix), {
        'free': None,
        'reward': {j: a for t in good for j,a in [(t,0),(t+1,2)]},
        'bad': {j: a for t in bad for j,a in [(t,0),(t+1,2)]},
        'right': {j: a for t in good for j,a in [(t,2),(t+1,0)]},
    }


def one(variant, seed, model, arm, steps, prefix, schedule):
    v = VARIANTS[variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb+seed)
    agent = agent_class(load(model))(v.n_inputs,v.n_actions,11,seed=bb+seed)
    f = agent.factor
    head = f.reward
    planner = f.two_step_values
    diag = {}

    def capture(x,u,theta=None):
        diag['q'], diag['u'] = planner(x,u,theta), u
        return diag['q']

    f.two_step_values = capture
    x = env.reset()
    pre = Counter()
    post = Counter()
    bins = Counter()
    decisions = []
    slots = []
    first = None
    checksum = []
    for t in range(steps):
        a = agent.step(x)
        if schedule is not None and t < prefix:
            actual = schedule.get(t, 1)
            if actual != a:
                f.predict(x, actual)  # discard unused action memo and prepare the actual transition
                agent.pending = (x, actual)
            a = actual
        y, reward, _, _ = env.step(a)
        if reward == 1 and first is None:
            first = t
        if t < prefix:
            pre[str(reward)] += 1
            checksum.append([int(x[3]),int(x[7]),*map(int,x[12:])])
        else:
            post[str(reward)] += 1
            cue, trap, pos = bool(x[3]), bool(x[7]), int(np.argmax(x[:3]))
            tag = 'bad' if cue and trap else 'valid' if cue else 'other'
            bins[f'{tag}:{pos}:{int(np.argmax(y[:3]))}'] += 1
            if cue and len(decisions)<20:
                rule = head.rule()
                c = int(head.code(f._bits(y)))
                decisions.append(dict(t=t,tag=tag,pos=pos,a=a,r=reward,
                    q=diag['q'].tolist(), fixed_q=provisional_values(f,x,diag['u'],False).tolist(),
                    scope=head.slots.tolist(), landed=c, positive=int(head.type_count[c,-1]),
                    parent=list(map(int,rule[0][c][rule[1][c]])),
                    evidence=float(head.evidence_type[c,3])))
        last_scope = head.slots.copy()
        agent.learn(y)
        if (last_scope!=head.slots).any():
            slots.append([t,head.slots.tolist()])
        x=y
    return dict(prefix=prefix, pre=dict(pre), post=dict(post), first=first, bins=dict(bins),
                transitions=slots, decisions=decisions, end_scope=head.slots.tolist(),
                end_positive=head.type_count[:,-1].tolist(), tape=checksum)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('variant',choices=('canc','canc-noise3'))
    p.add_argument('--seeds',type=int,nargs='+',default=[0,6,13])
    p.add_argument('--steps',type=int,default=420)
    p.add_argument('--exposures',type=int,default=4)
    p.add_argument('--model',default='v48')
    a=p.parse_args()
    if not 1 <= a.exposures <= 16:
        p.error('exposures must be in 1..16')
    seen,dup,bad=load_journal(a.out)
    if dup or bad:
        raise ValueError('journal invalid')
    _,_,eb,_=SPLITS['dev']
    for seed in a.seeds:
        prefix,schedules=tapes(VARIANTS[a.variant].make,eb+seed,a.exposures)
        if prefix>=a.steps:
            p.error('no post-prefix steps')
        ref=None
        for arm in ARMS:
            key=(os.environ['PG_TREE'],seed,f'{a.model}|{a.variant}|{arm}|{a.exposures}|{a.steps}')
            result=seen[key]['m'] if key in seen else one(a.variant,seed,a.model,arm,a.steps,prefix,schedules[arm])
            if ref is None:
                ref=result['tape']
            if result['tape']!=ref or (arm=='reward' and result['pre'].get('1.0')!=a.exposures) or (
                arm in ('bad','right') and result['pre'].get('0.0')!=a.exposures):
                raise ValueError('prefix did not preserve the intended intervention')
            if key not in seen:
                append(a.out,dict(k=key,m=result))
            print(seed, arm, result['first'], result['post'], flush=True)


if __name__=='__main__':
    with cli():
        main()
