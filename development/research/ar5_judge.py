"""Frozen AR-5 predictor judge: explicit slices, expected keys, no repeated looks.

Launch each batch under harness.bg with timeout <= 169s. Slice offsets are
execution details and never change the required full sample or the verdict.
Logs and manifest live outside the read-only development snapshot.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

from envs.family import WORLD_SETS
from envs.phenomena import SETS
from eval.phenomena import run_group
from eval.predictive import run_one
from harness.journal import append, load
from models.factor import CANONICAL
from research.ar5_holdout import ROWS, SEEDS, FLOOR

T1 = {'v1': (2000, .80), 'd2': (4000, .89), 'd3': (4000, .95)}
T2 = {'A': 'v1', 'h1': 'h1'}
GUARDS = ('C','A','h1','v1','d2','d3')
NEXT = 'holdout-next'
T975 = {5: 2.570582, 15: 2.131450, 23: 2.068659}


def plan(suite):
    if suite == 'C':
        return [(0, w) for w in ('s-2x3', 's-4x3')]
    if suite in T1:
        return [(seed, w) for seed in range(6) for w in WORLD_SETS[suite]()]
    if suite in T2:
        return [(seed, f'{row}|{g}') for seed in range(16)
                for row, p in SETS[T2[suite]][0].items() for g in p.groups]
    if suite == 'holdout':
        return [(seed, f'{row}|{g}') for seed in SEEDS for row, p in ROWS.items() for g in p.groups]
    if suite == NEXT:
        from research.ar5_holdout_next import SEEDS as new, ROWS as tests
        return [(seed, f'{row}|{g}') for seed in new for row, p in tests.items() for g in p.groups]
    if suite == 'credit':
        return [(seed,f'{row}|{group}') for seed in range(74700,74706)
                for row in ('signed','background') for group in (0,1)]
    raise ValueError(suite)


def key(tree, suite, task):
    return tree, task[0], f'{CANONICAL}|{suite}|{task[1]}'


def t_interval(vals):
    if len(vals) - 1 not in T975:
        raise ValueError('interval needs all registered seeds')
    mu, sd = statistics.mean(vals), statistics.stdev(vals)
    half = sd / math.sqrt(len(vals)) * T975[len(vals) - 1]
    return dict(mean=mu, sd=sd, lo=mu-half, hi=mu+half, n=len(vals))


def episode(tree, suite, task):
    seed, slug = task
    t0 = time.monotonic()
    if suite in T1 or suite == 'C':
        steps, wset = (600, 'dscale') if suite == 'C' else (T1[suite][0], suite)
        _, _, m = run_one(slug, [CANONICAL, 'marginal'], seed, steps, wset)
        ml, ol, nl = (m[n]['ll'] for n in (CANONICAL, 'filter', 'marginal'))
        out = dict(sec=m[CANONICAL]['sec'], ll=ml, oracle=ol, null=nl, gap=nl-ol,
                   gc=(nl-ml)/(nl-ol))
    elif suite in T2:
        row, group = slug.split('|')
        out = run_group(CANONICAL, row, group, seed, T2[suite])[-1]
    elif suite == NEXT:
        from research.ar5_next_reference import episode as next_episode
        row, group = slug.split('|')
        out = next_episode(CANONICAL,row,group,seed)
    elif suite == 'credit':
        from eval.scripts.ar5_credit_probe import episode as controlled_episode
        row, group = slug.split('|')
        out = controlled_episode(row,bool(int(group)),seed,200,True)['4']
    else:
        from research.ar5_screen import episode as holdout_episode
        row, group = slug.split('|')
        out = holdout_episode(CANONICAL, (row, group, seed))['m']
    return dict(k=list(key(tree, suite, task)), m=out, dt=time.monotonic()-t0)


def evidence(path, tree, suite):
    want = {key(tree, suite, t) for t in plan(suite)}
    recs, dup, bad = load(path / (suite + '.jsonl'))
    if dup or bad or not recs.keys() <= want:
        raise ValueError(f'{suite}: incompatible journal: duplicate={dup} malformed={bad} unknown={len(recs.keys()-want)}')
    return recs, want


def report(root, tree, suite):
    recs, expected = evidence(root, tree, suite)
    if recs.keys() != expected:
        return dict(status='INCOMPLETE', tuples=len(recs), expected=len(expected))
    got = {(s, t.split('|', 2)[2]): r['m'] for (_,s,t),r in recs.items()}
    if suite == 'C':
        pair = {w: got[0,w]['sec'] for w in ('s-2x3','s-4x3')}
        c = pair['s-4x3'] / pair['s-2x3']
        return dict(status='PASS' if c <= 2.5 else 'FAIL', estimate=c, pair=pair, tuples=len(recs))
    if suite in T1:
        worlds = list(WORLD_SETS[suite]())
        per_seed = [statistics.mean(got[s,w]['gc'] for w in worlds) for s in range(6)]
        ci = t_interval(per_seed)
        by_world = {w: {f:statistics.mean(got[s,w][f] for s in range(6))
                        for f in ('ll','oracle','null','gap','gc')} for w in worlds}
        return dict(status='PASS' if ci['mean'] >= T1[suite][1] else 'FAIL', **ci,
                    threshold=T1[suite][1], by_world=by_world, tuples=len(recs))
    if suite == NEXT:
        from research.ar5_next_reference import next_verdict
        ans = next_verdict([(seed,*slug.split('|'),m) for (seed,slug),m in got.items()])
        return dict(**ans,tuples=len(recs))
    if suite == 'credit':
        from research.ar5_credit_report import estimates_for
        s,b = ([{bool(i):got[seed,f'{row}|{i}'] for i in (0,1)} for seed in range(74700,74706)]
               for row in ('signed','background'))
        estimates, ok = estimates_for(s,b)
        return dict(status='PASS' if ok else 'FAIL',estimates=estimates,tuples=len(recs))
    rows, floor = (ROWS, FLOOR) if suite == 'holdout' else (SETS[T2[suite]][0], SETS[T2[suite]][2])
    seeds = SEEDS if suite == 'holdout' else range(16)
    out = {}
    for nm, p in rows.items():
        val = [p.effect({g:got[s,f'{nm}|{g}'] for g in p.groups}) for s in seeds]
        ci = t_interval(val)
        th = floor if p.min_effect is None else p.min_effect
        verdict = ('PASS' if ci['lo'] > 0 and ci['mean'] >= th else
                   'REV' if ci['hi'] < 0 and ci['mean'] <= -th else 'FAIL')
        out[nm] = dict(status=verdict, **ci, threshold=th)
    hit = {nm for nm,d in out.items() if d['status'] == 'PASS'}
    rev = {nm for nm,d in out.items() if d['status'] == 'REV'}
    if suite == 'A':
        ok = len(hit) >= 9 and not rev and {'pree','cond-inhibition','reinstatement'} <= hit
    elif suite == 'h1':
        ok = len(hit) >= 5 and not rev
    else:
        ok = len(hit) >= 3 and not rev and 'inhibitor-credit' in hit and bool(hit & {'background-credit','return-specificity'})
        # The training contrast is separately checked before a mediation or
        # signed credit claim. Do not silently drop data on an invalid control.
        dr = [(.5 * (got[s,'associate-crossover|linked']['B'] - got[s,'associate-crossover|linked']['X']
               + got[s,'associate-crossover|alternate']['X'] - got[s,'associate-crossover|alternate']['B'])) for s in seeds]
        sr = [got[s,'inhibitor-credit|control']['BY'] - got[s,'inhibitor-credit|control']['BX'] for s in seeds]
        out['controls'] = dict(direct=t_interval(dr), summation=t_interval(sr))
        ok = ok and out['controls']['summation']['lo'] > 0
        if 'associate-crossover' in hit:
            ok = ok and out['controls']['direct']['lo'] > 0
    return dict(status='PASS' if ok else 'FAIL', pass_rows=sorted(hit), rev_rows=sorted(rev),
                rows=out, tuples=len(recs))


def run_batch(root, freeze, suite, start, stop, workers):
    tree = os.environ['PG_TREE']
    if tree != freeze['tree'] or not freeze.get('predictor_development_closed'):
        raise ValueError('judge requires closed development and matching snapshot')
    held = freeze.get('holdout_suite','holdout')
    if held == NEXT:
        from research.ar5_holdout_next import SEEDS as fixed_seeds
    else:
        fixed_seeds = SEEDS
    if freeze['seeds'] != list(fixed_seeds):
        raise ValueError('holdout seed manifest mismatch')
    if suite in ('holdout',NEXT):
        if suite != held:
            raise RuntimeError('holdout registry not authorized by freeze')
        for guard in GUARDS + (('credit',) if suite == NEXT else ()):
            if report(root, tree, guard)['status'] != 'PASS':
                raise RuntimeError(f'holdout locked by {guard}')
    tasks = plan(suite)
    if not 0 <= start < stop <= len(tasks):
        raise ValueError(f'invalid offset slice; suite has {len(tasks)} tasks')
    recs, _ = evidence(root, tree, suite)
    todo = [task for task in tasks[start:stop] if key(tree,suite,task) not in recs]
    from eval.scripts.ar5_probe import warm
    warm()
    fn = partial(episode,tree,suite)
    path = root / (suite + '.jsonl')
    if workers == 1:
        for task in todo:
            append(path, fn(task))
    else:
        with mp.get_context('fork').Pool(workers) as pool:
            for x in pool.imap_unordered(fn, todo, chunksize=1):
                append(path,x)
    print(json.dumps(dict(tree=tree,suite=suite,start=start,stop=stop,new=len(todo))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan','report','batch'])
    p.add_argument('suite', choices=[*GUARDS,'credit','holdout',NEXT])
    p.add_argument('root',type=Path); p.add_argument('start',type=int,nargs='?',default=0)
    p.add_argument('stop',type=int,nargs='?'); p.add_argument('--workers',type=int,choices=[1,2],default=1)
    a=p.parse_args()
    if a.command == 'plan':
        print(json.dumps(plan(a.suite))); return
    manifest = a.root / 'freeze.json'
    freeze = json.loads(manifest.read_text())
    if a.command == 'report':
        print(json.dumps(report(a.root, freeze['tree'], a.suite), indent=2, sort_keys=True)); return
    if a.suite == 'C' and a.workers != 1:
        p.error('C must run on one core')
    run_batch(a.root, freeze, a.suite, a.start, len(plan(a.suite)) if a.stop is None else a.stop, a.workers)


if __name__ == '__main__':
    main()
