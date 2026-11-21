"""Immutable AR-5 batch entry point. Run through harness.bg and an external watchdog."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from eval.phenomena import run_group
from eval.predictive import run_one
from harness.journal import append
from models.factor import CANONICAL, make_factor
from models.kernels import copy_ancestor_rows, systematic_ancestry


def warm():
    """Fixed kernel signatures, independent of candidate and world timing."""
    idx = systematic_ancestry(np.array([.9, .1]), .31)
    for dtype in (np.float32, np.float64):
        copy_ancestor_rows(np.zeros((2, 2, 13), dtype), idx, np.array([1,2]))
    m = make_factor(CANONICAL, 5, 2, 4)
    m._reset_pos(0, 0)
    from models.kernels import binary_dependency_scores
    binary_dependency_scores(np.zeros((4, 1, 2, 1)))
    x = [0., 0., 1., 0., 0.]
    for k in range(8):
        y = [float(k % 2), 0., 1., 0., 0.]; m.predict(x, 0); m.learn(x, 0, y); x = y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['t1', 't2', 'cost'])
    p.add_argument('out'); p.add_argument('suite'); p.add_argument('tasks', nargs='*')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--particles', type=int, default=25)
    p.add_argument('--steps', type=int, default=600)
    a = p.parse_args()
    tree = os.environ['PG_TREE']
    model = CANONICAL if a.particles == 200 else f'{CANONICAL}-N{a.particles}'
    if a.mode == 'cost':
        t = time.monotonic(); warm(); print('warm seconds', time.monotonic() - t, flush=True)
    for task in a.tasks:
        start = time.monotonic()
        if a.mode == 't2':
            row, group = task.split(':')
            *_, value, trace = run_group(model, row, group, a.seed, a.suite, trace=True)
            m = dict(probe=value, trace=trace)
        else:
            _, _, m = run_one(task, [model, 'marginal'], a.seed, a.steps, a.suite)
            ls, lf, lm = (m[key]['ll'] for key in (model, 'filter', 'marginal'))
            m = dict(seconds=m[model]['sec'], gc=(lm-ls)/(lm-lf), ll=ls, oracle=lf, marginal=lm)
        append(a.out, dict(k=[tree, a.seed, f'{model}|{a.suite}|{task}|{a.steps}'], m=m,
                           dt=time.monotonic()-start))
        print(task, m if a.mode != 't2' else m['probe'], flush=True)


if __name__ == '__main__':
    from harness.beacon import cli

    with cli():
        main()
