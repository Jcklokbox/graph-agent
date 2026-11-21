"""AR-6 exposed-seed sweep: discovery statistics per episode (development only).

Per episode, records the causal chain a hypothesis about discovery needs:
when the first real payoff arrived, how many cue-without-lamp opportunities were
seen before/after it, how often the payoff action was actually taken in each
lamp condition, and the tail score. All seeds are already-exposed dev seeds.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from agents import agent_class, load
from envs.tasks import SPLITS, VARIANTS
from harness.journal import append


def one(job):
    model, variant, seed, steps = job
    eb, bb = SPLITS['dev'][2], SPLITS['dev'][3]
    v = VARIANTS[variant]
    env = v.make(eb + seed)
    brain = agent_class(load(model))(v.n_inputs, v.n_actions, 11, seed=bb + seed)
    x = env.reset()
    R, A, POS, CUE, LAMP = [], [], [], [], []
    for _ in range(steps):
        a = brain.step(x)
        y, r, _, _ = env.step(a)
        brain.learn(y)
        R.append(float(r)); A.append(int(a)); POS.append(int(np.argmax(x[:3])))
        CUE.append(int(x[3] > .5)); LAMP.append(int(x[7] > .5))
        x = y
    R = np.array(R); A = np.array(A); POS = np.array(POS); CUE = np.array(CUE); LAMP = np.array(LAMP)
    tail = float(R[-200:].mean())
    orc = v.oracle(); oenv = v.make(eb + seed); ox = oenv.reset(); OR = []
    for _ in range(steps):
        ox, orr, _, _ = oenv.step(int(orc(ox))); OR.append(float(orr))
    otail = float(np.mean(OR[-200:]))
    pay = np.flatnonzero(R == 1.)
    ok = (CUE == 1) & (LAMP == 0)          # payoff opportunity reaches a cue
    bad = (CUE == 1) & (LAMP == 1)
    ok_trials = np.flatnonzero(ok)
    first_pay = int(pay[0]) if len(pay) else None
    # opportunities available before the first payoff and left-choices there
    before = ok_trials[ok_trials < (first_pay if first_pay is not None else steps)]
    after = ok_trials[ok_trials >= (first_pay if first_pay is not None else steps)]
    m = dict(variant=variant, seed=seed, tail=tail, oracle_tail=otail,
             score=float((tail - .5) / (otail - .5)),
             payoffs=int(len(pay)), first_pay=first_pay,
             shocks=int((R == 0).sum()), late_shocks=float((R[-200:] == 0).mean()),
             ok_steps=int(ok.sum()), bad_steps=int(bad.sum()),
             ok_before=int(len(before)), ok_after=int(len(after)),
             cue_lamp_share=float(LAMP[CUE == 1].mean()) if (CUE == 1).any() else None,
             ok_share=float(ok.sum() / max(1, (CUE == 1).sum())),
             left_ok_before=float(A[before].mean() == 0) if len(before) else None,
             left_ok_after=float((A[after] == 0).mean()) if len(after) else None,
             left_bad=float((A[bad] == 0).mean()) if bad.any() else None,
             left_nocue=float((A[CUE == 0] == 0).mean()),
             ok_after_frac=float((A[after] == 0).mean()) if len(after) else None,
             act=np.bincount(A, minlength=v.n_actions).tolist(),
             sec=None)
    return [os.environ.get('PG_TREE', 'local'), seed,
            f'{model}|{variant}|sweep|{steps}'], m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('variants', nargs='+')
    p.add_argument('--model', default='v48')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    jobs = [(a.model, v, s, a.steps) for v in a.variants for s in a.seeds]
    import time
    from multiprocessing import get_context
    t0 = time.time()
    if a.workers <= 1:
        out = map(one, jobs)
        for key, m in out:
            append(a.out, dict(k=key, m=m))
            print(key[1:], {k: m[k] for k in ('score', 'payoffs', 'first_pay', 'ok_before',
                                              'left_ok_after', 'left_bad')}, flush=True)
    else:
        with get_context('fork').Pool(a.workers) as pool:
            for key, m in pool.imap_unordered(one, jobs, chunksize=1):
                append(a.out, dict(k=key, m=m))
                print(key[1:], {k: m[k] for k in ('score', 'payoffs', 'first_pay', 'ok_before',
                                                  'left_ok_after', 'left_bad')}, flush=True)
    print('elapsed', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    from harness.beacon import cli

    with cli():
        main()
