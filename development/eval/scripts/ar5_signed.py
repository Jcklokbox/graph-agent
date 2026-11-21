"""Development-only intervention on a learned signed cue; unrelated to the locked holdout."""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
from envs.phenomena import Schedule, A, B, X, Y, US, VALUE, CTX1, HOME, N_CH_V1
from harness.journal import append
from models.factor import CANONICAL, make_factor


def run(seed, inhibited, particles):
    s = Schedule(seed, N_CH_V1)
    s.gap(60)
    # Independently calibrated excitor, inhibitor and irrelevant familiar cue.
    s.mixed([([A], .85), ([A, X], .15), ([A, Y], .85), ([X], 0.), ([Y], 0.)], 18)
    b = lambda cues: tuple(float(i in (*cues, CTX1)) for i in range(N_CH_V1))
    pre = len(s.obs) - 1
    # Counterbalanced probe in a crossover experiment. The new B is present
    # only in these matched training compounds; equal outcomes and exposures.
    s.trials([B, X if inhibited else Y], .8, 12)
    s.gap(10)
    m = make_factor(CANONICAL if particles == 200 else f'{CANONICAL}-N{particles}',
                    N_CH_V1, 2, VALUE, seed)
    res = {}
    j = m.ch.index(US)
    for t, (x, y) in enumerate(zip(s.obs, s.obs[1:])):
        m.learn(x, 0, y)
        if t+1 == pre or t+1 == len(s.obs)-1:
            label = 'before' if t+1 == pre else 'after'
            avail = {slot[0]:k + len(m.own_lags) + m.A for k,slot in enumerate(m.slots[j]) if slot is not None}
            res[label] = {key: float(m.predict(b(cues), 0, hypothetical=True)[US])
                         for cues, key in (([A,X], 'AX'), ([A,Y], 'AY'), ([B], 'B'), ([B,X], 'BX'), ([B,Y], 'BY'))}
            res[label]['shared'] = {str(src): float(m.shared_mu[j, p]) for src,p in avail.items()}
            res[label]['mixture'] = float(m.expert[j])
            res[label]['trace'] = m.trace()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out'); p.add_argument('--seeds', type=int, default=3); p.add_argument('--particles', type=int, default=25)
    a=p.parse_args()
    for i in range(a.seeds):
        seed = 73000 + i
        for h in (False, True):
            m = run(seed,h,a.particles)
            append(a.out, {'k':[os.environ['PG_TREE'],seed,f'signed|{int(h)}|N{a.particles}'], 'm':m})
            print(seed,h, {k: v['B'] for k,v in m.items()}, flush=True)


if __name__ == '__main__':
    from harness.beacon import cli

    with cli():
        main()
