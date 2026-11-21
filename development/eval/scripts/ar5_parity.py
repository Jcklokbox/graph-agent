"""Paired deterministic playback for D-50 bank resets; not an acceptance test."""
import argparse
import os
from pathlib import Path

import numpy as np
from envs.family import WORLD_SETS, WorldEnv
from envs.phenomena import SETS, VALUE
from models.factor import CANONICAL, make_factor


def playback(name, out, n=200):
    records=[]
    if name == 'sticky':
        w=WORLD_SETS['dscale']()['s-4x3']
        env=WorldEnv(w,1000)
        x=env.reset(); rng=np.random.default_rng(2000)
        m=make_factor(f'{CANONICAL}-N{n}',w.n_channels,w.n_actions,w.value_index)
        for _ in range(320):
            a=int(rng.integers(w.n_actions)); records.append(m.predict(x,a).copy())
            y=env.step(a); m.learn(x,a,y); x=y
    else:
        table, channels, _ = SETS['v1']
        p=table[name]; s=p.build(p.groups[0],0)
        m=make_factor(f'{CANONICAL}-N{n}',channels,2,VALUE)
        for x,y in zip(s.obs,s.obs[1:]):
            records.append(m.predict(x,0).copy()); m.learn(x,0,y)
    path=Path(out)/f'{os.environ["PG_TREE"]}-{name}.npy'
    np.save(path,np.array(records))


if __name__=='__main__':
    from harness.beacon import cli

    with cli():
        p=argparse.ArgumentParser(description=__doc__); p.add_argument('out')
        a=p.parse_args()
        for nm in ('sticky','savings'):
            playback(nm,a.out)
