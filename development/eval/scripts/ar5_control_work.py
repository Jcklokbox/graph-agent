"""Pinned warm N200 full act+learn work and throughput for factor-only agents."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from agents import agent_class, load
from envs.family import WORLD_SETS, WorldEnv
from eval.scripts.ar5_probe import warm
from harness.journal import append


def measure(model, world, steps, seed, dense):
    w=WORLD_SETS['dscale']()[world]; env=WorldEnv(w,seed+1000)
    a=agent_class(load(model))(w.n_channels,w.n_actions,w.value_index,seed=seed)
    f=a.factor
    if f.N!=200:
        raise ValueError('controller cost not measured at accepted N200 priority')
    x=env.reset(); rng=np.random.default_rng(seed+2000); lat=[]; learn=[]; t0=time.monotonic()
    for _ in range(steps):
        t=time.process_time_ns(); move=a.step(x); u=time.process_time_ns()
        if dense:
            y=(rng.random(w.n_channels)<.65).astype(float);y[w.value_index]=rng.random()
        else:
            y=env.step(move)
        v=time.process_time_ns();a.learn(y); q=time.process_time_ns()
        lat.append((u-t)*1e-9);learn.append((q-v)*1e-9);x=y
    l=np.add(lat,learn)
    def bytes_of(o):
        return sum(v.nbytes for v in vars(o).values() if isinstance(v,np.ndarray))
    return dict(n=f.N, channels=f.J,steps=steps,wall=time.monotonic()-t0, sec=float(l.sum()),
                ms_plan=1000*float(np.mean(lat)),ms_learn=1000*float(np.mean(learn)),
                ms_p99=1000*float(np.quantile(l,.99)),factor_bytes=bytes_of(f),
                reward_bytes=bytes_of(f.reward), discovery_edges=f.discovery.source.size,
                groups=f.proc.partition())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--model',required=True);p.add_argument('--worlds',nargs='+',default=['s-2x3','s-4x3'])
    p.add_argument('--steps',type=int,default=400);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dense',action='store_true');a=p.parse_args()
    warm();measure(a.model,a.worlds[0],8,a.seed,a.dense)
    for world in a.worlds:
        r=measure(a.model,world,a.steps,a.seed,a.dense)
        append(a.out,dict(k=[os.environ['PG_TREE'],a.seed,f'{a.model}|{world}|{"dense" if a.dense else "process"}|{a.steps}'],m=r))
        print(world,r,flush=True)


if __name__=='__main__':main()
