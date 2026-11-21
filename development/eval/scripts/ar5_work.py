"""Operation footprints and warm model-only latency, without constructing the joint oracle."""
from __future__ import annotations
import argparse
import os
import time
import numpy as np
from envs.family import WORLD_SETS, WorldEnv
from eval.scripts.ar5_probe import warm
from harness.journal import append
from models.factor import CANONICAL, make_factor


def measure(world, seed, steps, dense=False):
    w = WORLD_SETS['dscale']()[world]
    rng = np.random.default_rng(2000+seed)
    env = WorldEnv(w, 1000+seed)
    m = make_factor(CANONICAL, w.n_channels, w.n_actions, w.value_index, seed)
    x = env.reset()
    per_step=[]; used=0; proposals=0
    for _ in range(steps):
        a = int(rng.integers(w.n_actions))
        if dense:
            y = (rng.random(w.n_channels) < .65).astype(float); y[w.value_index] = 0.
        else:
            y = env.step(a)
        t = time.process_time_ns(); q=m.predict(x,a); m.learn(x,a,y)
        per_step.append(time.process_time_ns()-t)
        if not np.isfinite(q).all():
            raise AssertionError('nonfinite prediction')
        used = max(used, int(m.used.max())); proposals=max(proposals,len(m.proc.cand))
        x=y
    def footprint(obj):
        return sum(x.nbytes for x in vars(obj).values() if isinstance(x,np.ndarray))
    s=np.asarray(per_step)*1e-9
    return dict(channels=m.J, steps=steps, elapsed=float(s.sum()), q90=float(np.quantile(s,.9)),
                q99=float(np.quantile(s,.99)), median=float(np.median(s)), cap_places=m.K,
                factor_bank_bytes=footprint(m), structural_bank_bytes=footprint(m.proc),
                factor_edges=m.discovery.source.size, structure_edges=m.proc.discovery.source.size,
                max_live_regimes=used,max_open_proposals=proposals,groups=m.proc.partition())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out'); p.add_argument('world',nargs='+');p.add_argument('--dense',action='store_true')
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();warm(); tree=os.environ['PG_TREE']
    for w in a.world:
        r=measure(w,a.seed,a.steps,a.dense)
        append(a.out,dict(k=[tree,a.seed,f'{w}|{"dense" if a.dense else "process"}|{a.steps}'],m=r,dt=r['elapsed']))
        print(w,r,flush=True)


if __name__=='__main__':main()
