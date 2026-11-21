"""Non-gating work stress with a controlled scope active on matched real tapes."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from agents.v58 import Agent
from envs.tasks import VARIANTS
from eval.scripts.ar5_probe import warm
from harness.beacon import cli
from harness.journal import append


def measure(seed,pad,warmup,steps):
    v=VARIANTS['canc-noise3'];e=v.make(seed);x=e.reset();a=Agent(v.n_inputs+pad,3,11,seed=1000+seed)
    rng=np.random.default_rng(2000+seed);counts=[];plan=[];learn=[];digest=hashlib.sha256()
    cue=-9;escape=0;events=0;fracs=[]
    def widen(z):return np.r_[z,rng.random(pad)<.5] if pad else z
    z=widen(x)
    for t in range(warmup+steps):
        # Shared off-cue entry, dwell and exit; actual valid cues also earn
        # genuine reward. The candidate sees observations, not this schedule.
        if x[3]:
            events+=1;cue=t
            if t<warmup//2:
                move=1;escape=0 if events%2 else 2;script={1:escape,2:1,3:2-escape}
            else:
                move=1 if x[7] else 0
                script=({1:0,2:1,3:2} if x[7] and events%2 else {1:2-move})
        else:move=script.get(t-cue,1) if cue>=0 else 1
        start=time.process_time_ns();guess=a.step(z);middle=time.process_time_ns()
        if guess!=move:a.factor.predict(z,move);a.pending=(z,move)
        y=e.step(move)[0];p=widen(y);a.learn(p);end=time.process_time_ns()
        digest.update(y.tobytes()+bytes((move,)))
        if t>=warmup:
            plan.append((middle-start)*1e-9);learn.append((end-middle)*1e-9)
            scopes=set(a.factor.reward.slots.tolist());counts.append(len(scopes-{-1}));fracs.append({0,1,2}<=scopes)
        x=y;z=p
    return dict(n=a.factor.N,channels=a.factor.J,steps=steps,script_sha256=digest.hexdigest(),
                active_fraction=sum(fracs)/steps,slots=sorted(a.factor.reward.slots.tolist()),
                mean_slots=float(np.mean(counts)),plan_cpu=sum(plan),learn_cpu=sum(learn),
                p99_ms=1e3*float(np.quantile(np.add(plan,learn),.99)),total_cpu=sum(plan)+sum(learn))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[1006,1007,1012]);a=p.parse_args();warm()
    for seed in a.seeds:
        low=measure(seed,0,120,360);high=measure(seed,14,120,360)
        if low['script_sha256']!=high['script_sha256']:raise ValueError('physical replay changed across widths')
        report=dict(low=low,high=high,cpu_ratio=high['total_cpu']/low['total_cpu'])
        append(a.out,dict(k=[os.environ['PG_TREE'],seed,'v58|active-work'],m=report));print(seed,report,flush=True)


if __name__=='__main__':
    with cli():main()
