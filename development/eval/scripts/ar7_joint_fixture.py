"""Replay known interventions and compare the deployed head to the AR6 oracle."""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from agents.v60 import JointReward
from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build, run
from eval.scripts.ar6_parent_projection import projected
from models.factor_rollout import provisional_values


@contextmanager
def upgraded(f):
    incumbent=f.reward
    h=JointReward(incumbent.J,incumbent.A,incumbent.L-1)
    h.__dict__.update(incumbent.__dict__)
    f.reward=h
    try:
        h.refresh_joint()
        yield h
    finally:f.reward=incumbent


def compare(f,c,x,u):
    ref=projected(f,c,x,u,True)
    with upgraded(f) as h:
        actual={str(j):h.joint_key[j,:3].tolist() for j in np.flatnonzero(h.joint_edge)}
        assert actual=={j:k[:3] for j,k in ref['choices'].items()}, (actual,ref['choices'])
        before=pickle.dumps(f);q=provisional_values(f,x,u,False)
        assert before==pickle.dumps(f)
        error=float(np.max(np.abs(q-ref['q'])))
        np.testing.assert_allclose(q,ref['q'],rtol=0,atol=4e-15)
        return dict(error=error,selected=actual,delta=float(q[0]-q[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args();out=[]
    for variant in ('canc','canc-noise3'):
        v=VARIANTS[variant]
        for seed in (6,15):
            for arm in ('same','opposite','neutral'):
                table=build(v,seed,4,16,5);result=run(v,seed,arm,table,5,compare)
                rows=[dict(cancelled=row['cancelled'],**row['structure']) for row in result['results']]
                out.append(dict(variant=variant,seed=seed,arm=arm,probes=rows));print(variant,seed,arm,len(rows),flush=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
