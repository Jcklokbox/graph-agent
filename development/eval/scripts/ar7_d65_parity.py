"""Emit raw float bytes from the D65 sensory path or active controller on fixed tapes.

To execute the reference, run this file by absolute path with PYTHONPATH pointing
only to the source tree reconstructed from the D65 freeze, plus dependencies.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from envs.family import WORLD_SETS, WorldEnv
from envs.tasks import VARIANTS
from models.factor import CANONICAL, make_factor


class NoLegacy:
    def find_spec(self, name, *args):
        if name in {'agents.v11','agents.v12','models.context_tree'}:
            raise AssertionError(f'legacy import: {name}')


def play(model, tape):
    if tape=='sticky':
        w=WORLD_SETS['dscale']()['s-4x3'];env=WorldEnv(w,1000)
        n,a,v=w.n_channels,w.n_actions,w.value_index
        advance=env.step
    else:
        variant=VARIANTS[tape];env=variant.make(12713)
        n,a,v=variant.n_inputs,variant.n_actions,11
        def advance(move):return env.step(move)[0]
    x=env.reset(); agent=None
    if model=='d65':f=make_factor(CANONICAL,n,a,v,113)
    else:
        agent=importlib.import_module('agents.'+model).Agent(n,a,v,seed=113);f=agent.factor
        assert f.N==200
    tape_rng=np.random.default_rng(7671)
    out=[];inputs=hashlib.sha256();plans=0
    for k in range(320):
        move=int(tape_rng.integers(a));y=advance(move)
        inputs.update(np.asarray(x).tobytes());inputs.update(move.to_bytes(4,'little'));inputs.update(np.asarray(y).tobytes())
        if k in (0,31,97,199,319):
            u=tape_rng.random((5,f.J+4))
        else:u=None
        if agent is not None and u is not None:
            assert f.reward_control
            before=pickle.dumps(agent);q=f.two_step_values(x,u)
            assert q.tobytes()==f.two_step_values(x,u).tobytes() and np.isfinite(q).all()
            assert before==pickle.dumps(agent);plans+=1
        out.append(f.predict(x,move).copy());f.learn(x,move,y);x=y
    return np.stack(out),inputs.hexdigest(),plans


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dest',type=Path);p.add_argument('model');a=p.parse_args()
    sys.meta_path.insert(0,NoLegacy());a.dest.mkdir(exist_ok=False,parents=True);report={}
    for tape in ('sticky','canc-noise3'):
        arr,tape_hash,pure=play(a.model,tape)
        np.save(a.dest/f'{tape}.npy',arr,allow_pickle=False)
        report[tape]=dict(transitions=len(arr),dtype=arr.dtype.str,shape=list(arr.shape),
                          tape_sha256=tape_hash,output_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),pure_plans=pure)
    (a.dest/'tapes.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
