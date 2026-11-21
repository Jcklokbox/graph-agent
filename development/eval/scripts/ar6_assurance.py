"""Controller isolation and exact paired predictor replay on an exogenous tape."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from envs.family import WORLD_SETS, WorldEnv
from envs.tasks import VARIANTS
from eval.scripts.ar5_control_work import measure, warm
from harness.beacon import cli
from models.factor import CANONICAL, make_factor


class NoLegacy:
    def find_spec(self, fullname, *args):
        if fullname in {'agents.v11','agents.v12','models.context_tree'}:
            raise AssertionError(f'legacy controller dependency: {fullname}')


def compare(model, tape):
    if tape=='sticky':
        world=WORLD_SETS['dscale']()['s-4x3']; env=WorldEnv(world,1000)
        n,A,V=world.n_channels,world.n_actions,world.value_index
        step=env.step
    else:
        v=VARIANTS[tape]; env=v.make(12713)
        n,A,V=v.n_inputs,v.n_actions,11
        def step(a):return env.step(a)[0]
    x=env.reset(); agent=importlib.import_module('agents.'+model).Agent(n,A,V,seed=113)
    f=agent.factor; passive=make_factor(CANONICAL,n,A,V,113)
    tape_rng=np.random.default_rng(7671); digest=hashlib.sha256(); plans=0
    for k in range(320):
        a=int(tape_rng.integers(A)); y=step(a)
        if k in (0,31,97,199,319):
            u=tape_rng.random((5,f.J+3+int(f.reward_control)))
            before=pickle.dumps(agent)
            q=f.two_step_values(x,u)
            np.testing.assert_array_equal(q,f.two_step_values(x,u))
            assert before==pickle.dumps(agent) and np.isfinite(q).all()
            plans+=1
        a0=passive.predict(x,a); a1=f.predict(x,a)
        np.testing.assert_array_equal(a0,a1); digest.update(a0.tobytes())
        passive.learn(x,a,y); f.learn(x,a,y); x=y
    return dict(transitions=k+1,plans=plans,sha256=digest.hexdigest(),slots=f.reward.slots.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--model',required=True);p.add_argument('--scale',action='store_true');a=p.parse_args()
    sys.meta_path.insert(0,NoLegacy())
    warm()
    report=dict(tree=os.environ['PG_TREE'],model=a.model,
                playback={tape:compare(a.model,tape) for tape in ('sticky','canc-noise3')})
    if a.scale:
        reports={w:measure(a.model,w,360,0,False) for w in ('s-2x3','s-4x3')}
        report['scale']=dict(worlds=reports,ratio=reports['s-4x3']['sec']/reports['s-2x3']['sec'],limit=2.50)
    result=json.dumps(report,sort_keys=True,indent=2)+'\n';a.out.write_text(result);print(result,flush=True)
    return 0 if not a.scale or report['scale']['ratio']<=2.50 else 2


if __name__=='__main__':
    with cli():raise SystemExit(main())
