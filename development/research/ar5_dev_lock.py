"""Freeze and supervise the exact paired AR-5 action gate (no early acceptance)."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from envs.tasks import SPLITS, VARIANTS
from eval.scripts.ar5_controller_run import schedule
from harness.journal import append, load
from harness.snap import snapshot, tree_hash
from research.ar5_dev_protocol import N, paired_score

REFERENCE = 'f34b36b38386bd88ebdeca104d90fbcc49b0cab715e91aa71227c73f32ce7370'
MAX_STAGE = 165


def locked_ref(path):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=REFERENCE:
        raise ValueError('wrong fixed DEV-16 reference cache')
    return json.loads(data)


def jobs():
    todo=schedule(SPLITS['dev'][0],range(N))
    if len(todo)!=240 or sum(VARIANTS[v].steps for v,i,_,_ in todo)!=177600:
        raise ValueError('registered DEV schedule changed')
    return todo


def lock(root,reference,model,commit,deadline,episode_bound):
    locked_ref(reference);todo=jobs()
    if not 0.<episode_bound<MAX_STAGE:
        raise ValueError('invalid bounded-stage throughput estimate')
    checkout=Path(__file__).resolve().parents[1];tree,snap=snapshot(checkout)
    if not (Path(snap)/'agents'/f'{model}.py').is_file():
        raise ValueError('model does not exist in frozen agent tree')
    affinity=sorted(os.sched_getaffinity(0))[:2]
    if len(affinity)!=2:
        raise ValueError('exact two-core qualification required')
    root.mkdir(parents=True,exist_ok=False);(root/'reference.json').write_bytes(reference.read_bytes())
    frozen=dict(tree=tree,snap=snap,model=model,commit=commit,reference_sha256=REFERENCE,
                deadline_epoch=deadline,dispatch_epoch=time.time(),episode_bound=episode_bound,
                batch=max(2,min(10,2*math.floor(126./episode_bound))),
                pairs=len(todo),steps=177600,seeds=list(range(N)),cpus=affinity,
                python=platform.python_version(),deps={x:version(x) for x in ('numpy','numba','llvmlite')},
                schedule_sha256=hashlib.sha256(repr(todo).encode()).hexdigest())
    (root/'freeze.json').write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
    print(json.dumps(frozen,sort_keys=True),flush=True)


def run(root,cache):
    cfg=json.loads((root/'freeze.json').read_text());tree=cfg['tree'];snap=cfg['snap'];model=cfg['model']
    ref=locked_ref(root/'reference.json');todo=jobs()
    if tree_hash(snap)!=tree or cfg['reference_sha256']!=REFERENCE:
        raise ValueError('freeze manifest mismatch')
    if cfg['schedule_sha256']!=hashlib.sha256(repr(todo).encode()).hexdigest():
        raise ValueError('schedule changed')
    if cfg['python']!=platform.python_version() or {p:version(p) for p in cfg['deps']}!=cfg['deps']:
        raise ValueError('execution environment changed')
    cpus=','.join(map(str,cfg['cpus']));jp=root/'candidate.jsonl';sj=root/'stages.jsonl'
    env=dict(os.environ,PG_TREE=tree,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1',NUMBA_CACHE_DIR=cache,PYTHONPATH=os.pathsep.join(
                 [snap,os.environ.get('PYTHONPATH','/tmp/graph-agent-deps')]))
    keys=[(tree,s,f'{model}|DEV|{v}') for v,s,_,_ in todo]
    for first in range(0,len(todo),cfg['batch']):
        stop=min(len(todo),first+cfg['batch'])
        records,dup,bad=load(jp);previous,dup2,bad2=load(sj)
        if dup or bad or dup2 or bad2 or not records.keys()<=set(keys[:stop]):
            raise ValueError('invalid candidate or stage journal')
        key=(tree,0,f'DEV|{first}|{stop}')
        if key in previous:
            if previous[key]['m']['exit'] or not set(keys[first:stop])<=records.keys():
                raise ValueError('failed/incomplete frozen stage; no hidden retry')
            continue
        cap=min(MAX_STAGE,math.floor(cfg['deadline_epoch']-time.time()-3))
        if cap<40:
            print('STOP before reserved wall time',first,flush=True);break
        cmd=['timeout','-k','1',str(cap),'nice','-n','19','taskset','-c',cpus,sys.executable,
             '-m','eval.scripts.ar5_controller_run',str(jp),'--model',model,
             '--start',str(first),'--stop',str(stop),'--workers','2']
        t=time.monotonic();print('START',first,stop,cap,flush=True)
        try:
            code=subprocess.run(cmd,cwd=root,env=env,timeout=cap+3).returncode
        except subprocess.TimeoutExpired:
            code=124
        append(sj,dict(k=list(key),m=dict(exit=code,cpus=cpus,bounds=[first,stop]),dt=time.monotonic()-t))
        if code:
            print('STOP failed frozen stage',first,stop,code,flush=True);break
    rows,dup,bad=load(jp)
    if dup or bad:
        raise ValueError('candidate journal corrupted')
    report=paired_score(ref,rows,tree,model)
    (root/'DEV.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(report['status'],len(rows),report.get('paired'),report.get('ci99'),flush=True)
    return {'PASS':0,'FAIL':2,'INCOMPLETE':3}[report['status']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['lock','run'])
    p.add_argument('root',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--commit')
    p.add_argument('--model');p.add_argument('--deadline',type=float);p.add_argument('--episode-bound',type=float)
    p.add_argument('--cache',default='/tmp/ar5-cache');a=p.parse_args()
    if a.mode=='lock':
        if not all((a.reference,a.commit,a.model,a.deadline,a.episode_bound)):
            p.error('lock requires reference, commit, model, deadline and episode bound')
        lock(a.root.resolve(),a.reference,a.model,a.commit,a.deadline,a.episode_bound)
    else:
        raise SystemExit(run(a.root.resolve(),a.cache))


if __name__=='__main__':main()
