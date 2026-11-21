"""Fresh fixed-seed DEV policy noninferiority; known units, not T4."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from envs.tasks import SPLITS, VARIANTS
from eval.scripts.ar5_controller_run import schedule
from harness.beacon import cli
from harness.journal import append, load
from harness.snap import snapshot, tree_hash

N, T995, MARGIN, BATCH, STAGE_CAP = 20, 2.861, -.02, 8, 165
VARIANTS_LOCKED=tuple(SPLITS['dev'][0])
# Candidate-independent domain namespace. Offsets exclude all development and historical held-out streams.
SEEDS=tuple(100_000+int.from_bytes(hashlib.sha256(f'graph-agent/ar6/dev20/v1/{i}'.encode()).digest()[:4],'big')%1_000_000_000
            for i in range(N))


def jobs():
    result=schedule(VARIANTS_LOCKED,SEEDS)
    if len(VARIANTS_LOCKED)!=15 or len(set(SEEDS))!=N or len(result)!=300:
        raise ValueError('unexpected locked schedule')
    return result


def plan(cfg):
    tasks=jobs()
    for lo in range(0,len(tasks),BATCH):
        hi=min(lo+BATCH,len(tasks))
        for model in ('v12',cfg['model']):
            yield model,lo,hi


def key(tree,model,v,i):
    return (tree,i,f'{model}|DEV|{v}')


def _read(path):
    rows,dup,bad=load(path)
    if dup or bad:
        raise ValueError(f'corrupt or duplicate journal: {path.name}')
    return rows


def _norm(r):
    raw=[r[x] for x in ('brain','stay','oracle')]
    if not all(math.isfinite(x) and 0.<=x<=1. for x in raw) or not raw[2]>raw[1]:
        raise ValueError('invalid raw benchmark payoff or gap')
    return (raw[0]-raw[1])/(raw[2]-raw[1])


def score(cfg, rows, stages):
    models=('v12',cfg['model']); tasks=jobs(); tree=cfg['tree']
    expected={key(tree,m,v,i) for m in models for v,i,_,_ in tasks}
    p=list(plan(cfg)); skeys=[(tree,0,f'{m}|{lo}|{hi}') for m,lo,hi in p]
    if not rows.keys()<=expected or not stages.keys()<=set(skeys):
        raise ValueError('unexpected model, seed, snapshot or stage')
    verified=set(); successful=0
    for (m,lo,hi), sk in zip(p,skeys):
        if sk not in stages:
            break
        st=stages[sk]
        if not 0.<=st['dt']<=STAGE_CAP+4 or st['m']['cpus']!=cfg['cpus'] or st['m']['slice']!=[m,lo,hi]:
            raise ValueError('bad stage identity or timing')
        kk={key(tree,m,v,i) for v,i,_,_ in tasks[lo:hi]}
        if st['m']['exit']!=0:
            break
        if not kk<=rows.keys():
            raise ValueError('successful stage missing rows')
        verified.update(kk); successful+=1
    report=dict(status='INCOMPLETE',observed=len(rows),verified=len(verified),total=len(expected),
                stages=successful,expected_stages=len(p),tree=tree,model=cfg['model'])
    if verified!=expected:
        return report
    by=collections.defaultdict(list); raw=collections.defaultdict(list)
    for v,i,_,_ in tasks:
        a,b=(rows[key(tree,m,v,i)]['m'] for m in models)
        if any(abs(a[k]-b[k])>1e-12 for k in ('oracle','stay')):
            raise ValueError('mismatched denominators')
        z0,z1=_norm(a),_norm(b)
        by[v].append(z1-z0);raw[v].append([z0,z1])
    blocks=[statistics.mean(by[v][j] for v in VARIANTS_LOCKED) for j in range(N)]
    mean=statistics.mean(blocks);sd=statistics.stdev(blocks);radius=T995*sd/math.sqrt(N)
    report.update(status='PASS' if mean-radius>=MARGIN else 'FAIL',paired=mean,sd=sd,
                  candidate=statistics.mean(p[1] for pairs in raw.values() for p in pairs),
                  reference=statistics.mean(p[0] for pairs in raw.values() for p in pairs),
                  ci99=[mean-radius,mean+radius],margin=MARGIN,n=N,per_seed=blocks,
                  variants={v:statistics.mean(by[v]) for v in VARIANTS_LOCKED})
    return report


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def dependencies():
    return dict(python=platform.python_version(),**{k:version(k) for k in ('numpy','numba','llvmlite')})


def lock(root,model,commit,deadline):
    if not (model.startswith('v') and model[1:].isdigit() and model!='v12'):
        raise ValueError('invalid candidate name')
    base=Path(__file__).resolve().parents[1];tree,snap=snapshot(base)
    if not (Path(snap)/'agents'/f'{model}.py').is_file():
        raise ValueError('candidate missing')
    cpus=sorted(os.sched_getaffinity(0))[:2]
    if len(cpus)!=2:
        raise ValueError('need exactly two declared workers')
    cfg=dict(version=1,tree=tree,model=model,commit=commit,cpus=cpus,deadline=deadline,
             models=['v12',model],seed_ids=list(SEEDS),variants=list(VARIANTS_LOCKED),n=N,
             pairs=len(jobs()),margin=MARGIN,tcrit=T995,program_sha256=digest(jobs()),batch=BATCH,
             deps=dependencies(),steps_per_model=sum(VARIANTS[v].steps for v,_,_,_ in jobs()))
    root.mkdir(parents=True,exist_ok=False);(root/'freeze.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
    print('LOCK',tree,model,len(jobs()),cfg['program_sha256'],flush=True)


def verify(cfg):
    if (cfg['program_sha256']!=digest(jobs()) or cfg['models']!=['v12',cfg['model']]
        or cfg['seed_ids']!=list(SEEDS) or cfg['variants']!=list(VARIANTS_LOCKED)
        or cfg['margin']!=MARGIN or cfg['tcrit']!=T995 or cfg['batch']!=BATCH or cfg['n']!=N):
        raise ValueError('frozen protocol mismatch')


def run(root,cfg,cache):
    snap=Path(__file__).resolve().parents[1]
    if cfg['tree']!=tree_hash(snap) or cfg['deps']!=dependencies():
        raise ValueError('must run on the unchanged locked tree and environment')
    jp,sj=root/'episodes.jsonl',root/'stages.jsonl'; cpus=','.join(map(str,cfg['cpus']))
    env=dict(os.environ,PG_TREE=cfg['tree'],GRAPH_AGENT_MACHINE='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
             NUMBA_CACHE_DIR=cache,PYTHONPATH=str(snap)+':'+os.environ.get('PYTHONPATH',''))
    for m,lo,hi in plan(cfg):
        stage=(cfg['tree'],0,f'{m}|{lo}|{hi}'); prior=_read(sj); rows=_read(jp)
        if stage in prior:
            if prior[stage]['m']['exit']:
                break
            continue
        cap=min(STAGE_CAP,math.floor(cfg['deadline']-time.time()-5))
        if cap<35:
            break
        temp=root/f'{m}.jsonl'
        cmd=['timeout','-k','2',str(cap),'taskset','-c',cpus,sys.executable,'-m','eval.scripts.ar5_controller_run',
             str(temp),'--model',m,'--start',str(lo),'--stop',str(hi),'--workers','2','--seed',*map(str,SEEDS)]
        t=time.monotonic();print('START',m,lo,hi,cap,flush=True)
        try:
            code=subprocess.run(cmd,cwd=root,env=env,timeout=cap+3,stdout=subprocess.DEVNULL).returncode
        except subprocess.TimeoutExpired:
            code=124
        sr=_read(temp)
        for v,i,_,_ in jobs()[lo:hi]:
            k=key(cfg['tree'],m,v,i)
            if k in sr and k not in rows:
                append(jp,sr[k])
        append(sj,dict(k=stage,m=dict(exit=code,cpus=cfg['cpus'],slice=[m,lo,hi]),dt=time.monotonic()-t))
        if code:
            break
    result=score(cfg,_read(jp),_read(sj))
    (root/'DEV.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'],result['verified'],flush=True)
    return {'PASS':0,'FAIL':2,'INCOMPLETE':3}[result['status']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['lock','run','audit'])
    p.add_argument('path',type=Path);p.add_argument('--model');p.add_argument('--commit');p.add_argument('--deadline',type=float)
    p.add_argument('--cache',default='/tmp/ar6-cache');a=p.parse_args();root=a.path.resolve()
    if a.action=='lock':
        if not all((a.model,a.commit,a.deadline)):
            p.error('lock requires model, commit and deadline')
        lock(root,a.model,a.commit,a.deadline);return
    cfg=json.loads((root/'freeze.json').read_text());verify(cfg)
    if a.action=='run':
        raise SystemExit(run(root,cfg,a.cache))
    print(json.dumps(score(cfg,_read(root/'episodes.jsonl'),_read(root/'stages.jsonl')),indent=2,sort_keys=True))


if __name__=='__main__':
    with cli():
        main()
