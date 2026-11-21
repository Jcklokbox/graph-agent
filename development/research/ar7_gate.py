"""Fresh paired gate, independent freeze, and full registered watchdogs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from eval.scripts.ar5_controller_run import schedule
from harness.journal import append
from harness.snap import snapshot, tree_hash
from research.ar6_pair_gate import (N, MARGIN, T995, VARIANTS_LOCKED, _norm, _read,
                                   dependencies, digest, key)

BATCH = 10
SEEDS = tuple(1_100_000_000 + int.from_bytes(hashlib.sha256(
    f'graph-agent/ar7/paired20/v1/{i}'.encode()).digest()[:4],'big') % 900_000_000 for i in range(N))
# Historical per-variant median batch wall seconds from spent AR6 journals.
# Used for scheduling only, not to set the watchdog or select candidates.
WEIGHTS = {
    'v12': (15.1,12.5,10.4,15.3,49.8,8.9,45.8,40.9,35.0,16.8,21.6,28.0,21.3,28.2,35.0),
    'factor': (16.4,16.6,9.3,16.2,22.4,9.2,46.8,22.7,20.8,10.4,10.7,7.0,10.4,11.4,7.8),
}


def packs(model, seeds=SEEDS):
    rows = schedule(VARIANTS_LOCKED, seeds)
    if len(rows) % BATCH or len(set(seeds)) != len(seeds):
        raise ValueError('unaligned or repeated program')
    w = dict(zip(VARIANTS_LOCKED, WEIGHTS['v12' if model=='v12' else 'factor'], strict=True))
    bins = [[] for _ in range(len(rows)//BATCH)]; weights = [0.] * len(bins)
    for j in sorted(rows, key=lambda j: -w[j[0]]):
        i = min((i for i,b in enumerate(bins) if len(b)<BATCH), key=lambda i:(weights[i],i))
        bins[i].append(j); weights[i] += w[j[0]]
    return bins


EXPOSED = (6,13,2,15,7,12,8,0,1,3,4,5,9,10,11,14)


def cal_packs(model):
    # Benchmark every distinct production stage shape with only DEV16 seeds.
    shapes=dict.fromkeys(tuple(v for v, *_ in b) for b in packs(model))
    return [[schedule((v,),(EXPOSED[(stage*5+i)%len(EXPOSED)],))[0]
             for i,v in enumerate(shape)] for stage,shape in enumerate(shapes)]


def stage_key(tree, model, index):
    return tree, index, f'{model}|stage'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def calibration(path, model, tree):
    cfg=read_json(path)
    if cfg['tree']!=tree or cfg['model']!=model or cfg['workers'] not in (1,2) or cfg['deps']!=dependencies():
        raise ValueError('timings must benchmark the exact finalist tree')
    ts=cfg['stages']
    if cfg['program_sha256']!=digest(cal_packs(model)):
        raise ValueError('non-production-shape calibration')
    if not isinstance(ts,list) or len(ts)!=len(cal_packs(model)) or any(t['episodes']!=BATCH or t['exit']!=0
            or not math.isfinite(t['seconds']) or t['seconds']<=0 for t in ts):
        raise ValueError('need successful full measured stages for every production shape')
    return cfg, math.ceil(1.25*max(t['seconds'] for t in ts)) + 1


def lock(root, model, commit, deadline, measurements):
    if not (model.startswith('v') and model[1:].isdigit() and model!='v12'):
        raise ValueError('bad model')
    base=Path(__file__).resolve().parents[1];tree,snap=snapshot(base)
    if not (Path(snap)/'agents'/f'{model}.py').is_file():raise ValueError('candidate missing')
    stats={};caps={};models=['v12',model]
    program={m:packs(m) for m in models}
    for m,p in zip(models,measurements):
        stats[m],caps[m]=calibration(p,m,tree)
    if stats['v12']['workers']!=stats[model]['workers']:
        raise ValueError('different worker settings were not registered')
    cpus=sorted(os.sched_getaffinity(0))[:stats[model]['workers']]
    if len(cpus)!=stats[model]['workers'] or any(stats[m]['cpus']!=cpus for m in models):
        raise ValueError('timing and production affinity differ')
    required=sum(len(program[m])*(caps[m]+4) for m in models)+30
    if not math.isfinite(deadline) or deadline-time.time()<required:
        raise ValueError(f'cannot fund full measured watchdogs: need {required}s')
    cfg=dict(tree=tree,commit=commit,model=model,models=models,n=N,batch=BATCH,seeds=list(SEEDS),
             program_sha256=digest(program),variants=list(VARIANTS_LOCKED),margin=MARGIN,tcrit=T995,
             caps=caps,workers=len(cpus),cpus=cpus,deps=dependencies(),required=required,deadline=deadline,
             pairs=N*len(VARIANTS_LOCKED),latest_start=deadline-required,
             timings=stats)
    root.mkdir(exist_ok=False,parents=True);write_json(root/'freeze.json',cfg);print('LOCK',tree,required,caps,flush=True)


def verify(cfg):
    if (cfg['program_sha256']!=digest({m:packs(m) for m in cfg['models']}) or cfg['models']!=['v12',cfg['model']]
            or cfg['seeds']!=list(SEEDS) or cfg['variants']!=list(VARIANTS_LOCKED) or cfg['batch']!=BATCH
            or cfg['n']!=N or cfg['margin']!=MARGIN or cfg['tcrit']!=T995 or cfg['workers']!=len(cfg['cpus'])):
        raise ValueError('frozen protocol mismatch')


def score(cfg, rows, stages):
    tree=cfg['tree']; models=cfg['models']; program={m:packs(m) for m in models}
    all_e={key(tree,m,v,i) for m in models for b in program[m] for v,i,_,_ in b}
    all_s={stage_key(tree,m,i) for m in models for i in range(len(program[m]))}
    if not rows.keys()<=all_e or not stages.keys()<=all_s:
        raise ValueError('unexpected episode/stage')
    verified=set(); successful=0
    for m in models:
        for i,b in enumerate(program[m]):
            k=stage_key(tree,m,i)
            if k not in stages:continue
            r=stages[k]
            if not (r['m']['cpus']==cfg['cpus'] and r['m']['cap']==cfg['caps'][m]
                    and r['m']['program']==digest(b) and 0.<r['dt']<=cfg['caps'][m]+4):
                raise ValueError('wrong stage witness')
            if r['m']['exit']!=0:continue
            wanted={key(tree,m,v,s) for v,s,_,_ in b}
            if not wanted<=rows.keys():raise ValueError('successful stage missing episodes')
            verified.update(wanted);successful+=1
    counts=Counter(k[2] for k in verified);raw_counts=Counter(k[2] for k in rows)
    cells=[f'{m}|DEV|{v}' for m in models for v in VARIANTS_LOCKED]
    pairs=sum(all(key(tree,m,v,j) in verified for m in models) for v,j,_,_ in schedule(VARIANTS_LOCKED,SEEDS))
    report=dict(status='INCOMPLETE',tree=tree,model=cfg['model'],observed=len(rows),verified=len(verified),
                total=len(all_e),stages=successful,expected_stages=len(all_s),per_cell={c:counts[c] for c in cells},
                raw_per_cell={c:raw_counts[c] for c in cells},verified_pairs=pairs)
    if verified!=all_e:return report
    diffs=defaultdict(list);raw=defaultdict(list)
    for v,i,_,_ in schedule(VARIANTS_LOCKED,SEEDS):
        a,b=(rows[key(tree,m,v,i)]['m'] for m in models)
        if any(abs(a[k]-b[k])>1e-12 for k in ('stay','oracle')):
            raise ValueError('unmatched denominators')
        x,y=_norm(a),_norm(b);raw[v].append((x,y));diffs[v].append(y-x)
    blocks=[statistics.mean(diffs[v][i] for v in VARIANTS_LOCKED) for i in range(N)]
    delta=statistics.mean(blocks);radius=T995*statistics.stdev(blocks)/math.sqrt(N)
    report.update(status='PASS' if delta-radius>=MARGIN else 'FAIL',paired=delta,n=N,
                  ci99=[delta-radius,delta+radius],variants={v:statistics.mean(diffs[v]) for v in VARIANTS_LOCKED},
                  reference=statistics.mean(x for r in raw.values() for x,_ in r),
                  candidate=statistics.mean(y for r in raw.values() for _,y in r),per_seed=blocks)
    return report


def gate(root, cfg, cache):
    verify(cfg);snap=Path(__file__).resolve().parents[1];tree=tree_hash(snap)
    if tree!=cfg['tree'] or cfg['deps']!=dependencies():raise ValueError('not on the frozen tree and environment')
    if not set(cfg['cpus'])<=os.sched_getaffinity(0):raise ValueError('wrong CPUs')
    episodes,stages=root/'episodes.jsonl',root/'stages.jsonl'
    if episodes.exists() or stages.exists():raise ValueError('no unregistered retry')
    if cfg['deadline']-time.time()<cfg['required']:raise ValueError('insufficient time at actual start')
    env=dict(os.environ,PG_TREE=tree,GRAPH_AGENT_MACHINE='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
             NUMBA_CACHE_DIR=cache,PYTHONPATH=str(snap)+':'+os.environ.get('PYTHONPATH',''))
    for m in cfg['models']:
        cap=cfg['caps'][m]
        for i,b in enumerate(packs(m)):
            if cfg['deadline']-time.time()<cap+4:break
            print('START',m,i,cap,flush=True)
            dest=root/f'{m}-{i:02}.jsonl'; plan=root/'stage-program.json'
            write_json(plan,b);t=time.monotonic()
            cmd=['timeout','-k','2',str(cap),'taskset','-c',','.join(map(str,cfg['cpus'])),sys.executable,
                 '-m','eval.scripts.ar7_stage',str(dest),m,str(plan),str(cfg['workers'])]
            try:
                p=subprocess.run(cmd,env=env,cwd=root,timeout=cap+3,stdout=subprocess.DEVNULL)
                code=p.returncode
            except subprocess.TimeoutExpired:code=124
            dt=time.monotonic()-t
            observed=_read(dest)
            expected={key(tree,m,v,seed) for v,seed,_,_ in b}
            if not observed.keys()<=expected or code==0 and observed.keys()!=expected:
                raise ValueError('stage emitted wrong keys')
            for r in observed.values():append(episodes,r)
            append(stages,dict(k=stage_key(tree,m,i),dt=dt,m=dict(exit=code,cap=cap,cpus=cfg['cpus'],program=digest(b))))
            if code:break
            dest.unlink()  # merged with all exact rows checked above
        else:continue
        break
    result=score(cfg,_read(episodes),_read(stages));write_json(root/'DEV.json',result)
    (root/'stage-program.json').unlink(missing_ok=True)
    print(result['status'],result['verified'],flush=True)
    return {'PASS':0,'FAIL':2,'INCOMPLETE':3}[result['status']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['lock','run','audit'])
    p.add_argument('path',type=Path);p.add_argument('--model');p.add_argument('--commit');p.add_argument('--deadline',type=float)
    p.add_argument('--measure',type=Path,nargs=2);p.add_argument('--cache',default='/tmp/ar6-cache');a=p.parse_args();root=a.path.resolve()
    if a.action=='lock':
        if not all((a.model,a.commit,a.deadline,a.measure)):p.error('lock requires model, commit, deadline and two stage measurements')
        lock(root,a.model,a.commit,a.deadline,a.measure);return
    cfg=read_json(root/'freeze.json');verify(cfg)
    if a.action=='run':raise SystemExit(gate(root,cfg,a.cache))
    print(json.dumps(score(cfg,_read(root/'episodes.jsonl'),_read(root/'stages.jsonl')),indent=2,sort_keys=True))


if __name__=='__main__':main()
