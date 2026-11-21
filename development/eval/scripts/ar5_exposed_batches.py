"""Bounded, seed-major exposed panels with throughput logging, never gate evidence."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import math
from envs.tasks import SPLITS,VARIANTS
from harness.journal import append,load
from harness.snap import snapshot,tree_hash
from eval.scripts.ar5_controller_run import schedule


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--model',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--variants',nargs='+',choices=SPLITS['dev'][0],default=SPLITS['dev'][0])
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--batch',type=int,default=6)
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    tree,snap=snapshot(Path(__file__).resolve().parents[2]);tasks=schedule(a.variants,a.seeds)
    if tree_hash(snap)!=tree:raise ValueError('bad code snapshot')
    cpus=','.join(map(str,sorted(os.sched_getaffinity(0))[:2]));out=root/'candidate.jsonl'
    env=dict(os.environ,PG_TREE=tree,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
             NUMBA_CACHE_DIR='/tmp/ar5-cache',PYTHONPATH=snap+':'+os.environ.get('PYTHONPATH','/tmp/graph-agent-deps'))
    for first in range(0,len(tasks),a.batch):
        stop=min(len(tasks),first+a.batch)
        cap=min(164,math.floor(a.deadline-time.time()-3))
        if cap<25:break
        t=time.monotonic()
        command=['timeout','-k','1',str(cap),'nice','-n','19','taskset','-c',cpus,sys.executable,
                 '-m','eval.scripts.ar5_controller_run',str(out),'--model',a.model,'--start',str(first),
                 '--stop',str(stop),'--seed',*map(str,a.seeds),'--variants',*a.variants,'--workers','2']
        code=subprocess.run(command,cwd=root,env=env,timeout=cap+3).returncode
        records,dup,bad=load(out)
        if dup or bad:raise ValueError('panel journal corrupt')
        completed=[(v,i) for v,i,_,_ in tasks[first:stop] if (tree,i,f'{a.model}|DEV|{v}') in records]
        append(root/'stages.jsonl',dict(k=[tree,0,f'{first}|{stop}'],m=dict(exit=code,steps=sum(VARIANTS[v].steps for v,i in completed),count=len(completed)),dt=time.monotonic()-t))
        if code:raise SystemExit(code)

if __name__=='__main__':main()
