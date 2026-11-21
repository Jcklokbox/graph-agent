"""Measure exactly the stage command used by a future locked paired gate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from research.ar7_gate import (BATCH, cal_packs, dependencies, digest, key,
                               _read, write_json)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dest',type=Path);p.add_argument('model');p.add_argument('workers',type=int,choices=(1,2))
    a=p.parse_args();dest=a.dest.resolve();dest.mkdir(exist_ok=False,parents=True)
    cpus=sorted(os.sched_getaffinity(0))[:a.workers];cpu=','.join(map(str,cpus));tree=os.environ['PG_TREE']
    program=cal_packs(a.model);times=[]
    for i,b in enumerate(program):
        plan=dest/f'{i}-program.json';rows=dest/f'{i}-episodes.jsonl';write_json(plan,b)
        t=time.monotonic()
        status=subprocess.run(['timeout','-k','2','300','taskset','-c',cpu,sys.executable,
             '-m','eval.scripts.ar7_stage',str(rows),a.model,str(plan),str(a.workers)],cwd=dest).returncode
        r=dict(seconds=time.monotonic()-t,episodes=len(_read(rows)),exit=status)
        if status or _read(rows).keys()!={key(tree,a.model,v,s) for v,s,_,_ in b} or r['episodes']!=BATCH:
            raise ValueError(f'invalid stage {i}: {r}')
        times.append(r);print(a.model,i,r,flush=True)
    write_json(dest/'timings.json',dict(model=a.model,tree=tree,cpus=cpus,workers=a.workers,
               deps=dependencies(),program_sha256=digest(program),stages=times))


if __name__=='__main__':main()
