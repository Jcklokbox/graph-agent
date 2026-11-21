"""Lock one AR-5 predictor snapshot and supervise bounded immutable judge batches."""
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

from harness.journal import append
from harness.snap import snapshot, tree_hash
from research.ar5_holdout_next import SEEDS
from research.ar5_judge import GUARDS, NEXT, evidence, key, plan, report

WORK = {'C': (2, 1), 'A': (24, 2), 'h1': (24, 2),
        'v1': (8, 2), 'd2': (7, 2), 'd3': (8, 2), 'holdout': (24, 2),
        'credit': (12, 2), NEXT: (16, 2)}
SUITES = (*GUARDS, 'holdout')  # historical frozen evidence
NEXT_SUITES = (*GUARDS, 'credit', NEXT)
NEXT_SHA = '53577bc71a55c426d6357fd2a435f3373f75a93ce8dd052618d3b58803432f15'
CREDIT_SHA = 'd71868f8264a3d18753593d3d6cb146b6cef4b8820392cb266bcef806a6c74b6'


def lock(path, t0, commit):
    checkout = Path(__file__).resolve().parents[1]
    tree, snap = snapshot(checkout)
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) != 2:
        raise ValueError('exactly the planned two available CPUs are needed')
    hold_sha = hashlib.sha256((Path(snap)/'research/ar5_holdout_next.py').read_bytes()).hexdigest()
    credit_sha = hashlib.sha256((Path(snap)/'eval/scripts/ar5_credit_probe.py').read_bytes()).hexdigest()
    if hold_sha != NEXT_SHA or credit_sha != CREDIT_SHA:
        raise ValueError('registered D-61/D-62 schedule code changed')
    path.mkdir(parents=True, exist_ok=False)
    manifest = dict(tree=tree, snap=snap, commit=commit, t0=t0, hard=t0+14400.,
                    dispatch_deadline=t0+14100., cpus=cores, seeds=list(SEEDS), holdout_sha256=hold_sha,
                    holdout_suite=NEXT, credit_sha256=credit_sha, python=platform.python_version(), host=platform.platform(),
                    dependencies={pkg:version(pkg) for pkg in ('numpy','numba','llvmlite')},
                    predictor_development_closed=True, start=time.monotonic())
    (path/'freeze.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


def run(path, suite_list, prefix):
    f = json.loads((path/'freeze.json').read_text())
    tree, snap = f['tree'], f['snap']
    if tree_hash(snap) != tree:
        raise ValueError('snapshot changed')
    held = f.get('holdout_suite','holdout')
    allowed = NEXT_SUITES if held == NEXT else SUITES
    suite_list = allowed if suite_list is None else tuple(suite_list)
    if not set(suite_list) <= set(allowed):
        raise ValueError('dispatch cannot cross the registered holdout')
    name = 'ar5_holdout_next.py' if held == NEXT else 'ar5_holdout.py'
    wanted = hashlib.sha256((Path(snap)/'research'/name).read_bytes()).hexdigest()
    if f['holdout_sha256'] != wanted or (held == NEXT and wanted != NEXT_SHA):
        raise ValueError('held-out schedule bytes fail freeze check')
    if held == NEXT:
        credit_sha = hashlib.sha256((Path(snap)/'eval/scripts/ar5_credit_probe.py').read_bytes()).hexdigest()
        if credit_sha != CREDIT_SHA or f.get('credit_sha256') != credit_sha:
            raise ValueError('credit control bytes fail freeze check')
    runtime = {pkg:version(pkg) for pkg in f['dependencies']}
    if runtime != f['dependencies'] or platform.python_version() != f['python']:
        raise ValueError('dependency/interpreter mismatch')
    env = dict(os.environ, PG_TREE=tree, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=prefix, PYTHONPATH=os.pathsep.join(
                   [snap, os.environ.get('PYTHONPATH','/tmp/graph-agent-deps')]))
    for suite in suite_list:
        if suite == held and any(report(path,tree,g)['status'] != 'PASS' for g in allowed[:-1]):
            print('LOCKED holdout: predictor guard not passed', flush=True); return 2
        steps, workers = WORK[suite]
        tasks = plan(suite)
        while True:
            recs, _ = evidence(path,tree,suite)
            first = next((i for i,t in enumerate(tasks) if key(tree,suite,t) not in recs), len(tasks))
            if first == len(tasks):
                break
            last = min(len(tasks), first+steps)
            t = time.monotonic()
            cap = min(166, math.floor(f['dispatch_deadline'] - t - 3))
            if cap < 35:
                print('STOP near reserve', suite, first, flush=True); return 3
            cpus = ','.join(map(str, f['cpus'][:workers]))
            cmd = ['timeout','-k','1',str(cap),'nice','-n','19','taskset','-c',cpus,
                   sys.executable,'-m','research.ar5_judge','batch',suite,str(path),
                   str(first),str(last),'--workers',str(workers)]
            print('START', suite, first, last, cap, flush=True)
            proc = subprocess.run(cmd,cwd=path,env=env,check=False,timeout=cap+3)
            took = time.monotonic()-t
            append(path/'stages.jsonl', dict(k=[tree,0,f'{suite}|{first}|{last}'],
                                            m=dict(exit=proc.returncode, bounds=[first,last], cpus=cpus),dt=took))
            if proc.returncode:
                print('STOP stage', suite, first, proc.returncode, flush=True); return 4
        summary = report(path,tree,suite)
        (path/(suite+'.json')).write_text(json.dumps(summary,indent=2, sort_keys=True)+'\n')
        print('SUITE',suite,summary['status'], summary.get('estimate',summary.get('mean',summary.get('pass_rows'))), flush=True)
        if summary['status'] != 'PASS':
            return 2
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['lock','run']); p.add_argument('root',type=Path)
    p.add_argument('--t0',type=float); p.add_argument('--commit')
    p.add_argument('--suites',nargs='+', choices=WORK)
    p.add_argument('--cache',default='/tmp/ar5-cache')
    a=p.parse_args(); root=a.root.resolve()
    if a.command=='lock':
        if a.t0 is None or not a.commit:
            p.error('lock requires --t0 and --commit')
        lock(root,a.t0,a.commit)
    else:
        raise SystemExit(run(root,a.suites,a.cache))


if __name__=='__main__': main()
