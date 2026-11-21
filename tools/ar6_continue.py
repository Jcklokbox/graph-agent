"""One clean-boundary operational continuation of the untouched AR-6 DEV-20 lock.

Run with PYTHONPATH beginning with the locked development snapshot. No failed,
partial, unverified or repeated stages are recoverable for qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from harness.beacon import cli
from research import ar6_pair_gate as g


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save_new(p,data):
    with p.open('x') as out:
        json.dump(data,out,indent=2,sort_keys=True);out.write('\n')


def prefix(root,cfg):
    rows=g._read(root/'episodes.jsonl');stages=g._read(root/'stages.jsonl')
    full=g.score(cfg,rows,stages)
    if not (full['status']=='INCOMPLETE' and 0<full['stages']==len(stages)
            and full['observed']==full['verified'] and full['verified']<full['total']):
        raise ValueError('ineligible: not a nonempty clean prefix of whole successful stages')
    for m in cfg['models']:
        temp=g._read(root/f'{m}.jsonl')
        expected={k:r for k,r in rows.items() if k[2].startswith(m+'|')}
        if temp!=expected:
            raise ValueError('ineligible: worker ledger is not exactly the published prefix')
    result={p:sha(root/p) for p in ('freeze.json','episodes.jsonl','stages.jsonl')}
    return dict(digests=result,stages=full['stages'],episodes=full['verified'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('root',type=Path);p.add_argument('--previous-done',type=Path)
    p.add_argument('--dispatch-deadline',type=float);p.add_argument('--cache',default='/tmp/ar6-cache')
    a=p.parse_args();root=a.root.resolve();f=root/'freeze.json';cfg=read(f)
    g.verify(cfg)
    snap=Path(g.__file__).resolve().parents[1]
    if g.tree_hash(snap)!=cfg['tree'] or g.dependencies()!=cfg['deps']:
        raise ValueError('operational continuation must use actual locked code and dependencies')
    prior=prefix(root,cfg);note=root/'continuation.json';initial=root/'DEV.initial.json'
    if a.action=='prepare':
        if a.previous_done is None or int(a.previous_done.read_text())!=3:
            raise ValueError('prior supervisor must have exited cleanly incomplete')
        if a.dispatch_deadline is None or a.dispatch_deadline<=max(time.time()+165,cfg['deadline']):
            raise ValueError('not enough time after the initial watchdog')
        result=read(root/'DEV.json')
        if result['status']!='INCOMPLETE' or result['verified']!=prior['episodes']:
            raise ValueError('ineligible initial result')
        save_new(initial,result)
        save_new(note,dict(reason='unstarted stages after original operational watchdog',previous=prior,
                           previous_deadline=cfg['deadline'],deadline=a.dispatch_deadline,tree=cfg['tree'],
                           previous_result_sha256=sha(initial),stage_cap=g.STAGE_CAP))
        print('operational prefix',prior['stages'],'stages,',prior['episodes'],'episode rows',flush=True)
        return 0
    ext=read(note)
    if (ext['previous']!=prior or ext['stage_cap']!=g.STAGE_CAP or ext['tree']!=cfg['tree']
        or ext['previous_deadline']!=cfg['deadline'] or ext['previous_result_sha256']!=sha(initial)
        or ext['deadline']<=max(cfg['deadline'],time.time()+35)):
        raise ValueError('continuation registration invalid or already advanced')
    return g.run(root,dict(cfg,deadline=ext['deadline']),a.cache)


if __name__=='__main__':
    with cli():raise SystemExit(main())
