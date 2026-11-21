"""Read-only independent rollup of a frozen predictor's task and stage journals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.journal import load
from research.ar5_judge import evidence, report
from research.ar5_lock import SUITES


def validate(path):
    manifest = json.loads((path/'freeze.json').read_text())
    tree = manifest['tree']
    stages, dup, bad = load(path/'stages.jsonl')
    if dup or bad:
        raise ValueError(f'stage journal invalid: duplicate={dup}, malformed={bad}')
    if any(k[0]!=tree or v['dt']>180. for k,v in stages.items()):
        raise ValueError('stage clock or tree mismatch')
    sizes = {}
    for suite in SUITES:
        records, expected = evidence(path,tree,suite)
        saved = path/(suite+'.json')
        live = report(path,tree,suite)
        if saved.exists() and json.loads(saved.read_text()) != live:
            raise ValueError(f'{suite}: verdict disagrees with journal')
        sizes[suite] = dict(completed=len(records),expected=len(expected),status=live['status'])
    stage_order = [k[2].split('|',1)[0] for k in stages]
    if 'holdout' in stage_order:
        if any(sizes[s]['status']!='PASS' for s in SUITES[:-1]):
            raise ValueError('holdout attempted before guard acceptance')
        if stage_order[stage_order.index('holdout'):] != ['holdout']*stage_order.count('holdout'):
            raise ValueError('non-holdout stages after opening holdout')
    return dict(tree=tree,commit=manifest['commit'],sets=sizes,
                all_pass=all(s['status']=='PASS' for s in sizes.values()),
                heldout_opened='holdout' in stage_order,
                stage_count=len(stages),max_stage_s=max((r['dt'] for r in stages.values()),default=0.),
                summed_stage_s=sum(r['dt'] for r in stages.values()),
                registered_tuples=sum(r['expected'] for r in sizes.values()),
                completed_tuples=sum(r['completed'] for r in sizes.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('gate',type=Path);a=p.parse_args()
    print(json.dumps(validate(a.gate),indent=2,sort_keys=True))


if __name__=='__main__':main()
