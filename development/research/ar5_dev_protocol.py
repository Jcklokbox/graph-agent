"""Candidate-independent fixed-sample DEV comparison for a future 3b-2 freeze.

One seed is a block over the 15 registered variants. Pairing requires both raw
normalization denominators and exact seed IDs, not just a cached .891 mean.
This is an exposed development battery; generalization needs its own split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from envs.tasks import SPLITS
from harness.journal import load

N = 16
T995_15 = 2.946714  # rounded upward; conservative at the locked two-sided nominal 99%
MARGIN = -.02


def paired_score(reference, candidate, tree, name):
    variants = list(SPLITS['dev'][0])
    if reference['agent'] != 'v12' or reference['split'] != 'dev' or set(reference['variants']) != set(variants):
        raise ValueError('requires the locked DEV-16 v12 cache')
    expected = {(tree, s, f'{name}|DEV|{v}') for s in range(N) for v in variants}
    if candidate.keys() != expected:
        return dict(status='INCOMPLETE', observed=len(candidate), expected=len(expected))
    diffs, rawscores, refs = {}, {}, {}
    for v in variants:
        c = reference['variants'][v]
        if c['seeds'] != list(range(N)):
            raise ValueError('reference seed order disagreement')
        rows = [candidate[tree,s,f'{name}|DEV|{v}']['m'] for s in range(N)]
        diff, cscores, rscores = [], [], []
        for row, rr, refnorm in zip(rows,c['raw'],c['per_seed']):
            if any(not math.isclose(row[k],rr[k],abs_tol=1e-12,rel_tol=0.) for k in ('oracle','stay')):
                raise ValueError(f'{v}: environment/normalization mismatch')
            gap = rr['oracle'] - rr['stay']
            if gap <= 0. or not math.isfinite(row['brain']) or not math.isfinite(refnorm):
                raise ValueError(f'{v}: invalid benchmark score')
            candidate_score = (row['brain'] - rr['stay']) / gap
            diff.append(candidate_score-refnorm); cscores.append(candidate_score); rscores.append(refnorm)
        diffs[v]=diff; rawscores[v]=cscores; refs[v]=rscores
    block = [statistics.mean(diffs[v][s] for v in variants) for s in range(N)]
    mu, sd = statistics.mean(block), statistics.stdev(block)
    half = T995_15 * sd / math.sqrt(N)
    by_variant = {v: {'diff': statistics.mean(diffs[v]), 'per_seed':diffs[v],
                       'candidate': statistics.mean(rawscores[v]), 'v12':statistics.mean(refs[v])} for v in variants}
    return dict(status='PASS' if mu-half >= MARGIN else 'FAIL', paired=mu, sd=sd,
                ci99=[mu-half,mu+half], n_blocks=N, margin=MARGIN, tuples=len(expected),
                v12=statistics.mean(z['v12'] for z in by_variant.values()),
                candidate=statistics.mean(z['candidate'] for z in by_variant.values()), variants=by_variant)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path); p.add_argument('candidate',type=Path)
    p.add_argument('tree');p.add_argument('name');p.add_argument('--ref-sha',required=True)
    a=p.parse_args()
    data=a.reference.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    if digest!=a.ref_sha: p.error('cached reference digest disagreement')
    journal,dups,bad=load(a.candidate)
    if dups or bad: p.error('duplicate or corrupt candidate evidence')
    print(json.dumps(paired_score(json.loads(data),journal,a.tree,a.name),indent=2,sort_keys=True))


if __name__=='__main__': main()
