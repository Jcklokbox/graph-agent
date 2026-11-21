"""Exercise the independent streaming identity on actual observed replay rows."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build,run
from eval.scripts.ar6_parent_evidence import structure_scores
from harness.beacon import cli
from harness.journal import append
from research.ar6_partition_math import WindowEvidence


def audit(f,c,x,u):
    h=f.reward;z=structure_scores(h,c,True);rows=[r for r in h.replay if h.code(r[3])==c]
    unary=z['best_unary'];joint=z['best_compound'];m=len(unary)+len(joint)
    if not m:return dict(rows=len(rows),hypotheses=0)
    bank=WindowEvidence(m,len(rows))
    prior=np.r_[np.full(len(unary),np.log(max(1,z['unary_candidates']))),
                np.full(len(joint),np.log(max(1,z['compound_candidates'])))]
    for _,_,r,_,feat,phase in rows:
        groups=[int(bool(feat[i])) for _,i in unary]
        groups.extend(int(bool(feat[l*h.J+j2])) if phase[l*h.J+j1] else 2 for _,j1,j2,l,_ in joint)
        bank.append(float(r),groups)
    ref=np.r_[[p[0] for p in unary],[p[0] for p in joint]]
    np.testing.assert_allclose(bank.score-prior,ref,rtol=0,atol=3e-12)
    return dict(rows=len(rows),hypotheses=m,max_error=float(np.max(np.abs(bank.score-prior-ref))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args()
    for v in ('canc','canc-noise3'):
        variant=VARIANTS[v]
        for s in (6,15):
            for period in (5,7):
                table=build(variant,s,4,16,period)
                for arm in ('same','opposite','neutral'):
                    record=run(variant,s,arm,table,period,audit)
                    rows=[r['structure'] for r in record['results']]
                    key=[os.environ['PG_TREE'],s,f'{v}|{arm}|period{period}']
                    append(a.out,dict(k=key,m=rows));print(key[1:],len(rows),sum(r['hypotheses'] for r in rows),flush=True)


if __name__=='__main__':
    with cli():main()
