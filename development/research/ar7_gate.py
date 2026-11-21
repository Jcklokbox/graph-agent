"""Read-only audit of the completed AR-7 paired qualification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from eval.paired import episode_key, normalized_payoff, schedule
from harness.journal import load_strict
from harness.snap import program_digest

N, T995, MARGIN, BATCH = 20, 2.861, -.02, 10
VARIANTS_LOCKED = (
    'aut', 'canc', 'dist', 'aut-half', 'aut-noise4', 'dist-delay3', 'aut-serial300',
    'aut-noise4-sticky', 'canc-noise3', 'dist-noise2-sticky', 'dist-noise2',
    'aut-fooddelay2', 'dist-delay3-noise2', 'dist-noise3-p4', 'aut-fooddelay3-noise1',
)
SEEDS = tuple(1_100_000_000 + int.from_bytes(hashlib.sha256(
    f'graph-agent/ar7/paired20/v1/{i}'.encode()).digest()[:4],'big') % 900_000_000 for i in range(N))
# Historical per-variant median batch wall seconds from spent AR6 journals.
# Retained to reconstruct the registered stage program during audit.
WEIGHTS = {
    'v12': (15.1,12.5,10.4,15.3,49.8,8.9,45.8,40.9,35.0,16.8,21.6,28.0,21.3,28.2,35.0),
    'factor': (16.4,16.6,9.3,16.2,22.4,9.2,46.8,22.7,20.8,10.4,10.7,7.0,10.4,11.4,7.8),
}


def packs(model):
    rows = schedule(VARIANTS_LOCKED, SEEDS)
    if len(rows) % BATCH or len(set(SEEDS)) != len(SEEDS):
        raise ValueError('unaligned or repeated program')
    w = dict(zip(VARIANTS_LOCKED, WEIGHTS['v12' if model=='v12' else 'factor'], strict=True))
    bins = [[] for _ in range(len(rows)//BATCH)]; weights = [0.] * len(bins)
    for j in sorted(rows, key=lambda j: -w[j[0]]):
        i = min((i for i,b in enumerate(bins) if len(b)<BATCH), key=lambda i:(weights[i],i))
        bins[i].append(j); weights[i] += w[j[0]]
    return bins


def stage_key(tree, model, index):
    return tree, index, f'{model}|stage'


def verify(cfg):
    if (cfg['program_sha256']!=program_digest({m:packs(m) for m in cfg['models']}) or cfg['models']!=['v12',cfg['model']]
            or cfg['seeds']!=list(SEEDS) or cfg['variants']!=list(VARIANTS_LOCKED) or cfg['batch']!=BATCH
            or cfg['n']!=N or cfg['margin']!=MARGIN or cfg['tcrit']!=T995 or cfg['workers']!=len(cfg['cpus'])):
        raise ValueError('frozen protocol mismatch')


def score(cfg, rows, stages):
    tree=cfg['tree']; models=cfg['models']; program={m:packs(m) for m in models}
    all_e={episode_key(tree,m,v,i) for m in models for b in program[m] for v,i,_,_ in b}
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
                    and r['m']['program']==program_digest(b) and 0.<r['dt']<=cfg['caps'][m]+4):
                raise ValueError('wrong stage witness')
            if r['m']['exit']!=0:continue
            wanted={episode_key(tree,m,v,s) for v,s,_,_ in b}
            if not wanted<=rows.keys():raise ValueError('successful stage missing episodes')
            verified.update(wanted);successful+=1
    counts=Counter(k[2] for k in verified);raw_counts=Counter(k[2] for k in rows)
    cells=[f'{m}|DEV|{v}' for m in models for v in VARIANTS_LOCKED]
    pairs=sum(all(episode_key(tree,m,v,j) in verified for m in models) for v,j,_,_ in schedule(VARIANTS_LOCKED,SEEDS))
    report=dict(status='INCOMPLETE',tree=tree,model=cfg['model'],observed=len(rows),verified=len(verified),
                total=len(all_e),stages=successful,expected_stages=len(all_s),per_cell={c:counts[c] for c in cells},
                raw_per_cell={c:raw_counts[c] for c in cells},verified_pairs=pairs)
    if verified!=all_e:return report
    diffs=defaultdict(list);raw=defaultdict(list)
    for v,i,_,_ in schedule(VARIANTS_LOCKED,SEEDS):
        a,b=(rows[episode_key(tree,m,v,i)]['m'] for m in models)
        if any(abs(a[k]-b[k])>1e-12 for k in ('stay','oracle')):
            raise ValueError('unmatched denominators')
        x,y=normalized_payoff(a),normalized_payoff(b);raw[v].append((x,y));diffs[v].append(y-x)
    blocks=[statistics.mean(diffs[v][i] for v in VARIANTS_LOCKED) for i in range(N)]
    delta=statistics.mean(blocks);radius=T995*statistics.stdev(blocks)/math.sqrt(N)
    report.update(status='PASS' if delta-radius>=MARGIN else 'FAIL',paired=delta,n=N,
                  ci99=[delta-radius,delta+radius],variants={v:statistics.mean(diffs[v]) for v in VARIANTS_LOCKED},
                  reference=statistics.mean(x for r in raw.values() for x,_ in r),
                  candidate=statistics.mean(y for r in raw.values() for _,y in r),per_seed=blocks)
    return report



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit'])
    parser.add_argument('path', type=Path)
    root = parser.parse_args(argv).path.resolve()
    cfg = json.loads((root / 'freeze.json').read_text())
    verify(cfg)
    report = score(cfg, load_strict(root / 'episodes.jsonl'), load_strict(root / 'stages.jsonl'))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
