"""Exact-key rollup for the previously registered D-62 development contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.journal import load
from research.ar5_judge import t_interval

SEEDS = tuple(range(74700, 74706))


def panel(path, name):
    rows, dup, bad = load(path)
    trees = {k[0] for k in rows}
    wanted = {(s,f'{name}-controlled|{i}|200') for s in SEEDS for i in (0,1)}
    if dup or bad or len(trees)!=1 or {k[1:] for k in rows} != wanted:
        raise ValueError(f'{name}: requires complete N200 paired evidence under one tree')
    tree = next(iter(trees))
    out = [{bool(i): rows[tree,s,f'{name}-controlled|{i}|200']['m']['4'] for i in (0,1)} for s in SEEDS]
    return tree, out


def report(signed, background):
    st, s = panel(signed,'signed')
    bt, b = panel(background,'background')
    if st!=bt:
        raise ValueError('different predictor trees')
    estimates, ok = estimates_for(s,b)
    return dict(tree=st,estimates=estimates,status='PASS' if ok else 'FAIL',tuples=2*(len(s)+len(b)))


def estimates_for(s,b):
    if len(s)!=6 or len(b)!=6:
        raise ValueError('D-62 uses six fixed seed blocks')
    def f(p): return p['pred']
    def inc(g): return f(g['neutral'])-f(g['base'])
    estimates = {
        'signed': t_interval([f(g[True]['B'])-f(g[False]['B']) for g in s]),
        'inhibitor_anchor': t_interval([f(g[True]['AY'])-f(g[True]['AX']) for g in s]),
        'background': t_interval([inc(g[False])-inc(g[True]) for g in b]),
        'background_anchor': t_interval([f(g[True]['background'])-f(g[False]['background']) for g in b]),
    }
    ok = all(e['lo'] > 0. and (k.endswith('_anchor') or e['mean'] >= .05) for k,e in estimates.items())
    return estimates,ok


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('signed',type=Path);p.add_argument('background',type=Path)
    a=p.parse_args(); print(json.dumps(report(a.signed,a.background),indent=2,sort_keys=True))


if __name__=='__main__':main()
