"""Futility/development screens, never acceptance; invoke with an external watchdog."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from harness.journal import append, load
from research.ar5_judge import T1, T2, episode, key, plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',choices=(*T2,*T1));p.add_argument('out',type=Path)
    p.add_argument('--seeds',type=int,default=4)
    p.add_argument('--rows',nargs='+');a=p.parse_args();h=os.environ['PG_TREE']
    limit = 16 if a.suite in T2 else 6
    if not 0 < a.seeds <= limit:
        p.error(f'require 1–{limit} seeds')
    todo = [t for t in plan(a.suite) if t[0]<a.seeds and (not a.rows or t[1].split('|')[0] in a.rows)]
    if not todo:
        p.error('no tasks selected')
    expected = {key(h,a.suite,t) for t in todo}
    recs, dup, bad = load(a.out)
    if dup or bad or not recs.keys()<=expected:
        raise ValueError('corrupt or incompatible screen journal')
    for t in todo:
        if key(h,a.suite,t) not in recs:
            rec = episode(h,a.suite,t); append(a.out,rec)
            print(t, round(rec['dt'],3),flush=True)


if __name__=='__main__':main()
