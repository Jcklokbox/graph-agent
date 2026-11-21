"""Matched-clock dose probe for common-window signed evidence (exposed only)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build,run
from eval.scripts.ar6_parent_projection import projected
from harness.beacon import cli
from harness.journal import append


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args()
    for v in ('canc','canc-noise3'):
        for s in (6,15):
            variant=VARIANTS[v];pos,neg,probe,cal=build(variant,s,4,16,5);ref=None
            for n in (1,2,4):
                # On the same clock and exogenous tape, only the last n
                # eligible opportunities of the full schedule deliver contact.
                for arm in ('same','opposite','neutral'):
                    z=run(variant,s,arm,(pos[-n:],neg[-n:],probe,cal),5,
                          lambda f,c,x,u: projected(f,c,x,u,True))
                    if ref is None:ref=(z['sensor'],z['calibration_rewards'])
                    elif ref!=(z['sensor'],z['calibration_rewards']):raise ValueError('mismatched dose tapes')
                    key=[os.environ['PG_TREE'],s,f'{v}|{arm}|dose{n}']
                    append(a.out,dict(k=key,m=z))
                    print(key[1:],[(r['cancelled'],round(r['structure']['q'][0]-r['structure']['q'][1],3),
                         str(r['left']) in r['structure']['choices']) for r in z['results']],flush=True)


if __name__=='__main__':
    with cli():main()
