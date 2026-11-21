"""Outcome-blind stage runner; only the gate assessor reads fresh payoffs."""
import json
from multiprocessing import get_context
from pathlib import Path
import sys

from eval.scripts.ar5_controller_run import one
from harness.journal import append


def main():
    _,out,model,program,n=sys.argv
    work=[(model,*r) for r in json.loads(Path(program).read_text())]
    path=Path(out)
    if path.exists():raise ValueError('stage output already exists')
    if n=='1':
        for r in map(one,work):append(path,r)
    else:
        with get_context('fork').Pool(int(n)) as pool:
            for r in pool.imap_unordered(one,work,chunksize=1):append(path,r)


if __name__=='__main__':main()
