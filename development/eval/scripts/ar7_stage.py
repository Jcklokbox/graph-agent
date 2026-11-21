"""Outcome-blind stage runner; only the gate assessor reads fresh payoffs."""
import json
from multiprocessing import get_context
from pathlib import Path
import sys

from eval.paired import run_episode
from harness.journal import append


def main():
    _,out,model,program,n=sys.argv
    work=[(model,*r) for r in json.loads(Path(program).read_text())]
    path=Path(out)
    if path.exists():raise ValueError('stage output already exists')
    if n=='1':
        for r in map(run_episode,work):append(path,r)
    else:
        with get_context('fork').Pool(int(n)) as pool:
            for r in pool.imap_unordered(run_episode,work,chunksize=1):append(path,r)


if __name__=='__main__':main()
