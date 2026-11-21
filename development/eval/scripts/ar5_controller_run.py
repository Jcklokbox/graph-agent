"""Immutable DEV controller screen/batch. Journals use the locked raw score schema."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from multiprocessing import get_context

from envs.tasks import SPLITS, VARIANTS
from eval.paired import episode_key, run_episode, schedule
from harness.journal import append, load


def run(out, model, names, seeds, start=None, stop=None, workers=1):
    todo = schedule(names,seeds)
    tree = os.environ['PG_TREE']
    keys = {episode_key(tree, model, name, seed) for name,seed,_,_ in todo}
    records, dup, bad = load(out)
    if dup or bad or not records.keys()<=keys:
        raise ValueError('mixed or damaged controller journal')
    tasks = [(model,*j) for j in todo[start:stop] if episode_key(tree, model, j[0], j[1]) not in records]
    def emit(seq):
        for r in seq:
            append(out,r)
            m=r['m']; norm=(m['brain']-m['stay'])/(m['oracle']-m['stay'])
            print(r['k'][1:],round(norm,5),round(m['sec'],3),flush=True)
    if workers==1:
        emit(map(run_episode,tasks))
    elif tasks:
        with get_context('fork').Pool(workers) as pool:
            emit(pool.imap_unordered(run_episode,tasks,chunksize=1))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('out',type=Path)
    p.add_argument('--model',default='v21'); p.add_argument('--seed',type=int,nargs='+',default=list(range(16)))
    p.add_argument('--variants',nargs='+',choices=SPLITS['dev'][0],default=SPLITS['dev'][0])
    p.add_argument('--start',type=int); p.add_argument('--stop',type=int);p.add_argument('--workers',type=int,choices=[1,2],default=1)
    a=p.parse_args(); run(a.out,a.model,a.variants,a.seed,a.start,a.stop,a.workers)


if __name__=='__main__':
    from harness.beacon import cli

    with cli():
        main()
