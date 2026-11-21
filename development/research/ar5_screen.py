"""Write AR-5 reference or locked-candidate journals; never combine the two."""
from __future__ import annotations

import argparse
from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import time

from envs.phenomena import US, VALUE
from eval.phenomena import make
from harness.journal import append, load
from models.factor import CANONICAL
from research.ar5_holdout import ROWS, SEEDS


def episode(name, task):
    row, group, seed = task
    start = time.monotonic()
    s = ROWS[row].build(group, seed)
    p = make(name, seed, s.n_ch)
    by_index = {}
    for idx, key, obs in s.probes:
        by_index.setdefault(idx, []).append((key, obs))
    out = {}
    for idx, x in enumerate(s.obs):
        if idx:
            p.learn(x)
        for key, obs in by_index.get(idx, []):
            out[key] = p.predict_us(obs)
        if idx + 1 < len(s.obs):
            p.step(x)
    return {'k': [os.environ['PG_TREE'], seed, f'{name}|ar5-new|{row}|{group}'],
            'm': out, 'dt': time.monotonic() - start}


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('name'); arg.add_argument('out'); arg.add_argument('start', type=int)
    arg.add_argument('stop', type=int); arg.add_argument('--freeze', type=Path)
    arg.add_argument('--workers', type=int, default=1)
    a = arg.parse_args()
    if not 0 <= a.start < a.stop <= len(SEEDS):
        arg.error('expected an interval of locked seed offsets')
    if a.name not in ('rw', 'cfg'):
        import json
        if a.name != CANONICAL or a.freeze is None:
            arg.error('candidate requires a freeze manifest')
        freeze = json.loads(a.freeze.read_text())
        if freeze['tree'] != os.environ['PG_TREE'] or freeze['seeds'] != list(SEEDS):
            arg.error('snapshot/seed mismatch')
        if not freeze.get('predictor_development_closed'):
            arg.error('development not closed')
    records, dup, bad = load(a.out)
    if dup or bad or any(k[0] != os.environ['PG_TREE'] or not k[2].startswith(a.name + '|')
                         for k in records):
        arg.error('refusing mixed or damaged journal')
    tasks = [(r, g, seed) for seed in SEEDS[a.start:a.stop] for r, p in ROWS.items() for g in p.groups
             if (os.environ['PG_TREE'], seed, f'{a.name}|ar5-new|{r}|{g}') not in records]
    with mp.get_context('fork').Pool(a.workers) as pool:
        for record in pool.imap(partial(episode, a.name), tasks, chunksize=1):
            append(a.out, record)


if __name__ == '__main__':
    main()
