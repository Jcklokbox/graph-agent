"""Candidate-blind reference audit of the second locked schedule registry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.phenomena import make
from harness.journal import append, load
from research.ar5_holdout_next import SEEDS, ROWS, FLOOR, controls
from research.ar5_judge import t_interval


def episode(model, row, group, seed):
    schedule = ROWS[row].build(group, seed)
    learner = make(model, seed, schedule.n_ch)
    readings = {}
    for idx, label, x in schedule.probes:
        readings.setdefault(idx, []).append((label,x))
    out = {}
    for idx, x in enumerate(schedule.obs):
        if idx:
            learner.learn(x)
        for label, obs in readings.get(idx, ()):
            out[label] = learner.predict_us(obs)
        if idx+1 < len(schedule.obs):
            learner.step(x)
    return out


def rollup(rows):
    panel = {seed: {r: {} for r in ROWS} for seed in SEEDS}
    for seed, row, group, out in rows:
        if group in panel[seed][row] or group not in ROWS[row].groups:
            raise ValueError('unexpected group or duplicate')
        panel[seed][row][group] = out
    for p in panel.values():
        if any(set(p[r]) != set(ROWS[r].groups) for r in ROWS):
            raise ValueError('incomplete panel')
    ctl = [controls(p) for p in panel.values()]
    return dict(effects={r:t_interval([phen.effect(p[r]) for p in panel.values()]) for r,phen in ROWS.items()},
                controls={name:t_interval([c[name] for c in ctl]) for name in ctl[0]})


def next_verdict(rows):
    rpt = rollup(rows)
    hits = sorted(k for k,v in rpt['effects'].items() if v['lo'] > 0 and v['mean'] >= FLOOR)
    revs = sorted(k for k,v in rpt['effects'].items() if v['hi'] < 0 and v['mean'] <= -FLOOR)
    valid = all(v['lo'] > 0 for k,v in rpt['controls'].items() if k != 'direct' or 'forward-associate' in hits)
    ok = len(hits) >= 3 and not revs and valid and {'compensated-inhibitor','competing-hazards'} <= set(hits)
    return dict(**rpt,status='PASS' if ok else 'FAIL',pass_rows=hits,rev_rows=revs,valid_controls=valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref', choices=['rw','cfg']); parser.add_argument('out', type=Path)
    a = parser.parse_args(); h=os.environ['PG_TREE']
    recs, dup, bad = load(a.out)
    if dup or bad or any(k[0] != h or not k[2].startswith(a.ref+'|') for k in recs):
        raise ValueError('mixed or malformed reference journal')
    for seed in SEEDS:
        for row, p in ROWS.items():
            for group in p.groups:
                key = h, seed, f'{a.ref}|{row}|{group}'
                if key not in recs:
                    rec = dict(k=key,m=episode(a.ref,row,group,seed))
                    append(a.out,rec); recs[key] = rec
    rows = []
    for (_, seed, slug), rec in recs.items():
        _, row, group = slug.split('|')
        rows.append((seed,row,group,rec['m']))
    print(json.dumps(rollup(rows),indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
