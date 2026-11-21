"""Dispatch one frozen gate with the existing bg/staged runner and journal.

Usage in development/: python results/spec333/run_gate.py TREE T0 EVIDENCE
The evidence directory must exist, be empty, and be outside the snapshot.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

from harness import bg, journal
from harness.snap import SNAP_ROOT, tree_hash
from harness.staged import t2_metrics
from eval.stats import interval
from envs.phenomena import SETS
from envs.family import WORLD_SETS

H, T0, OUT = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]).resolve()
DEADLINE = T0 + 1565.  # leaves 55 s for checkpoint and 180 s untouched
HERE = Path(__file__).resolve().parent
AGENT = 'factor-L4-P-C'
N_T2, N_B = 16, 6
T975_15 = 2.13144955  # rounded upwards, alpha <= .05


def run(name, args, cap=169, cpu='0,1'):
    remaining = min(cap, math.floor(DEADLINE - time.monotonic() - 2))
    if remaining < 4:
        return -1
    start = time.monotonic()
    bg.start('333-final-' + name, ['timeout', '-k', '1s', str(remaining), 'env',
             'OPENBLAS_NUM_THREADS=1', 'NUMBA_CACHE_DIR=/tmp/333-numba', 'PG_PROCS=2', *args],
             tree=H, cpu=cpu)
    rc = bg.wait('333-final-' + name, remaining + 1, 2000)
    print('STAGE', name, time.monotonic() - start, rc, flush=True)
    return int(rc) if rc.isdecimal() else -1


def t2(wset):
    path = OUT / f't2-{wset}.jsonl'
    if path.exists():
        raise RuntimeError(f'refusing a second suite attempt: {path}')
    i = 1
    while DEADLINE - time.monotonic() > 12:
        rc = run(f't2-{wset}-{i}', ['python', '-m', 'harness.staged', 't2', '--agents', AGENT,
                 '--set', wset, '--stages', '0-15', '--journal', str(path)])
        if rc != 3:
            break
        i += 1
    records, dup, bad = journal.load(path)
    rows, _, default = SETS[wset]
    metrics = t2_metrics(records, wset, H)
    out = {}
    for name, row in rows.items():
        seeds = metrics.get((AGENT, name), {})
        values = [seeds[seed] for seed in sorted(seeds)]
        n = len(values)
        mu, sd, _ = interval(values) if n else (None, None, None)
        threshold = default if row.min_effect is None else row.min_effect
        ci = None if n != N_T2 else [mu - T975_15 * sd / 4, mu + T975_15 * sd / 4]
        verdict = ('INCOMPLETE' if ci is None or dup or bad else 'PASS' if ci[0] > 0 and mu >= threshold
                   else 'REV' if ci[1] < 0 and mu <= -threshold else 'FAIL')
        out[name] = dict(delta=mu, sd=sd if n > 1 else None, n=n, ci95=ci, status=verdict,
                         z=mu * n**.5 / sd if n > 1 and sd > 0 else None)
    print('T2', wset, {s: sum(r['status'] == s for r in out.values())
                     for s in ('PASS', 'REV', 'FAIL', 'INCOMPLETE')}, 'records', len(records), dup, bad,
          flush=True)
    (OUT / f't2-{wset}.json').write_text(json.dumps(out, indent=2) + '\n')


def b():
    thresholds = {'v1': .80, 'd2': .89, 'd3': .95}
    sizes = {'d2': 4000, 'd3': 4000, 'v1': 2000}
    # Bench traces are much cheaper after the HMM grouping has committed. Initial
    # planning bounds include the preceding uncommitted N200 filter time.
    initial = {'d2': 18., 'd3': 12., 'v1': 9.}
    # Lock all tuple IDs in advance. Balance progress by seed, then by world.
    worlds = {w: list(WORLD_SETS[w]()) for w in sizes}
    paths = {w: OUT / f'b-{w}.jsonl' for w in sizes}
    if any(p.exists() for p in paths.values()):
        raise RuntimeError('refusing to replay a T1 journal')
    for w in sizes:
        todo = [(world, seed) for seed in range(N_B) for world in worlds[w]]
        i = 1
        while todo and DEADLINE - time.monotonic() > initial[w] * .55 + 5:
            recs = journal.load(paths[w])[0]
            samples = {}
            for k, r in recs.items():
                world = k[2].split('|')[2]
                samples.setdefault(world, []).append(r['dt'])
            # Guard stage budget even if seed-dependent grouping has not yet converged.
            cap = min(137., DEADLINE - time.monotonic() - 5.) * 2
            tot = 0.
            chunk = []
            for world, seed in todo:
                estimate = 1.45 * max([initial[w], *samples.get(world, [])])
                if chunk and tot + estimate > cap:
                    break
                chunk.append(f'{world}:{seed}')
                tot += estimate
            if tot > 300 or not chunk:
                break
            rc = run(f'b-{w}-{i}', ['python', str(HERE / 'bench_b.py'), w, str(sizes[w]), AGENT,
                     str(paths[w]), *chunk])
            if rc != 0:
                break
            todo = todo[len(chunk):]
            i += 1
        recs, dup, bad = journal.load(paths[w])
        data = {(k[1], k[2].split('|')[2]): r['m']['gc'] for k, r in recs.items() if k[0] == H}
        values = [sum(data[s, world] for world in worlds[w]) / len(worlds[w]) for s in range(N_B)
                  if all((s, world) in data for world in worlds[w])]
        n = len(values)
        mu, sd, ci = interval(values) if n else (None, None, None)
        row = dict(delta=mu, sd=sd if n > 1 else None, z=(mu-thresholds[w])*n**.5/sd if n > 1 and sd > 0 else None,
                   n=n, ci95=ci if n > 1 else None, tuples=len(data), dup=dup, bad=bad,
                   status='INCOMPLETE' if n != N_B or dup or bad else 'PASS' if mu >= thresholds[w] else 'FAIL')
        (OUT / f'b-{w}.json').write_text(json.dumps(row, indent=2) + '\n')
        print('B', w, row, flush=True)


if __name__ == '__main__':
    if any(OUT.iterdir()) or tree_hash(os.path.join(SNAP_ROOT, H)) != H:
        raise RuntimeError('expected an immutable frozen tree and an empty evidence directory')
    lock = {'tree': H, 't0': T0, 'start': time.monotonic(), 'deadline': DEADLINE,
            't2_seed_ids': list(range(N_T2)), 'b_seed_ids': list(range(N_B)), 'held_out': ['h1', 'h2']}
    (OUT / 'freeze.json').write_text(json.dumps(lock, indent=2) + '\n')
    run('c', ['python', str(HERE / 'cost_c.py'), str(OUT / 'c.json')], cap=50, cpu='0')
    t2('v1')
    t2('h1')
    t2('h2')
    b()
    print('GATE END', time.monotonic(), flush=True)
