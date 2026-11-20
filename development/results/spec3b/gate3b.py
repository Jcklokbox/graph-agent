import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from harness import bg, journal
from harness.staged import t2_metrics, t2_tasks, t_table, t_est

H = sys.argv[1]
T0 = float(Path('/tmp/spec3b-t0').read_text())
T2 = time.monotonic()
END = min(T2 + 800, T0 + 1500)
ROOT = Path(bg.RUNS)
AG = 'factor-L4-P-C'
ENV = ['env', 'OPENBLAS_NUM_THREADS=1', 'PG_PROCS=2']


def run(name, cmd, cap=165, tree=H, cpu='0,1'):
    left = math.floor(min(cap, END - time.monotonic() - 2))
    if left < 3:
        return -1
    bg.start(name, ['timeout', '-k', '1s', str(left), *ENV, *cmd], cpu=cpu, tree=tree)
    rc = bg.wait(name, left + 3, 2000)
    return int(rc) if rc.isdecimal() else -1


def grade(wset, path):
    from envs.phenomena import SETS
    from eval.stats import interval
    rows, _, default = SETS[wset]
    rec, dup, bad = journal.load(path)
    effects = t2_metrics(rec, wset, H)
    n_yes = n_rev = 0
    for row, ph in rows.items():
        v = effects.get((AG, row), {})
        mean, sd, (lo, hi) = interval([v[i] for i in sorted(v)])
        minimum = default if ph.min_effect is None else ph.min_effect
        status = ('INCOMPLETE' if len(v) != 16 else 'PASS' if lo > 0 and mean >= minimum
                  else 'REV' if hi < 0 and mean <= -minimum else 'FAIL')
        n_yes += status == 'PASS'
        n_rev += status == 'REV'
        print('ROW', wset, row, len(v), mean, sd, lo, hi, status, flush=True)
    print('RESULT', wset, 'PASS', n_yes, 'REV', n_rev, 'counts', len(rec), dup, bad, flush=True)


def t2(wset, max_seconds=None):
    global END
    original_end = END
    if max_seconds is not None:
        END = min(END, time.monotonic() + max_seconds)
    p = ROOT / f'spec3b-final-{wset}-1' / 'j.jsonl'
    i = 1
    while END - time.monotonic() > 8:
        n = f'spec3b-final-{wset}-{i}'
        args = ['python', '-m', 'harness.staged', 't2', '--agents', AG, '--set', wset,
                '--stages', '0-15', '--journal', str(p)]
        rc = run(n, args)
        if rc != 3:
            break
        i += 1
    grade(wset, str(p))
    END = original_end


def b(wset, max_seconds=None):
    from envs.family import WORLD_SETS
    from eval.stats import interval
    global END
    old_end = END
    if max_seconds is not None:
        END = min(END, time.monotonic() + max_seconds)
    worlds = list(WORLD_SETS[wset]())
    steps = 2000 if wset == 'v1' else 4000
    todo = [(w, seed) for seed in range(6) for w in worlds]
    i = 1
    p = str(ROOT / f'spec3b-final-b-{wset}-1' / 'j.jsonl')
    estimate = 19 if wset == 'd3' else 30
    while todo and END - time.monotonic() > estimate * .55 + 5:
        count = min(12, max(2, int(1.5 * min(END - time.monotonic() - 5, 135) / estimate)))
        rc = run(f'spec3b-final-b-{wset}-{i}', ['python', '/tmp/bench3b-b.py', wset, str(steps), AG, p,
                 *(f'{w}:{seed}' for w, seed in todo[:count])])
        if rc != 0:
            break
        todo = todo[count:]
        i += 1
        durations = sorted(x['dt'] for x in journal.load(p)[0].values())
        estimate = max(8, 1.35 * durations[int(.9 * len(durations))])
    recs, dup, bad = journal.load(p)
    data = {(k[1], k[2].split('|')[2]): r['m']['gc'] for k, r in recs.items() if k[0] == H}
    seed_effects = [sum(data[(s, w)] for w in worlds) / len(worlds) for s in range(6)
                    if all((s, w) in data for w in worlds)]
    result = interval(seed_effects) if seed_effects else (None, None, None)
    print('RESULT B', wset, len(seed_effects), result, 'records', len(data), dup, bad, flush=True)
    END = old_end


if __name__ == '__main__':
    print('stage2 deadline', END, 'start', T2, flush=True)
    run('spec3b-final-c', ['python', '-m', 'eval.predictive', '--agents', AG, '--set', 'dscale',
         '--worlds', 's-2x3,s-4x3', '--steps', '600', '--seeds', '1', '--out', 'cost.json'], cap=65)
    t2('v1', max_seconds=610)
    if END - time.monotonic() > 35:
        b('d3', max_seconds=115)
    remaining = END - time.monotonic()
    # Keep the holdout untouched if the same-tree measured cost makes complete
    # 16-seed execution unreasonable; don't consume it for a dev-sized sample.
    cost = t_table(str(ROOT))
    holdout = 16 * sum(t_est(t, cost) for w in ('h1','h2') for t in t2_tasks([AG], w, None)) / 2
    print('holdout estimated seconds', holdout, 'remaining', remaining, flush=True)
    if remaining >= holdout + 12:
        t2('h1')
        t2('h2')
    else:
        print('RESULT D not started: budget guard', flush=True)
    if END - time.monotonic() > 28:
        b('d2', max_seconds=(END - time.monotonic()) * .52)
    if END - time.monotonic() > 28:
        b('v1')
