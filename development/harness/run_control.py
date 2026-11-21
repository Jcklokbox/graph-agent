"""Durable run checkpoint and batched local operations; see docs/AUTONOMY.md."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import time

from harness import bg
from harness.snap import snapshot

ROOT = Path(bg.HERE).parent
BOOT = Path('/proc/sys/kernel/random/boot_id')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
MAX_WAIT = 900


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def boot_id():
    return BOOT.read_text().strip()


def bounded_text(text, label, limit):
    if not text.strip() or len(text) > limit or not text.isprintable():
        raise ValueError(f'{label} must be a printable single line of 1..{limit} characters')
    return text


def initialize(path, run, seconds, reserve_seconds, budget, reserve_calls, calls, next_action):
    if path.exists():
        raise ValueError('checkpoint exists: resume it; never reset a live run clock')
    if not (0 < reserve_seconds < seconds and 0 <= calls < budget - reserve_calls and reserve_calls > 0):
        raise ValueError('invalid allowances or reserves')
    now = time.monotonic()
    state = dict(version=1, run=run, boot=boot_id(), started=now, dispatch=now + seconds - reserve_seconds,
                 hard=now + seconds, calls=calls, budget=budget, reserve_calls=reserve_calls,
                 next=next_action, jobs=[], phase='active')
    save(path, state)
    return state


def load(path):
    state = json.loads(path.read_text())
    if state.get('version') != 1:
        raise ValueError('unknown checkpoint version')
    if state['boot'] != boot_id():
        raise ValueError('sandbox rebooted; do not reset the old clock. Reconcile from persisted evidence.')
    return state


def charge(path, state, calls):
    state['calls'] += calls
    save(path, state)


def dispatch_left(state):
    if state['phase'] != 'active' or state['calls'] >= state['budget'] - state['reserve_calls']:
        return 0.
    return max(0., state['dispatch'] - time.monotonic())


def name_ok(name):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f'invalid run/job name: {name!r}')
    return name


def job_status(name):
    d = Path(bg.RUNS) / name_ok(name)
    done = d / 'done'
    return int(done.read_text()) if done.exists() else ('running' if (d / 'cmd').exists() else 'missing')


def summary(state):
    return dict(run=state['run'], language='English', phase=state['phase'], next=state['next'],
                calls_used=state['calls'], budget=state['budget'], reserve_calls=state['reserve_calls'],
                dispatch_seconds=int(dispatch_left(state)), hard_seconds=max(0, int(state['hard']-time.monotonic())),
                tree=state.get('tree'), jobs={j: job_status(j) for j in state['jobs']})


def wait(state, seconds):
    end = time.monotonic() + min(seconds, MAX_WAIT, dispatch_left(state))
    while any(job_status(j) == 'running' for j in state['jobs']):
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1., remaining))


def launch(path, state, plan):
    if not isinstance(plan, list) or not 1 <= len(plan) <= 8:
        raise ValueError('plan must be a list of 1..8 jobs')
    jobs, cpus, names = [], set(), set()
    for spec in plan:
        name = name_ok(spec['name'])
        if name in names or (Path(bg.RUNS) / name).exists():
            raise ValueError(f'duplicate or existing job: {name}')
        cpu = spec['cpu']
        if not re.fullmatch(r'\d+(,\d+)*', cpu):
            raise ValueError('cpu must be a comma-separated list of CPU IDs')
        assigned = set(map(int, cpu.split(',')))
        if not assigned <= os.sched_getaffinity(0) or cpus & assigned:
            raise ValueError('CPUs must be available and disjoint in the batch')
        seconds, cmd = spec['seconds'], spec['argv']
        if type(seconds) is not int or seconds <= 0 or not isinstance(cmd, list) or not cmd or not all(
            isinstance(x, str) and '\0' not in x for x in cmd
        ) or not cmd[0]:
            raise ValueError('expected positive integer seconds and nonempty command argv')
        jobs.append((name, cpu, seconds, cmd))
        cpus.update(assigned)
        names.add(name)
    if any(job_status(j) == 'running' for j in state['jobs']):
        raise ValueError('finish the current batch before replacing its checkpoint')
    if max(j[2] for j in jobs) + 5 > dispatch_left(state):
        raise ValueError('insufficient dispatch budget for batch and watchdog grace')
    tree, _ = snapshot(bg.HERE)
    state['jobs'] = [j[0] for j in jobs]
    state['tree'] = tree
    save(path, state)  # intent first: a killed dispatch is recoverable as missing/running/done
    for name, cpu, seconds, cmd in jobs:
        if seconds + 5 > dispatch_left(state):
            raise ValueError('dispatch deadline too close after snapshot/startup')
        bg.start(name, ['timeout', '-k', '2', str(seconds), *cmd], cpu, tree)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=ROOT / '.run/checkpoint.json')
    p.add_argument('--used', type=int, default=1, help='tool calls since last update, including this one')
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('run')
    init.add_argument('--seconds', type=int, required=True)
    init.add_argument('--budget', type=int, required=True)
    init.add_argument('--reserve-seconds', type=int, default=600)
    init.add_argument('--reserve-calls', type=int, default=8)
    init.add_argument('--next', required=True)
    sub.add_parser('resume')
    check = sub.add_parser('checkpoint')
    check.add_argument('--next', required=True)
    check.add_argument('--close', action='store_true')
    waiting = sub.add_parser('wait')
    waiting.add_argument('--seconds', type=int, default=170)
    dispatch = sub.add_parser('dispatch')
    dispatch.add_argument('plan', type=Path)
    try:
        a = p.parse_args(argv)
        if a.used < 1:
            raise ValueError('--used must include this call')
        if a.command == 'init':
            state = initialize(a.state, name_ok(a.run), a.seconds, a.reserve_seconds, a.budget,
                               a.reserve_calls, a.used, bounded_text(a.next, 'next', 240))
        else:
            state = load(a.state)
            charge(a.state, state, a.used)
            if a.command == 'checkpoint':
                next_action = bounded_text(a.next, 'next', 240)
                if a.close and any(job_status(j) == 'running' for j in state['jobs']):
                    raise ValueError('cannot close while tracked jobs are running')
                state.update(next=next_action, phase='closed' if a.close else state['phase'])
                save(a.state, state)
            elif a.command == 'wait':
                if not 1 <= a.seconds <= MAX_WAIT:
                    raise ValueError(f'wait must be 1..{MAX_WAIT} seconds')
                wait(state, a.seconds)
            elif a.command == 'dispatch':
                launch(a.state, state, json.loads(a.plan.read_text()))
        print(json.dumps(summary(state), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f'run control: {exc}\n')


if __name__ == '__main__':
    main()
