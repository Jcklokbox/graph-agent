"""Durable run checkpoint and batched local operations; see docs/AUTONOMY.md."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata

from harness import bg, beacon
from harness.snap import snapshot

ROOT = beacon.ROOT
CALL_BUDGET, CALL_RESERVE, REFLECT_SECONDS = 240, 20, 1800
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
    if any('CYRILLIC' in unicodedata.name(c, '') for c in text):
        raise ValueError(f'{label} must be in English')
    return text


def initialize(path, run, seconds, reserve_seconds, budget, reserve_calls, calls, next_action):
    if path.exists():
        raise ValueError('checkpoint exists: resume it; never reset a live run clock')
    if not (0 < reserve_seconds < seconds and 0 <= calls < budget - reserve_calls and reserve_calls > 0):
        raise ValueError('invalid allowances or reserves')
    now, wall = time.monotonic(), time.time()
    state = dict(version=2, run=run, boot=boot_id(), started=now, dispatch=now + seconds - reserve_seconds,
                 hard=now + seconds, reflection=now + REFLECT_SECONDS, reflections=[],
                 start_epoch=wall, dispatch_epoch=wall + seconds - reserve_seconds,
                 hard_epoch=wall + seconds, reflection_epoch=wall + REFLECT_SECONDS,
                 calls=calls, budget=budget, reserve_calls=reserve_calls, next=next_action, jobs=[], phase='active')
    save(path, state)
    return state


def load(path):
    state = json.loads(path.read_text())
    if state.get('version') != 2:
        raise ValueError('unknown or obsolete checkpoint version; do not reinitialize an old run')
    if state['boot'] != boot_id():
        raise ValueError('sandbox rebooted: use ./ar resume to reconcile the original run')
    return state


def recover(path, published):
    """Resume a local or last published checkpoint, blocking work after reset."""
    if not path.exists() and not published.exists():
        return None
    source = path if path.exists() else published
    state = json.loads(source.read_text())
    if state.get('version') != 2:
        raise ValueError('unknown or obsolete recovery checkpoint version')
    if state['boot'] != boot_id() or source != path:
        wall, now = time.time(), time.monotonic()
        if wall < state['start_epoch']:
            raise ValueError('wall clock predates run; reconcile without dispatch')
        state.update(boot=boot_id(), started=state['start_epoch']-wall+now,
                     dispatch=state['dispatch_epoch']-wall+now, hard=state['hard_epoch']-wall+now,
                     reflection=state['reflection_epoch']-wall+now)
        if state['phase'] != 'closed':
            state['phase'] = 'reconcile'
        save(path, state)
    return state


def reflect(path, state, local, global_view, resolution, next_action):
    if state['phase'] != 'active':
        raise ValueError('reflect only after reconciling an active run')
    now = time.monotonic()
    state['reflections'].append(dict(elapsed_s=round(now-state['started']), local=local,
                                     global_view=global_view, resolution=resolution, next=next_action))
    state['next'] = next_action
    # Preserve the original cadence: a delayed reflection cannot defer the next one.
    n = int(max(0., now-state['started']) // REFLECT_SECONDS) + 1
    state['reflection'] = state['started'] + n * REFLECT_SECONDS
    state['reflection_epoch'] = state['start_epoch'] + n * REFLECT_SECONDS
    save(path, state)


def charge(path, state, calls):
    state['calls'] += calls
    save(path, state)
    if state['calls'] > state['budget']:
        raise ValueError('hard call ceiling exceeded; stop')


def dispatch_left(state):
    if state['phase'] != 'active' or state['calls'] >= state['budget'] - state['reserve_calls']:
        return 0.
    return max(0., min(state['dispatch'], state['reflection']) - time.monotonic())


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
                reflection_due=time.monotonic() >= state['reflection'] and state['phase'] != 'closed',
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
    if not dispatch_left(state) or max(j[2] for j in jobs) + 5 > state['dispatch'] - time.monotonic():
        raise ValueError('insufficient dispatch budget for batch and watchdog grace')
    tree, _ = snapshot(bg.HERE)
    state['jobs'] = [j[0] for j in jobs]
    state['tree'] = tree
    save(path, state)  # intent first: a killed dispatch is recoverable as missing/running/done
    for name, cpu, seconds, cmd in jobs:
        if not dispatch_left(state) or seconds + 5 > state['dispatch'] - time.monotonic():
            raise ValueError('dispatch deadline too close after snapshot/startup')
        bg.start(name, ['timeout', '-k', '2', str(seconds), *cmd], cpu, tree)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=beacon.LOCAL)
    p.add_argument('--published', type=Path, default=beacon.PUBLISHED)
    p.add_argument('--used', type=int, default=1, help='tool calls since last update, including this one')
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('run')
    init.add_argument('--seconds', type=int, required=True)
    init.add_argument('--budget', type=int, default=CALL_BUDGET)
    init.add_argument('--reserve-seconds', type=int, default=600)
    init.add_argument('--reserve-calls', type=int, default=CALL_RESERVE)
    init.add_argument('--next', required=True)
    sub.add_parser('resume')
    sub.add_parser('publish')
    reconcile = sub.add_parser('reconcile')
    reconcile.add_argument('--evidence', required=True, help='what survived, which work is unknown')
    reconcile.add_argument('--next', required=True)
    reflection = sub.add_parser('reflect')
    for field in ('local', 'global', 'resolve', 'next'):
        reflection.add_argument('--'+field, required=True)
    check = sub.add_parser('checkpoint')
    check.add_argument('--next', required=True)
    check.add_argument('--close', action='store_true')
    waiting = sub.add_parser('wait')
    waiting.add_argument('--seconds', type=int, default=170)
    dispatch = sub.add_parser('dispatch')
    dispatch.add_argument('plan', type=Path)
    run = sub.add_parser('exec', help='bundle non-dispatch inspection, patching or audit steps')
    run.add_argument('argv', nargs=argparse.REMAINDER)
    # Cover parse errors and help on stdout as well as normal returns.
    try:
        a = p.parse_args(argv)
    except SystemExit:
        beacon.emit()
        raise
    with beacon.cli(a.state, a.published):
        try:
            execute(a)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            p.exit(2, f'run control: {exc}\n')


def execute(a):
    if a.used < 1:
        raise ValueError('--used must include this call')
    if a.command == 'init':
        if a.published.exists() and json.loads(a.published.read_text())['phase'] != 'closed':
            raise ValueError('active published checkpoint: resume it; never reset the run clock')
        if a.budget > CALL_BUDGET or a.budget - a.reserve_calls > CALL_BUDGET - CALL_RESERVE:
            raise ValueError('ceiling is 220 working calls + 20 reserve calls')
        state = initialize(a.state, name_ok(a.run), a.seconds, a.reserve_seconds, a.budget,
                           a.reserve_calls, a.used, bounded_text(a.next, 'next', 240))
    else:
        state = recover(a.state, a.published) if a.command == 'resume' else (
            None if a.command == 'exec' and not a.state.exists() and not a.published.exists() else load(a.state))
        if state is None and a.command == 'resume':
            print('No active run; do not initialize unless authorized.')
            recovery_notes()
            return
        if state is not None:
            charge(a.state, state, a.used)
        if a.command == 'exec':
            cmd = a.argv[1:] if a.argv[:1] == ['--'] else a.argv
            if not cmd:
                raise ValueError('exec requires a command')
            code = subprocess.run(cmd, env=dict(os.environ, GRAPH_AGENT_MACHINE='1')).returncode
            if code:
                raise SystemExit(code if code > 0 else 128-code)
        elif a.command == 'reconcile':
            if state['phase'] != 'reconcile':
                raise ValueError('no sandbox reset to reconcile')
            reason = bounded_text(a.evidence, 'evidence', 480)
            next_action = bounded_text(a.next, 'next', 240)
            state.update(next=next_action, reset_evidence=reason, phase='active')
            save(a.state, state)
        elif a.command == 'publish':
            save(a.published, state)  # commit/push is a separate explicit operator decision
        elif a.command == 'reflect':
            reflect(a.state, state, bounded_text(a.local, 'local', 480),
                    bounded_text(getattr(a, 'global'), 'global', 480), bounded_text(a.resolve, 'resolve', 480),
                    bounded_text(a.next, 'next', 240))
        elif a.command == 'checkpoint':
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
    if state is not None:
        if a.state == beacon.LOCAL and a.command in ('init', 'reflect', 'reconcile', 'checkpoint'):
            update_state_md(state)
        print(json.dumps(summary(state), sort_keys=True))
    if a.command == 'resume':
        recovery_notes()


def update_state_md(state):
    path = ROOT / 'STATE.md'
    text = path.read_text()
    marker = '<!-- managed-next -->'
    next_line = f"- **Run {state['run']} ({state['phase']}):** {state['next']}"
    block = f'{marker}\n{next_line}\n{marker}'
    if text.count(marker) != 2:
        raise ValueError('STATE.md must contain one managed next-action block')
    before, _, ending = text.split(marker)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(before + block + ending)
            f.flush()
            os.fchmod(f.fileno(), path.stat().st_mode & 0o777)
            os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def recovery_notes():
    for name in ('STATE.md', 'docs/INDEX.md'):
        path = ROOT / name
        if path.exists():
            print(path.read_bytes()[:2048].decode(errors='ignore'), end='\n')


if __name__ == '__main__':
    main()
