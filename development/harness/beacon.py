"""Shared stdout recovery contract for human-facing runner CLIs."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Event, Thread
import time

ROOT = Path(os.environ.get('GRAPH_AGENT_ROOT', Path(__file__).resolve().parents[2]))
LOCAL = ROOT / '.run/checkpoint.json'
PUBLISHED = ROOT / 'ops/recovery.json'
MACHINE = 'GRAPH_AGENT_MACHINE'
REFLECTION_INTERVAL = 30 * 60
REFLECTION_CUE_WINDOW = 120  # Longer than the 75-second heartbeat.


def reflection_cue(elapsed):
    period, since = divmod(elapsed, REFLECTION_INTERVAL)
    if period and since < REFLECTION_CUE_WINDOW:
        return f'~{period * 30} min: reflect in STATE.md when useful (no pause needed)'
    return f'next cue in {REFLECTION_INTERVAL-since}s'


def emit(path=None, published=None):
    if os.environ.get(MACHINE) == '1':
        return
    path = Path(path or os.environ.get('GRAPH_AGENT_CHECKPOINT', LOCAL))
    try:
        source = path if path.exists() else (published or PUBLISHED)
        state = json.loads(source.read_text())
        now = time.monotonic()
        if state['boot'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip() or source != path:
            offset = now - time.time()
            start, hard = (state[k] + offset for k in ('start_epoch', 'hard_epoch'))
            phase = 'recovery-required'
        else:
            start, hard = state['started'], state['hard']
            phase = state['phase']
        elapsed = max(0, int(now-start))
        timer = 'off' if state['phase'] == 'closed' else reflection_cue(elapsed)
        report = (f"{state['run']} {phase} | wall {elapsed}/{round(hard-start)}s | "
                  f"calls {state['calls']}/{state['budget']} (working {state['budget']-state['reserve_calls']}) | "
                  f'reflection {timer}')
    except FileNotFoundError:
        report = 'no active checkpoint | wall --/-- | calls --/240 (working 220) | reflection off'
    except (OSError, KeyError, TypeError, ValueError):
        report = 'checkpoint unreadable | wall/calls/reflection unknown; reconcile, do not reinitialize'
    try:
        print(f'[STATE & CONTRACT BEACON] {report}. If compaction just occurred: run ./ar resume and read '
              'STATE.md + docs/INDEX.md immediately. DO NOT stop early during an authorized run. English only.',
              flush=True)
    except BrokenPipeError:
        pass  # A closed viewer must not mask the runner's exit status.


@contextmanager
def cli(path=None, published=None):
    if os.environ.get(MACHINE) == '1':
        yield
        return
    stop = Event()

    def pulse():
        while not stop.wait(75):
            emit(path, published)

    emit(path, published)
    heartbeat = Thread(target=pulse, daemon=True)
    heartbeat.start()
    try:
        yield
    finally:
        stop.set()
        heartbeat.join()
        emit(path, published)
