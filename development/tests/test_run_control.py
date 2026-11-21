import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness import run_control as rc


class RunControlTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(tmp)
        self.path = self.root / 'state.json'
        self.enterContext(patch.object(rc, 'boot_id', return_value='boot-A'))
        self.enterContext(patch.object(rc.bg, 'RUNS', tmp))

    def make(self):
        return rc.initialize(self.path, 'AR-test', 300, 60, 12, 3, 1, 'Inspect published evidence')

    def test_persistent_clock_and_call_reserve(self):
        with patch.object(rc.time, 'monotonic', return_value=100):
            state = self.make()
        with patch.object(rc.time, 'monotonic', return_value=200):
            self.assertEqual(rc.dispatch_left(rc.load(self.path)), 140)
            rc.charge(self.path, state, 8)
            self.assertEqual(rc.dispatch_left(rc.load(self.path)), 0)
            self.assertEqual(rc.summary(rc.load(self.path))['hard_seconds'], 200)
            with self.assertRaisesRegex(ValueError, 'resume'):
                self.make()
        with patch.object(rc, 'boot_id', return_value='boot-B'):
            with self.assertRaisesRegex(ValueError, 'rebooted'):
                rc.load(self.path)

    def test_wait_stops_at_dispatch_cutoff(self):
        with patch.object(rc.time, 'monotonic', return_value=100):
            state = self.make()
        state['jobs'] = ['live']
        (self.root / 'live').mkdir()
        (self.root / 'live/cmd').touch()
        clock = [338.5]
        with patch.object(rc.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(
            rc.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        ) as delay:
            rc.wait(state, 900)
        self.assertEqual(clock[0], 340.)
        self.assertEqual(delay.call_count, 2)

    def test_launch_uses_one_snapshot_and_saves_intent_first(self):
        state = self.make()
        example = dict(name='probe', cpu='7', seconds=120, argv=['python', '-m', 'tests'])
        with patch.object(os, 'sched_getaffinity', return_value={7, 8}), patch.object(
            rc, 'snapshot', return_value=('tree-id', '/snap')
        ) as snap, patch.object(rc.bg, 'start') as start:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                rc.launch(self.path, state, [example, dict(example, name='other')])
            with self.assertRaisesRegex(ValueError, 'insufficient'):
                rc.launch(self.path, state, [dict(example, seconds=600)])
            start.side_effect = lambda *args: self.assertEqual(rc.load(self.path)['jobs'], ['probe', 'other'])
            rc.launch(self.path, state, [example, dict(example, name='other', cpu='8')])
        snap.assert_called_once()
        self.assertEqual(start.call_count, 2)
        start.assert_any_call('probe', ['timeout', '-k', '2', '120', 'python', '-m', 'tests'], '7', 'tree-id')
        self.assertEqual(rc.summary(state)['tree'], 'tree-id')

    def test_close_disables_dispatch_but_keeps_checkpoint(self):
        self.make()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rc.main(['--state', str(self.path), '--used', '2', 'checkpoint', '--close', '--next', 'Report verdict'])
        result = json.loads(output.getvalue())
        self.assertEqual((result['phase'], result['dispatch_seconds'], result['calls_used']), ('closed', 0, 3))
        self.assertEqual(rc.load(self.path)['next'], 'Report verdict')

    def test_active_state_context_budget(self):
        state = rc.ROOT / 'STATE.md'
        # Immutable development-only snapshots intentionally omit root docs.
        if state.exists():
            self.assertLessEqual(state.stat().st_size, 2048)
