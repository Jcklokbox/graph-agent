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
        lines = output.getvalue().splitlines()
        self.assertIn('[STATE & CONTRACT BEACON]', lines[0])
        self.assertIn('English only', lines[-1])
        result = json.loads(lines[1])
        self.assertEqual((result['phase'], result['dispatch_seconds'], result['calls_used']), ('closed', 0, 3))
        self.assertEqual(rc.load(self.path)['next'], 'Report verdict')

    def test_active_state_context_budget(self):
        state = rc.ROOT / 'STATE.md'
        # Immutable development-only snapshots intentionally omit root docs.
        if state.exists():
            self.assertLessEqual(state.stat().st_size, 2048)

    def test_publish_and_restore_preserves_elapsed_and_blocks_dispatch(self):
        with patch.object(rc.time, 'monotonic', return_value=100), patch.object(rc.time, 'time', return_value=1000):
            initial = self.make()
        published = self.root / 'published.json'
        rc.save(published, initial)
        self.path.unlink()
        with patch.object(rc, 'boot_id', return_value='boot-B'), patch.object(
            rc.time, 'monotonic', return_value=25
        ), patch.object(rc.time, 'time', return_value=1100):
            recovered = rc.recover(self.path, published)
            self.assertEqual((recovered['phase'], recovered['calls'], recovered['hard']), ('reconcile', 1, 225))
            self.assertEqual(rc.dispatch_left(recovered), 0)
            self.assertEqual(rc.load(self.path), recovered)

    def test_reflection_blocks_next_dispatch_without_resetting_clock(self):
        with patch.object(rc.time, 'monotonic', return_value=100):
            state = rc.initialize(self.path, 'cadence', 3600, 120, 240, 20, 219, 'Try test')
        with patch.object(rc.time, 'monotonic', return_value=1920):
            self.assertEqual(rc.dispatch_left(state), 0)
            self.assertTrue(rc.summary(state)['reflection_due'])
            rc.reflect(self.path, state, 'Locus', 'L-7', 'Run a micro-probe', 'Inspect trace')
            self.assertEqual(state['reflection'], 3700)
            self.assertGreater(rc.dispatch_left(state), 0)
            rc.charge(self.path, state, 1)
            self.assertEqual(rc.dispatch_left(state), 0)
            with self.assertRaisesRegex(ValueError, 'hard call ceiling'):
                rc.charge(self.path, state, 21)
            self.assertEqual(rc.load(self.path)['calls'], 241)

    def test_beacon_keeps_machine_output_clean(self):
        with patch.dict(os.environ, {rc.beacon.MACHINE: '1'}), contextlib.redirect_stdout(io.StringIO()) as out:
            with rc.beacon.cli(self.path):
                print('{"value": 1}')
        self.assertEqual(json.loads(out.getvalue()), {'value': 1})

    def test_state_markdown_managed_block_only(self):
        prior = self.root / 'STATE.md'
        prior.write_text('Header\n<!-- managed-next -->\nOld\n<!-- managed-next -->\nFooter\n')
        with patch.object(rc, 'ROOT', self.root):
            rc.update_state_md(self.make())
        self.assertIn('**Run AR-test (active):** Inspect published evidence', prior.read_text())
        self.assertTrue(prior.read_text().startswith('Header\n'))
        self.assertTrue(prior.read_text().endswith('\nFooter\n'))
