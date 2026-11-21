"""Short dispatch checks without running a research cohort."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval import benchmark, paired, screen
from eval.parallel import pmap


class TinyWorld:
    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        return (self.seed,)

    def step(self, action):
        return (action,), action / 4., False, {'received_shock': action == 0}


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.first = SimpleNamespace(offset=0)
        self.second = SimpleNamespace(offset=1)
        for module in (self.first, self.second):
            module.Agent = self.agent_class(module)
        variant = SimpleNamespace(make=TinyWorld, n_inputs=12, n_actions=5,
                                  steps=3, oracle=lambda: lambda _: 4)
        self.enterContext(patch.dict(benchmark.VARIANTS, {'aut': variant}))
        self.enterContext(patch.object(benchmark, '_window', return_value=(0, 3)))
        self.enterContext(patch('agents.load', side_effect={'first': self.first,
                                                          'second': self.second}.__getitem__))

    @staticmethod
    def agent_class(module):
        class Agent:
            def __init__(self, n_inputs, n_actions, value_index, *, seed, bias=0):
                self.action = (seed + module.offset + bias) % n_actions
                self.seen = []

            def step(self, obs):
                self.action = (self.action + obs[0]) % 5 if not self.seen else self.action
                return self.action

            def learn(self, obs):
                assert obs == (self.action,)
                self.seen.append(obs)

        return Agent

    def test_mixed_jobs_keep_their_model_settings_and_seeds(self):
        jobs = [('aut', 1, 1, 'first', {'bias': 1}),
                ('aut', 2, 1, 'second', {}),
                ('aut', 1, 1, 'first', {}),
                ('aut', 1, 2, 'first', {})]
        for workers in ('1', '2'):
            with self.subTest(workers=workers), patch.dict('os.environ', PG_PROCS=workers):
                rows = pmap(benchmark.brain_run, jobs)
            self.assertEqual([row['brain'] for row in rows], [.75, 1., .5, .75])
            self.assertTrue(all(row['stay'] == .25 and row['oracle'] == 1. for row in rows))

    def test_benchmark_cli_preserves_metadata_and_scopes_module_overrides(self):
        with TemporaryDirectory() as tmp, patch.dict('os.environ', PG_PROCS='2'), redirect_stdout(StringIO()):
            out = Path(tmp) / 'report.json'
            benchmark.main(['--agent', 'first', '--kw', '{"bias": 1}',
                            '--set', 'offset=2', '--set', 'new_option=true',
                            '--seeds', '2', '--out', str(out), 'aut'])
            report = json.loads(out.read_text())
            self.assertEqual(report['agent'], 'first')
            self.assertEqual(report['settings'], {'bias': 1})
            self.assertEqual(report['module_sets'], {'offset': 2, 'new_option': True})
            _, _, eb, bb = benchmark.SPLITS['dev']
            raw = report['variants']['aut']['raw']
            self.assertEqual([row['brain'] for row in raw],
                             [((eb + bb + 2*s + 3) % 5) / 4 for s in range(2)])
        self.assertEqual(self.first.offset, 0)
        self.assertFalse(hasattr(self.first, 'new_option'))

    def test_module_overrides_are_restored_when_dispatch_fails(self):
        with patch.object(benchmark, 'pmap', side_effect=RuntimeError('dispatch failed')):
            with self.assertRaisesRegex(RuntimeError, 'dispatch failed'):
                benchmark.run('dev', ['aut'], agent='first', module_sets={'offset': 2, 'new_option': True})
        self.assertEqual(self.first.offset, 0)
        self.assertFalse(hasattr(self.first, 'new_option'))

    def test_paired_record_keeps_tree_key_and_raw_fields(self):
        row = paired.run_episode(('second', 'aut', 7, 2, 1), tree='test-tree')
        self.assertEqual(row['k'], ['test-tree', 7, 'second|DEV|aut'])
        self.assertEqual(set(row['m']), {'brain', 'stay', 'oracle', 'shocks', 'late_shocks', 'sec'})
        self.assertEqual(row['m']['brain'], 1.)

    def test_screen_passes_one_configuration_to_dispatch_and_output(self):
        with patch.object(screen, 'live') as live, redirect_stdout(StringIO()):
            screen.main(['--agent', 'second', '--kw', '{"bias": 2}', 'aut'])
        self.assertEqual(live.call_args.kwargs, {'agent': 'second', 'settings': {'bias': 2}})
        self.assertEqual(live.call_args.args[-1]['agent'], 'second')
        self.assertEqual(live.call_args.args[-1]['settings'], {'bias': 2})
