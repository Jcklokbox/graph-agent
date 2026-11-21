"""Protocol tests on synthetic records; no fresh environments are opened."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from eval.paired import episode_key, normalized_payoff
from harness.journal import append, load_strict
from harness.snap import program_digest
from research.ar7_gate import BATCH, SEEDS, VARIANTS_LOCKED, main, packs, score, stage_key
from research.spent_cohorts import AR6_DEV20


class AR7AuditTests(unittest.TestCase):
    def setup(self, delta):
        c=dict(tree='test',models=['v12','v60'],model='v60',caps={'v12':180,'v60':130},cpus=[0,1]);r={};s={}
        for m in c['models']:
            for i,b in enumerate(packs(m)):
                s[stage_key('test',m,i)]=dict(dt=5,m=dict(exit=0,cap=c['caps'][m],cpus=c['cpus'],program=program_digest(b)))
                for v,j,_,_ in b:r[episode_key('test',m,v,j)]=dict(m=dict(brain=.60+(m=='v60')*delta,stay=.50,oracle=.75))
        return c,r,s

    def test_block_identity(self):
        self.assertEqual(len(AR6_DEV20), 20)
        self.assertTrue(AR6_DEV20.isdisjoint(SEEDS))
        self.assertEqual(
            program_digest({m: packs(m) for m in ('v12', 'v60')}),
            '7f04badee1c825d7474ce20fe863ffd2f67243bf4b615a49b945d3c59d2838b3',
        )
        target={(v,s) for v in VARIANTS_LOCKED for s in SEEDS}
        for m in ('v12','v60'):
            p=packs(m);self.assertEqual(len(p),30);self.assertTrue(all(len(b)==BATCH for b in p))
            self.assertEqual({(v,j) for b in p for v,j,_,_ in b},target)

    def test_cli_cannot_reopen_the_completed_cohort(self):
        with TemporaryDirectory() as tmp:
            for action in ('lock', 'run'):
                with self.subTest(action=action), redirect_stderr(StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as exc:
                        main([action, tmp])
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn('invalid choice', stderr.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_verdict_requires_every_episode_in_a_successful_stage(self):
        c,r,s=self.setup(0.);self.assertEqual(score(c,r,s)['status'],'PASS')
        self.assertEqual(set(score(c,r,s)['per_cell'].values()),{20})
        s[next(iter(s))]['m']['exit']=124
        z=score(c,r,s);self.assertEqual(z['status'],'INCOMPLETE');self.assertNotIn('ci99',z)
        c,r,s=self.setup(-.01);self.assertEqual(score(c,r,s)['status'],'FAIL')
        r.pop(next(iter(r)))
        with self.assertRaises(ValueError):score(c,r,s)

    def test_verdict_rejects_unmatched_payoffs_and_wrong_stage_witness(self):
        cfg, rows, stages = self.setup(0.)
        first = next(iter(rows))
        rows[first]['m']['oracle'] = .74
        with self.assertRaisesRegex(ValueError, 'unmatched denominators'):
            score(cfg, rows, stages)
        rows[first]['m']['oracle'] = .75
        stages[next(iter(stages))]['m']['program'] = 'wrong-program'
        with self.assertRaisesRegex(ValueError, 'wrong stage witness'):
            score(cfg, rows, stages)


class PairedEvidenceTests(unittest.TestCase):
    def test_qualification_journal_rejects_duplicates_and_bad_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episodes.jsonl'
            key = episode_key('tree', 'v60', 'aut', 1)
            entry = dict(k=key, m=dict(brain=.6, stay=.5, oracle=.75))
            append(path, entry)
            self.assertEqual(load_strict(path)[key]['m'], entry['m'])
            append(path, entry)
            with self.assertRaisesRegex(ValueError, 'duplicate journal'):
                load_strict(path)
            path.write_text('{"k":')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                load_strict(path)

    def test_normalization_rejects_nonfinite_bounds_and_nonpositive_gap(self):
        self.assertEqual(normalized_payoff(dict(brain=.75, stay=.5, oracle=1.)), .5)
        for triplet in ((float('nan'), .5, .75), (1.01, .5, .75),
                        (.6, .5, .5), (.6, .5, .49)):
            with self.subTest(triplet=triplet), self.assertRaises(ValueError):
                normalized_payoff(dict(zip(('brain', 'stay', 'oracle'), triplet)))
