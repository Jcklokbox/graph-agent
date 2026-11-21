"""Translation and fixed-sample evidence tests; no predictor runs on the holdout."""
import json
from pathlib import Path
import tempfile
import unittest

from envs.phenomena import A, B, X, Y, CTX1, CTX2, HOME, US, VALUE
from harness.journal import append
from research import ar5_holdout as ho
from research.ar5_judge import evidence, plan, report
from research.ar5_dev_protocol import paired_score


class HoldoutProtocolTests(unittest.TestCase):
    def test_background_changes_location_not_free_outcome_tape(self):
        same, other = (ho.background(g, ho.SEEDS[0]) for g in ('same','other'))
        self.assertEqual(len(same.obs), len(other.obs))
        self.assertEqual([o[US] for o in same.obs], [o[US] for o in other.obs])
        self.assertEqual([o[A] for o in same.obs], [o[A] for o in other.obs])
        self.assertEqual(same.probes, other.probes)
        i = next(i for i,(x,y) in enumerate(zip(same.obs,other.obs)) if x!=y)
        self.assertEqual((same.obs[i][CTX1], other.obs[i][CTX2]), (1.,1.))
        self.assertTrue(all(p[2][HOME] for p in same.probes))

    def test_associate_crossover_balances_other_history(self):
        trial0 = ho.mediated('linked', ho.SEEDS[0]); trial1=ho.mediated('alternate',ho.SEEDS[0])
        self.assertEqual(len(trial0.obs),len(trial1.obs))
        self.assertEqual([o[US] for o in trial0.obs], [o[US] for o in trial1.obs])
        self.assertEqual([o[A]+o[Y] for o in trial0.obs], [o[A]+o[Y] for o in trial1.obs])
        self.assertEqual([o[B]+o[X] for o in trial0.obs], [o[B]+o[X] for o in trial1.obs])
        self.assertEqual(trial0.probes,trial1.probes)

    def test_omission_control_matches_elapsed_time_and_observed_outcome(self):
        for aq in ('full','partial'):
            om=ho.omission_vs_dwell(aq+'-omit',ho.SEEDS[0]); dw=ho.omission_vs_dwell(aq+'-dwell',ho.SEEDS[0])
            self.assertEqual(len(om.obs),len(dw.obs))
            self.assertEqual(sum(x[A] for x in om.obs)-sum(x[A] for x in dw.obs),8.)
            self.assertEqual([o[US] for o in om.obs], [o[US] for o in dw.obs])
            self.assertTrue(all(x[VALUE]==0. for x in om.obs))

    def test_incomplete_and_invalid_keys_are_not_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            self.assertEqual(report(p,'test','holdout')['status'],'INCOMPLETE')
            append(p/'C.jsonl', {'k':['bad-tree',0,'factor-L4-P-C|C|s-2x3'],'m':{'sec':1}})
            with self.assertRaises(ValueError): evidence(p,'test','C')
        self.assertEqual(len(plan('holdout')),24*12)

    def test_paired_v12_reference_self_comparison_uses_all_seed_blocks(self):
        # The frozen reference is shipped as evaluation evidence, not in the
        # code snapshot. Missing archived baseline means skip, not silent PASS.
        cache=Path(__file__).resolve().parents[1]/'results/dev16_v12.json'
        if not cache.is_file():
            self.skipTest('DEV reference kept external to predictor snapshots')
        ref=json.loads(cache.read_text()); out={}
        for v,c in ref['variants'].items():
            for i,r in zip(c['seeds'],c['raw']):
                out['h',i,'agent|DEV|'+v]={'m':r}
        r=paired_score(ref,out,'h','agent')
        self.assertEqual(r['status'],'PASS'); self.assertAlmostEqual(r['paired'],0.)
        self.assertEqual(r['n_blocks'],16); self.assertAlmostEqual(r['v12'],.8912708030208981)

    def test_two_step_reference_prefers_informative_act_without_double_bonus(self):
        from research.ar5_voi_fixture import observe_then_choose, dirichlet_mean_moments
        (q_info, voi),(q_idle, noinfo) = observe_then_choose()
        self.assertAlmostEqual(q_info,.855); self.assertAlmostEqual(voi,.4)
        self.assertAlmostEqual(q_idle,.675); self.assertEqual(noinfo,0.)
        m, ep, obs = dirichlet_mean_moments(4, 1.8, .81)
        self.assertAlmostEqual(m,.46); self.assertAlmostEqual(ep*6,obs)

    def test_new_registry_distinct_and_unopened_before_credit(self):
        import os
        from unittest.mock import patch
        from research.ar5_holdout_next import SEEDS
        from research.ar5_judge import NEXT, run_batch
        manifest={'tree':'fresh','seeds':list(SEEDS),'predictor_development_closed':True,
                  'holdout_suite':NEXT}
        self.assertFalse(set(SEEDS).intersection(ho.SEEDS))
        self.assertEqual(len(plan(NEXT)), 192); self.assertEqual(len(plan('credit')), 24)
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,PG_TREE='fresh'):
            with self.assertRaisesRegex(RuntimeError,'not authorized'):
                run_batch(Path(td),manifest,'holdout',0,1,1)
            with patch('research.ar5_judge.report',side_effect=lambda p,t,s:
                       {'status':'FAIL' if s=='credit' else 'PASS'}):
                with self.assertRaisesRegex(RuntimeError,'locked by credit'):
                    run_batch(Path(td),manifest,NEXT,0,1,1)
            self.assertFalse((Path(td)/(NEXT+'.jsonl')).exists())
