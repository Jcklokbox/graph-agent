import unittest
from research import ar6_pair_gate as gate


class FreshPairTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(tree='test-tree',model='v53',cpus=[0,1])
        self.stages={(self.cfg['tree'],0,f'{m}|{lo}|{hi}'):
            dict(dt=5.,m=dict(cpus=[0,1],slice=[m,lo,hi],exit=0)) for m,lo,hi in gate.plan(self.cfg)}
        self.rows={gate.key(self.cfg['tree'],m,v,i):dict(m=dict(brain=.7,stay=.5,oracle=.9))
                   for v,i,_,_ in gate.jobs() for m in ('v12','v53')}

    def test_all_300_seed_variant_pairs_required(self):
        r=gate.score(self.cfg,self.rows,self.stages)
        self.assertEqual((r['status'],r['verified'],r['stages']),('PASS',600,76))
        self.assertEqual(r['ci99'],[0.,0.])
        first=next(iter(self.stages));self.stages[first]['m']['exit']=124
        self.assertEqual(gate.score(self.cfg,self.rows,self.stages)['status'],'INCOMPLETE')

    def test_bad_pair_or_missing_successful_record(self):
        k=next(iter(self.rows));self.rows[k]['m']['oracle']=.89
        with self.assertRaisesRegex(ValueError,'denominators'):
            gate.score(self.cfg,self.rows,self.stages)
        del self.rows[k]
        with self.assertRaisesRegex(ValueError,'missing rows'):
            gate.score(self.cfg,self.rows,self.stages)

    def test_new_namespace_and_negative_gap(self):
        self.assertTrue(all(s>100_000 for s in gate.SEEDS))
        for (_,_,key),row in self.rows.items():
            if key.startswith('v53|'):
                row['m']['brain']=.68
        r=gate.score(self.cfg,self.rows,self.stages)
        self.assertEqual(r['status'],'FAIL')
        self.assertAlmostEqual(r['paired'],-.05)
