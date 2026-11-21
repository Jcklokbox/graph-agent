"""Protocol tests on synthetic records; no fresh environments are opened."""
import unittest

from research.ar6_pair_gate import SEEDS as SPENT
from research.ar7_gate import (BATCH, SEEDS, VARIANTS_LOCKED, cal_packs, digest,
                               key, packs, score, stage_key)


class FreshGateTests(unittest.TestCase):
    def setup(self, delta):
        c=dict(tree='test',models=['v12','v60'],model='v60',caps={'v12':180,'v60':130},cpus=[0,1]);r={};s={}
        for m in c['models']:
            for i,b in enumerate(packs(m)):
                s[stage_key('test',m,i)]=dict(dt=5,m=dict(exit=0,cap=c['caps'][m],cpus=c['cpus'],program=digest(b)))
                for v,j,_,_ in b:r[key('test',m,v,j)]=dict(m=dict(brain=.60+(m=='v60')*delta,stay=.50,oracle=.75))
        return c,r,s

    def test_block_identity(self):
        self.assertTrue(set(SEEDS).isdisjoint(SPENT))
        target={(v,s) for v in VARIANTS_LOCKED for s in SEEDS}
        for m in ('v12','v60'):
            p=packs(m);self.assertEqual(len(p),30);self.assertTrue(all(len(b)==BATCH for b in p))
            self.assertEqual({(v,j) for b in p for v,j,_,_ in b},target)
            self.assertEqual({tuple(v for v,*_ in b) for b in p},
                             {tuple(v for v,*_ in b) for b in cal_packs(m)})
            self.assertTrue(all(j in range(16) for b in cal_packs(m) for _,j,_,_ in b))

    def test_verdict_requires_every_episode_in_a_successful_stage(self):
        c,r,s=self.setup(0.);self.assertEqual(score(c,r,s)['status'],'PASS')
        self.assertEqual(set(score(c,r,s)['per_cell'].values()),{20})
        s[next(iter(s))]['m']['exit']=124
        z=score(c,r,s);self.assertEqual(z['status'],'INCOMPLETE');self.assertNotIn('ci99',z)
        c,r,s=self.setup(-.01);self.assertEqual(score(c,r,s)['status'],'FAIL')
        r.pop(next(iter(r)))
        with self.assertRaises(ValueError):score(c,r,s)
