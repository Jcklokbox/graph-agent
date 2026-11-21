"""Development diagnostic: split readout, history and place routing; never use on AR-5 holdout."""
from __future__ import annotations
import argparse
import os
import numpy as np
from models.factor import CANONICAL, make_factor
from envs.phenomena import SETS, US, VALUE
from harness.journal import append


def episode(row, grp, seed, particles):
    table, n, _ = SETS['v1']
    s = table[row].build(grp, seed)
    f = make_factor(f'{CANONICAL}-N{particles}', n, 2, VALUE, seed)
    events = {(i-1, tag): x for i, tag, x in s.probes}
    for t,(x,y) in enumerate(zip(s.obs,s.obs[1:])):
        f.learn(x, 0, y)
        for (ti, tag), probe in events.items():
            if t != ti:
                continue
            b = f._bits(probe)
            k = min(f.K,int(f.used.max())+1)
            p = f._signature(b)
            prior = f._regime_prior(p,k)
            priorplus = f._cue_prior(prior, b,k) if hasattr(f,'_cue_prior') else prior
            mem = f._members(b,0,f.hist,k)
            local, bg = f._conditional(mem,k,p,not b[~f.slow].any())
            j = f.ch.index(US)
            w = np.exp(f.logw-f.logw.max()); w /= w.sum()
            mark = f.places.get(p)
            yield tag, dict(t=t+1, actual=float(f.predict(probe,0,hypothetical=True)[US]),
                            signature=list(p), learned=list(f.places), role=np.flatnonzero(f.slow).tolist(),
                            h=float(w@f.hz), anchor=mark[0] if mark else None,
                            prior=(w[:,None]*prior).sum(0).tolist(),
                            cue_prior=(w[:,None]*priorplus).sum(0).tolist(),
                            contribution=(w[:,None]*priorplus*local[:,:,j]).sum(0).tolist(),
                            baseline=float((w[:,None]*priorplus*bg[:,:,j]).sum()),
                            local_weight=float(f.expert[j]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out');p.add_argument('row');p.add_argument('--particles',type=int,default=50)
    a=p.parse_args()
    for group in SETS['v1'][0][a.row].groups:
        for tag, m in episode(a.row, group, 0, a.particles):
            append(a.out, {'k':[os.environ['PG_TREE'],0,f'{a.row}|{group}|{tag}'], 'm':m})
            print(a.row,group,tag,round(m['actual'],3),m['signature'],m['role'],round(m['h'],4),m['anchor'],flush=True)


if __name__=='__main__':
    from harness.beacon import cli

    with cli():
        main()
