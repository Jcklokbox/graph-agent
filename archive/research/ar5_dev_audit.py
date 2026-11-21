"""Independent AR-5 DEV journal verifier; run with development on PYTHONPATH."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from envs.tasks import SPLITS,VARIANTS
from harness.journal import load

SHA='f34b36b38386bd88ebdeca104d90fbcc49b0cab715e91aa71227c73f32ce7370'
T=2.946714
LIMIT=-.02


def audit(root):
    freeze=json.loads((root/'freeze.json').read_text())
    tree,model=freeze['tree'],freeze['model']
    cache=(root/'reference.json').read_bytes()
    if hashlib.sha256(cache).hexdigest()!=SHA or freeze['reference_sha256']!=SHA:
        raise ValueError('reference changed')
    reference=json.loads(cache)
    variants,n,eb,bb=SPLITS['dev']
    seeds=list(range(16))
    jobs=[(v,s,eb+s,bb+s) for s in seeds for v in variants]
    names=[(tree,s,f'{model}|DEV|{v}') for v,s,_,_ in jobs]
    work=sum(VARIANTS[v].steps for v,_,_,_ in jobs)
    if (freeze['seeds']!=seeds or freeze['pairs']!=len(jobs) or freeze['steps']!=work
        or work!=177600 or len(jobs)!=240 or len(set(freeze['cpus']))!=2
        or hashlib.sha256(repr(jobs).encode()).hexdigest()!=freeze['schedule_sha256']):
        raise ValueError('frozen plan changed')
    cand,dup,bad=load(root/'candidate.jsonl');ledger,d1,b1=load(root/'stages.jsonl')
    if any((dup,bad,d1,b1)) or not cand.keys()<=set(names):
        raise ValueError('duplicate, foreign or damaged records')
    covered=0.;time=0.;max_stage=0.;stop_seen=False
    for k,r in ledger.items():
        lo,hi=r['m']['bounds'];dt=r['dt']
        if (stop_seen or lo!=covered or hi!=min(len(names),lo+freeze['batch'])
            or k!=(tree,0,f'DEV|{lo}|{hi}') or not 0<=dt<=168.5
            or r['m']['cpus']!=','.join(map(str,freeze['cpus']))):
            raise ValueError('invalid stage order or time cap')
        time+=dt;max_stage=max(max_stage,dt)
        if r['m']['exit']:
            stop_seen=True;continue
        if not set(names[lo:hi])<=cand.keys():
            raise ValueError('stage claims missing records')
        covered=hi
    status=dict(tree=tree,model=model,commit=freeze['commit'],observed=len(cand),
                ledger_complete=int(covered),expected=len(names),stages=len(ledger),
                seconds=time,max_stage_s=max_stage,status='INCOMPLETE')
    if covered!=len(names) or len(cand)!=len(names):
        return status
    blocks=[];raw=[];base=[]
    for s in seeds:
        br=[];rr=[]
        for v in variants:
            b=reference['variants'][v];r=b['raw'][s]
            if b['seeds'][s]!=s:
                raise ValueError('reference seed order changed')
            o=cand[tree,s,f'{model}|DEV|{v}']['m'];g=r['oracle']-r['stay']
            if g<=0 or any(abs(r[k]-o[k])>1e-12 for k in ('oracle','stay')):
                raise ValueError('paired environment denominator mismatch')
            x=(o['brain']-r['stay'])/g;y=(r['brain']-r['stay'])/g
            if not all(map(math.isfinite,(x,y))) or abs(y-b['per_seed'][s])>1e-12:
                raise ValueError('nonfinite or inconsistent reference normalization')
            br.append(x);rr.append(y)
        blocks.append(st.mean(x-y for x,y in zip(br,rr)));raw.extend(br);base.extend(rr)
    avg=st.mean(blocks);half=T*st.stdev(blocks)/math.sqrt(len(seeds))
    status.update(candidate=st.mean(raw),v12=st.mean(base),paired=avg,ci99=[avg-half,avg+half],
                  block_effects=blocks,margin=LIMIT,status='PASS' if avg-half>=LIMIT else 'FAIL')
    if (root/'DEV.json').exists():
        verdict=json.loads((root/'DEV.json').read_text())
        if status['status']!=verdict['status'] or any(abs(status[k]-verdict[k])>1e-12 for k in ('paired','candidate','v12')):
            raise ValueError('official verdict differs from independent arithmetic')
        if any(abs(x-y)>1e-12 for x,y in zip(status['ci99'],verdict['ci99'])):
            raise ValueError('official CI differs from independent arithmetic')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.root),indent=2,sort_keys=True))
