import json
import numpy as np
from eval.phenomena import make
from envs.phenomena import SETS,US
from envs.family import WORLD_SETS,WorldEnv
from eval.predictive import make as pred

def state(a):
 if hasattr(a,'m'): return a.m.trace()
 w=a.w if hasattr(a,'w') else np.exp(a.logw-a.logw.max()); w=w/w.sum()
 return {k: (np.bincount(getattr(a,k),weights=w).tolist() if k=='c' else float(w@getattr(a,k))) for k in ('c','used','haz') if hasattr(a,k) and np.shape(getattr(a,k))==np.shape(w)}
for row in ('reinstatement','spont-recovery-sessions','cond-inhibition'):
 for g in SETS['v1'][0][row].groups:
  s=SETS['v1'][0][row].build(g,0); checks={i for i,_,_ in s.probes}|{i-15 for i,_,_ in s.probes}
  for name in ('factor-L4-P-C','lc-sub-add-n200'):
   a=make(name,0,10)
   for i,(x,y) in enumerate(zip(s.obs,s.obs[1:]),1):
    a.step(x);a.learn(y)
    if i in checks:
     pr={nm:a.predict_us(z) for ix,nm,z in s.probes if ix==i}; print(json.dumps([row,g,name,i,pr,state(a)]),flush=True)
w=WORLD_SETS['d3']()['d3-and']; env=WorldEnv(w,1000); pol=np.random.default_rng(2000); x=env.reset(); ags=[pred(n,w.n_channels,w.n_actions,w.value_index,0,w) for n in ('factor-L4-P-C','decompw')]; loss=np.zeros((2,w.n_channels))
for t in range(1000):
 a=int(pol.integers(w.n_actions)); qs=np.array([ag.predict(x,a) for ag in ags]); y=np.array(env.step(a)); loss+=-np.log(np.where(y>.5,qs,1-qs));
 for ag in ags:ag.learn(x,a,y)
 if t in (0,199,499,999):print(json.dumps(['d3-and',t,qs.tolist(),(loss/(t+1)).tolist(),ags[0].trace(),ags[1].partition()]),flush=True)
 x=tuple(y)
