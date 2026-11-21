from models.factor import FactorModel
from tests.data.factor_ref import FactorModel as Ref
import numpy as np, time
rng=np.random.default_rng(42); n=10; v=7
m=FactorModel(n,2,v,N=200,seed=0,process=True,cond=True,k=5,lags=range(5),own_lags=range(3)); r=Ref(n,2,v,N=200,seed=0,process=True,cond=True,k=5,lags=range(5),own_lags=range(3)); x=np.zeros(n); err=0
for t in range(260):
 y=(rng.random(n)<.12).astype(float);y[4]=t<100;y[8]=t>=100; a=int(rng.integers(2)); p=m.predict(x,a);q=r.predict(x,a);err=max(err,np.max(abs(p-q)));m.learn(x,a,y);r.learn(x,a,y);x=y
print('max abs',err,flush=True); assert err<=1e-12
