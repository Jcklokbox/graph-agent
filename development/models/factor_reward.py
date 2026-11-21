"""Bounded real outcome member of the factor model (AR-5 D-66).

Its conditional mean has Gaussian coefficients over a bounded, evidence-selected
local lag map. Actual observations are real numbers, not Bernoulli successes.
A bounded three-point forecast matches the first two reward moments for the
small posterior-control calculation. The Gaussian inference is a projection.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


class BoundedReward:
    def __init__(self, channels, actions, lag, cap=7):
        self.J, self.A, self.L = channels, actions, lag + 1
        self.k = min(cap, channels * self.L)
        self.slots = np.full(self.k, -1, int)
        self.pairs = np.array(list(combinations(range(self.k), 2)), int).reshape(-1, 2)
        d = 1 + self.k + len(self.pairs)
        self.mean = np.zeros((actions, d))
        self.mean[:, 0] = .5
        self.cov = np.broadcast_to(np.eye(d), (actions,d,d)).copy()
        self.n = np.zeros(actions)
        self.sse = np.zeros(actions)
        self.xsum = np.zeros((actions, self.L*channels))
        self.yx = np.zeros_like(self.xsum)
        self.ex = np.zeros_like(self.xsum)
        self.ysum = np.zeros(actions)
        self.esum = np.zeros(actions)
        self._zero = np.zeros(channels)

    def stack(self, b, hist):
        if b.ndim == 1:
            return np.concatenate([b,*[hist[-l] if len(hist)>=l else self._zero for l in range(1,self.L)]])
        prior = np.concatenate([hist[-l] if len(hist)>=l else self._zero for l in range(1,self.L)]) if self.L>1 else np.empty(0)
        return np.concatenate((b,np.broadcast_to(prior,(len(b),len(prior)))),axis=1)

    def features(self, b, hist):
        # Gather only the bounded reward support on hypothetical branches.
        u = np.zeros((*b.shape[:-1],self.k))
        for p, idx in enumerate(self.slots):
            if idx >= 0:
                l,j = divmod(int(idx),self.J)
                if l==0:
                    u[...,p] = b[...,j]
                elif l<=len(hist):
                    u[...,p] = hist[-l][j]
        return np.concatenate((np.ones((*b.shape[:-1],1)), u, u[...,self.pairs[:,0]]*u[...,self.pairs[:,1]]),axis=-1)

    @property
    def noise(self):
        return np.maximum(1e-8, (.25 + self.sse) / (1. + self.n))

    def posterior(self, f, a):
        raw = self.mean[a] @ f
        covf = self.cov[a] @ f
        var = max(0., float(f @ covf))
        noise = self.noise[a]
        return raw, covf / (noise + var), var + noise

    @staticmethod
    def distribution(raw, variance):
        """Masses at 0, E[R], 1 that match bounded first and second moments."""
        m = float(np.clip(raw,0.,1.))
        bound = m*(1.-m)
        frac = min(1., variance / bound) if bound > 0. else 0.
        return np.array([0.,m,1.]), np.array([frac*(1.-m), 1.-frac, frac*m])

    def root(self,b,hist,a,theta=None):
        f=self.features(b,hist)
        raw, gain, var = self.posterior(f,a)
        if theta is not None:
            raw = theta[a] @ f
            var = self.noise[a]  # epistemic spread drawn once, not twice in the branch
        vals, mass = self.distribution(raw,var)
        return f,raw,gain,vals,mass

    def utility(self,root,a,successor,b,hist,theta=None):
        """E[R']+E[max E[R''|successor, previous a and R']].

        The virtual posterior shift is rank one. No live parameter, RNG or
        stored feature changes during a branch, and no second VOI bonus.
        """
        _,raw,gain,outcomes,mass=root
        u = self.features(successor,(*hist,b))
        next_values = u @ (self.mean if theta is None else theta).T
        # The previous action determines which posterior may be updated.
        surface = np.broadcast_to(next_values[:,None,:],(len(u),len(mass),self.A)).copy()
        surface[:,:,a] += (u @ gain)[:,None] * (outcomes[None,:] - raw)
        follow = np.clip(surface,0.,1.).max(2).mean(0)
        return float(mass @ (outcomes + follow))

    def learn(self,b,hist,a,value):
        y=float(value)
        if not np.isfinite(y) or not 0. <= y <= 1.:
            raise ValueError('bounded real reward member expects a finite observation in [0,1]')
        f = self.features(b,hist)
        raw,gain,_ = self.posterior(f,a)
        cf = self.cov[a] @ f
        err=y-raw
        self.mean[a] += gain*err
        self.cov[a] -= np.outer(gain,cf)
        self.n[a] += 1.
        self.sse[a] += (y-float(np.clip(raw,0.,1.)))**2
        x=self.stack(b,hist)
        self.xsum[a]+=x; self.yx[a]+=y*x; self.ex[a]+=err*x
        self.ysum[a]+=y; self.esum[a]+=err
        self._admit()

    def _admit(self):
        n = np.maximum(1.,self.n)[:,None]
        center = self.xsum/n
        cov_raw = self.yx - self.ysum[:,None]*center
        cov_err = self.ex - self.esum[:,None]*center
        scale = (1.+self.xsum*(1.-center))*self.noise[:,None]
        score = (np.maximum(cov_raw**2,cov_err**2)/scale).sum(0)
        if not self.k:
            return
        priority=score.copy();priority[self.slots[self.slots>=0]]=-np.inf
        loc = int(np.argmax(priority))
        old = np.where(self.slots>=0,score[np.maximum(self.slots,0)],0.)
        dest=int(np.argmin(old))
        # One log-density (BIC) unit to justify expanding or replacing a parent;
        # scale with observations, not an episode-age rule or task cutoff.
        charge=np.log1p(self.J*self.L)
        if priority[loc] <= old[dest]+charge:
            return
        self.slots[dest]=loc
        pair = np.flatnonzero((self.pairs==dest).any(1))
        reset = np.r_[1+dest,1+self.k+pair]
        self.mean[:,reset]=0.
        self.cov[:,reset,:]=0.; self.cov[:,:,reset]=0.; self.cov[:,reset,reset]=1.


def first_success(product_probs, u):
    """Bounded-work first-on draw from an independent pattern given nonempty."""
    with np.errstate(divide='ignore'):
        cdf = -np.expm1(np.cumsum(np.log1p(-product_probs)))
    if not len(cdf) or cdf[-1] <= 0.:
        raise ValueError('nonempty condition has zero probability')
    return min(len(cdf)-1, int(np.searchsorted(cdf,u*cdf[-1],side='right')))


def conditional_draws(pr, probs, slow, slow_p, quiet, uniforms):
    """Sample the factor's product mixture, including its normalized quiet clock.

    Uniform array supplies component, clock, first-success, and J emissions.
    Its rows can be reused at alternate root actions as common random numbers.
    """
    J = len(slow)
    weights = np.asarray(pr,float).reshape(-1)
    flat = np.asarray(probs).reshape(-1,J)
    cdf = np.cumsum(weights)
    ind = np.minimum(len(cdf)-1,np.searchsorted(cdf,uniforms[:,0]*cdf[-1],side='right'))
    rows = flat[ind]
    draw = (uniforms[:,3:] < rows)
    if slow.any():
        draw[:,slow] = uniforms[:,3:][:,slow] < slow_p
    fast = ~slow
    if quiet and fast.any():
        idle_by = np.prod(1.-flat[:,fast],axis=1)
        idle = float(weights @ idle_by/cdf[-1])
        # A zero-mass nonempty component would have undefined conditioning.
        # With the factor's nondegenerate emissions this cannot occur for a
        # positive-weight component (we never select unused zero-mass rows).
        jj=np.flatnonzero(fast)
        for i,u in enumerate(uniforms):
            if u[1] < idle:
                draw[i,jj]=False
            else:
                first=first_success(rows[i,jj],u[2]); draw[i,jj[:first]]=False; draw[i,jj[first]]=True
    return draw.astype(float)


class DiagonalReward(BoundedReward):
    """Linear-width factorized Gaussian over every potential unary lag.

    A bounded slot bank selects the interaction terms. Inactive/unadmitted
    unary features can remain uncertain rather than being asserted irrelevant.
    A sequential Gaussian projection keeps diagonal covariance in O(J).
    """
    def __init__(self,channels,actions,lag,cap=7):
        super().__init__(channels,actions,lag,cap)
        d=1+self.J*self.L+len(self.pairs)
        self.mean=np.zeros((self.A,d)); self.mean[:,0]=.5
        self.cov=np.full((self.A,d),.25)

    def features(self,b,hist):
        base=self.stack(b,hist)
        u=base[...,np.maximum(0,self.slots)]*(self.slots>=0)
        return np.concatenate((np.ones((*b.shape[:-1],1)),base,u[...,self.pairs[:,0]]*u[...,self.pairs[:,1]]),axis=-1)

    def posterior(self,f,a):
        raw=self.mean[a] @ f
        covf=self.cov[a]*f
        var=max(0.,float(f @ covf)); noise=self.noise[a]
        return raw,covf/(noise+var),var+noise

    def sample(self,rng):
        return self.mean + np.sqrt(self.cov)*rng.standard_normal(self.mean.shape)

    def learn(self,b,hist,a,value):
        y=float(value)
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        f=self.features(b,hist)
        raw,gain,_=self.posterior(f,a)
        err=y-raw
        self.mean[a]+=gain*err
        self.cov[a]*=(1.-gain*f)
        self.n[a]+=1.; self.sse[a]+=(y-float(np.clip(raw,0.,1.)))**2
        x=f[1:1+self.L*self.J]
        self.xsum[a]+=x; self.yx[a]+=y*x; self.ex[a]+=err*x
        self.ysum[a]+=y; self.esum[a]+=err
        self._admit()

    def _admit(self):
        n=np.maximum(1.,self.n[:,None]); center=self.xsum/n
        raw=self.yx-self.ysum[:,None]*center
        err=self.ex-self.esum[:,None]*center
        scores=(np.maximum(raw**2,err**2)/((1.+self.xsum*(1.-center))*self.noise[:,None])).sum(0)
        priority=scores.copy(); priority[self.slots[self.slots>=0]]=-np.inf
        dest=int(np.argmin(np.where(self.slots>=0,scores[np.maximum(self.slots,0)],0.)))
        loc=int(np.argmax(priority)); old=self.slots[dest]
        # Single reusable expansion per step, regardless of input width.
        if priority[loc] <= (scores[old] if old>=0 else 0.)+np.log1p(self.J*self.L):
            return
        self.slots[dest]=loc
        reset=1+self.J*self.L+np.flatnonzero((self.pairs==dest).any(1))
        self.mean[:,reset]=0.; self.cov[:,reset]=.25


class OutcomeReward(DiagonalReward):
    """A shared value member conditioned on observable successor cues."""
    def __init__(self,channels):
        super().__init__(channels,1,0,cap=9)


class HybridReward(OutcomeReward):
    """Shared successor reward plus a signed proximal conditional residual."""
    def __init__(self,channels,actions,lag):
        super().__init__(channels)
        self.context=DiagonalReward(channels,actions,lag)
        self.context.mean.fill(0.)
        self.context.cov.fill(.25 / (1. + lag))
        self.context.cov[:,0]=0.

    def sample(self,rng):
        return super().sample(rng), self.context.sample(rng)

    def joint_posterior(self,v,u,a):
        sg=self.cov[0]*v; su=self.context.cov[a]*u
        var=float(sg@v+su@u+self.noise[0])
        raw=self.mean[0]@v+self.context.mean[a]@u
        return raw,sg/var,su/var,var

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        u=self.context.features(b,hist); v=self.features(bn,())
        raw,kg,ku,_=self.joint_posterior(v,u,a)
        err=y-raw
        self.mean[0]+=kg*err; self.context.mean[a]+=ku*err
        self.cov[0]*=1.-kg*v; self.context.cov[a]*=1.-ku*u
        squared=(y-float(np.clip(raw,0.,1.)))**2
        for h,f,i in ((self,v,0),(self.context,u,a)):
            h.n[i]+=1.; h.sse[i]+=squared
            x=f[1:1+h.J*h.L]; h.xsum[i]+=x
            h.yx[i]+=y*x; h.ex[i]+=err*x
            h.ysum[i]+=y; h.esum[i]+=err
            h._admit()


class CausalReward:
    """Observed controllable successor marks interact with past sensory features.

    The bounded control view is selected by action-split vs action-pooled
    likelihood, not sensor names. Reward parameters pool across actions which
    deliver the same controlled outcome. A learned Dirichlet control-transition
    factor backs off to the sensory model on untested local transitions.
    """
    def __init__(self,J,A,lag,cap=5):
        self.J,self.A,self.L=J,A,lag+1
        self.k=min(J,cap); self.slots=np.full(self.k,-1,int)
        self.binary=2**np.arange(self.k)
        self.base=np.zeros((J,2,2)); self.split=np.zeros((J,A,2,2))
        self.evidence=np.zeros(J)
        self.block=1+self.L*J
        d=1+self.k*self.block
        self.mean=np.zeros((1,d)); self.mean[:,0]=.5
        self.cov=np.full((1,d),.25/self.L); self.cov[:,0]=.25
        self.n=np.zeros(1); self.sse=np.zeros(1)
        self._zero=np.zeros(J)
        # A categorical transition operator on a bounded local control factor,
        # not action labels or a prerecorded transition/reward lookup.
        self.trans=np.zeros((A,2**self.k,2**self.k))
        self.trans_total=np.zeros((A,2**self.k))

    @property
    def noise(self):
        return np.maximum(1e-8,(.25+self.sse)/(1.+self.n))

    def features(self,y,b,hist):
        v=y[...,np.maximum(0,self.slots)]*(self.slots>=0)
        f=np.concatenate(([1.],b,*[hist[-l] if l<=len(hist) else self._zero for l in range(1,self.L)]))
        cross=(v[...,None]*f).reshape((*y.shape[:-1],self.k*self.block))
        return np.concatenate((np.ones((*y.shape[:-1],1)),cross),axis=-1)

    def posterior(self,f,a=0):
        cf=f*self.cov[0]; var=max(0.,float(cf@f))+self.noise[0]
        return self.mean[0]@f,cf/var,var

    distribution=staticmethod(BoundedReward.distribution)

    def sample(self,rng):
        return self.mean+np.sqrt(self.cov)*rng.standard_normal(self.mean.shape)

    def code(self,b):
        return ((b[...,np.maximum(self.slots,0)]>.5)*(self.slots>=0))@self.binary

    def refine_draws(self,b,a,y,uniforms):
        active=self.slots>=0
        if not active.any():
            return y
        code=int(self.code(b)); total=self.trans_total[a,code]
        if total<=0.:
            return y
        # Mixture with one prior pseudo-observation from the sensory member.
        u=uniforms[:,-1]*(1.+total)
        choose=u<total
        cdf=np.cumsum(self.trans[a,code])
        idx=np.minimum(len(cdf)-1,np.searchsorted(cdf,u[choose],side='right'))
        y[np.ix_(choose,self.slots[active])] = (idx[:,None]&self.binary[active]) != 0
        return y

    def learn_control(self,b,a,bn):
        j=np.arange(self.J); x=(b>.5).astype(int); z=(bn>.5).astype(int)
        pooled=self.base[j,x]; specific=self.split[j,a,x]
        l0=np.log((pooled[j,z]+.5)/(pooled.sum(1)+1.))
        l1=np.log((specific[j,z]+.5)/(specific.sum(1)+1.))
        self.evidence += l1-l0
        self.base[j,x,z]+=1.; self.split[j,a,x,z]+=1.
        dst=int(np.argmin(np.where(self.slots>=0,self.evidence[np.maximum(0,self.slots)],0.)))
        score=self.evidence.copy(); score[self.slots[self.slots>=0]]=-np.inf
        src=int(np.argmax(score))
        revised = score[src]>max(0.,self.evidence[self.slots[dst]] if self.slots[dst]>=0 else 0.)+np.log1p(self.J)
        if revised:
            self.slots[dst]=src
            part=slice(1+dst*self.block,1+(dst+1)*self.block)
            self.mean[:,part]=0.; self.cov[:,part]=.25/self.L
            self.trans.fill(0.);self.trans_total.fill(0.)
        i,k=int(self.code(b)),int(self.code(bn))
        self.trans[a,i,k]+=1.;self.trans_total[a,i]+=1.
        return k, revised

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        self.learn_control(b,a,bn)
        f=self.features(bn,b,hist);raw,gain,_=self.posterior(f)
        self.mean[0]+=gain*(y-raw); self.cov[0]*=1.-gain*f
        self.n[0]+=1.; self.sse[0]+=(y-float(np.clip(raw,0.,1.)))**2


class PhaseReward(CausalReward):
    """A bounded control Markov factor and conditionally selected reward delay."""
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag,cap=4)
        self.C=2**self.k
        self.F=1+J*self.L
        shape=(self.C,self.F,2)
        self.nn=np.zeros(shape); self.ss=np.zeros(shape); self.s2=np.zeros(shape)
        self.logev=np.zeros((self.C,self.F)); self.logev[:,0]=np.log(self.F-1)
        self.control_mass=np.zeros(self.C)
        self.fi=np.arange(self.F); self.ci=np.arange(self.C)

    def input(self,b,hist):
        return np.concatenate(([True],b>.5,*[hist[-l]>.5 if len(hist)>=l else self._zero>0
                                 for l in range(1,self.L)])).astype(int)

    def sample(self,rng):
        e=np.exp(self.logev-self.logev.max(1,keepdims=True)); e/=e.sum(1,keepdims=True)
        u=rng.random(self.C)
        chosen=np.minimum(self.F-1,(np.cumsum(e,axis=1)<u[:,None]).sum(1))
        n=self.nn[self.ci,chosen]; s=self.ss[self.ci,chosen]; s2=self.s2[self.ci,chosen]
        prior=self.global_mean()[:,None]
        mean=(s+prior)/(1.+n)
        vv=np.maximum(0.,(s2+prior**2+prior*(1.-prior)/3.)/(1.+n)-mean**2)
        draw=np.clip(mean+rng.standard_normal(mean.shape)*np.sqrt(vv/(2.+n)),0.,1.)
        return chosen, draw, vv, n

    def global_mean(self):
        return (self.ss[:,0,1]+.5)/(1.+self.nn[:,0,1])

    def deterministic(self):
        # Blend structure posterior moments rather than inventing a hard top model.
        e=np.exp(self.logev-self.logev.max(1,keepdims=True)); e/=e.sum(1,keepdims=True)
        means=(self.ss+self.global_mean()[:,None,None])/(1.+self.nn)
        vv=np.maximum(0.,(self.s2+1./3)/(1.+self.nn)-means**2)
        return e,means,vv,self.nn

    def moment(self,param,x):
        f,mean,var,n=param
        if f.ndim==1:
            j=x[f]
            return mean[self.ci,j],var[self.ci,j],n[self.ci,j]
        j=x[None,:];names=self.fi[None,:];c=self.ci[:,None]
        return (np.sum(f*v[c,names,j],axis=1) for v in (mean,var,n))

    def transitions(self,b):
        i=int(self.code(b))
        valid=(self.ci & int(self.binary[self.slots<0].sum()))==0
        prior=(self.control_mass+.5)*valid
        prior/=prior.sum()
        p=self.trans[:,i,:]+prior
        return p/p.sum(1,keepdims=True)

    def refine_draws(self,b,a,y,uniforms):
        active=self.slots>=0
        if active.any():
            cdf=np.cumsum(self.transitions(b)[a]); ids=np.minimum(self.C-1,np.searchsorted(cdf,uniforms[:,-1],side='right'))
            y[:,self.slots[active]]=(ids[:,None]&self.binary[active])!=0
        return y

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        c,revised=self.learn_control(b,a,bn)
        if revised:
            self.nn.fill(0.);self.ss.fill(0.);self.s2.fill(0.);self.control_mass.fill(0.)
            self.logev.fill(0.);self.logev[:,0]=np.log(self.F-1)
        x=self.input(b,hist)
        m,n=self.global_mean()[c],self.nn[c,0,1]
        local=self.ss[c,self.fi,x]; obs=self.nn[c,self.fi,x]
        pred=(local+m)/(obs+1.)
        var=max(1e-8,(self.s2[c,0,1]+1./3)/(1.+n)-m*m)
        self.logev[c]-=.5*np.square(y-pred)/var
        self.logev[c]-=self.logev[c].max()
        self.control_mass[c]+=1.
        self.nn[c,self.fi,x]+=1.;self.ss[c,self.fi,x]+=y;self.s2[c,self.fi,x]+=y*y


class ExceptionReward(PhaseReward):
    """A conditional real-valued reward exception on a temporal feature.

    Every feature is scored against a predictor estimated without its present
    observations. Mere absence observations do not test the exception law.
    """
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.surprise_mass=np.zeros(self.F)

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        c,revised=self.learn_control(b,a,bn)
        if revised:
            self.nn.fill(0.);self.ss.fill(0.);self.s2.fill(0.);self.control_mass.fill(0.)
            self.logev.fill(0.);self.logev[:,0]=np.log(self.F-1)
        x=self.input(b,hist)
        # Count all opportunities, not just ones which paid off.
        self.surprise_mass+=x
        m=(self.ss[c,:,0]+.5)/(1.+self.nn[c,:,0])
        n=self.nn[c,:,1]
        p=(self.ss[c,:,1]+.5)/(1.+n)
        # Variance calibrated from both arms before updating. A shared
        # continuous Brier/energy scale compares only matched predictions.
        var=.25
        self.logev[c,1:]+=(x*((y-m)**2-(y-p)**2)/(2.*var))[1:]
        self.control_mass[c]+=1.
        self.nn[c,self.fi,x]+=1.;self.ss[c,self.fi,x]+=y;self.s2[c,self.fi,x]+=y*y

    def sample(self,rng):
        theta = super().sample(rng)
        chosen=theta[0];n=self.nn[self.ci,chosen];s=self.ss[self.ci,chosen];s2=self.s2[self.ci,chosen]
        mean=(s+.5)/(1.+n); vv=np.maximum(0.,(s2+1./3)/(1.+n)-mean**2)
        # Moment-match the posterior mean with a bounded beta, preserving the
        # separation between conditional observation and mean uncertainty.
        strength=np.maximum(1e-8,mean*(1.-mean)*(2.+n)/np.maximum(vv,1e-12)-1.)
        draw=rng.beta(mean*strength,(1.-mean)*strength)
        return chosen,draw,vv,n

    def moment(self,param,x):
        f=param[0]
        if f.ndim==1:
            vals,var,n=super().moment(param,x)
            # The baseline is unaffected by which exception was selected, except
            # for subtracting data collected under that exception.
            return vals,var,n
        return super().moment(param,x)


class OptimisticReward(ExceptionReward):
    """Sparse feature intervention with a bounded favorable posterior scenario."""
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.observations=0

    def learn_joint(self,b,hist,a,y,bn):
        super().learn_joint(b,hist,a,y,bn)
        self.observations+=1

    def optimistic_features(self,explore=True):
        counts=self.nn[:,:,1]; raw=self.ss[:,:,1]; absent=self.nn[:,:,0]
        baseline=(self.ss[:,0,1]+.5)/(1.+counts[:,0])
        local=(raw+.5)/(1.+counts)
        # Smooth intrinsic prevalence estimate. Permanent marks are unlikely
        # to introduce novel opportunities; controlled marks are not causes.
        frequency=(self.surprise_mass+.5)/(1.+self.observations)
        salience=1.-frequency
        eligible=np.ones(self.F,bool)
        eligible[1:]=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        salience*=eligible;salience[0]=0.
        # Penalize positive-only accidental correlations at small support;
        # a genuinely new model stays available to challenge the baseline.
        w=1.-1./np.sqrt(1.+counts)
        fitted=baseline[:,None]+w*(local-baseline[:,None])
        optimistic=np.maximum(fitted,local+salience*(1.-local)/(1.+counts)) if explore else fitted
        optimistic[:,0]=baseline
        var=np.maximum(0.,(self.s2[:,:,1]+1./3)/(1.+counts)-local**2)
        return optimistic,var,counts

    def optimistic_moment(self,param,x):
        u,var,n=param
        which=np.argmax(np.where(x,u,-np.inf),axis=1)
        return u[self.ci,which],var[self.ci,which],n[self.ci,which],which


class EventReward(PhaseReward):
    """Reward type × controlled successor, with sparse likelihood parents."""
    B = 16  # reward partition; observed values retain their within-bin means
    P = 4   # bounded temporal-likelihood degree

    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.type_count=np.zeros((self.C,self.B));self.type_sum=np.zeros_like(self.type_count)
        self.on=np.zeros((self.C,self.B,self.F-1));self.evidence_type=np.zeros((self.C,self.F-1))
        self.center=(np.arange(self.B)+.5)/self.B
        self.total=0
        self.present=np.zeros(self.F-1)

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        c,revised=self.learn_control(b,a,bn)
        if revised:
            for z in self.type_count,self.type_sum,self.on,self.evidence_type,self.control_mass:
                z.fill(0.)
        x=self.input(b,hist)[1:]
        self.present+=x; self.total+=1
        r=min(self.B-1,int(y*self.B)); n=self.type_count[c]
        pooled=(self.on[c].sum(0)+.5)/(1.+n.sum())
        alt=(self.on[c,r]+.5)/(1.+n[r])
        self.evidence_type[c]+=np.log(np.where(x,alt,1.-alt)/np.where(x,pooled,1.-pooled))
        self.type_count[c,r]+=1.;self.type_sum[c,r]+=y;self.on[c,r]+=x;self.control_mass[c]+=1.

    def rule(self):
        """Precompute a bounded set of ancestral cue likelihoods at each state."""
        n=self.type_count; s=self.type_sum
        eligible=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        strength=np.where(eligible,self.evidence_type,-np.inf)
        if strength.shape[1] >= self.P:
            idx=np.argpartition(-strength,self.P-1,axis=1)[:,:self.P]
        else:
            idx=np.broadcast_to(np.arange(strength.shape[1]),strength.shape)
        weight=np.take_along_axis(strength,idx,axis=1)>np.log1p(self.F-1)
        on=np.take_along_axis(self.on,idx[:,None,:],axis=2)
        p=(on+.5)/(1.+n[:,:,None]); mean=(s+self.center[None,:]/self.B)/(n+1./self.B)
        total=n.sum(1)
        p0=(n+1./self.B)/(1.+total[:,None])
        base=np.sum(p0*mean,axis=1)
        recent=self.on.sum(1)
        rarity=1.-(self.present+.5)/(1.+self.total)
        novelty=rarity[None,:]*eligible/(1.+recent)
        # Usable explained parents close the large novelty hypothesis; any
        # residual uncertainty still enters the branch from its own reward law.
        novelty*=~weight.any(1)[:,None]
        return idx,weight,p,mean,p0,base,novelty

    def event_moment(self,rule,x,optimistic):
        idx,weight,p,atoms,p0,base,novelty=rule
        x=x[1:]
        pp=np.prod(np.where(weight[:,None,:],np.where(x[idx][:,None,:],p,1.-p),1.),axis=2)
        q=p0*pp;q/=q.sum(1,keepdims=True)
        mu=np.sum(q*atoms,axis=1)
        if optimistic:
            s=np.max(novelty*x[None,:],axis=1)
            mu=mu+(1.-mu)*s
        return mu,q,atoms


class RetractableEventReward(EventReward):
    """Controlled-state projection with bounded replay on scope revision."""
    def __init__(self,J,A,lag):
        from collections import deque
        super().__init__(J,A,lag)
        self.replay=deque(maxlen=2*self.C*A)
        self.onset=np.zeros((self.C,self.F-1))
        self.onset_total=np.zeros(J)
        self.past_on=np.zeros(J)
        self._past=self._zero

    def _observe_mark(self,b,a,bn):
        j=np.arange(self.J);x=(b>.5).astype(int);z=(bn>.5).astype(int)
        pooled=self.base[j,x];specific=self.split[j,a,x]
        self.evidence+=np.log((specific[j,z]+.5)/(specific.sum(1)+1.))-np.log((pooled[j,z]+.5)/(pooled.sum(1)+1.))
        self.base[j,x,z]+=1.;self.split[j,a,x,z]+=1.
        p=(self.split[...,1]+.5)/(1.+self.split.sum(-1))
        # The inferred actuator is a strong, reproducible intervention, not
        # merely a weakly associated dependent emission. The soft prior odds
        # charge below grows with the number of candidate sensors.
        separated=np.max(p.max(1)-p.min(1),axis=1)
        good=(separated>.5)&(self.evidence>np.log1p(self.J))
        if getattr(self,'controlled_dwell',False):
            # Scope models sustained actuators; a successor conditioned on a
            # chosen approach is not thereby a controllable place. Require
            # both entry/exit evidence and witnessed action-sensitive dwell.
            stay=p[:,:,1]
            good&=(stay.max(1)-stay.min(1)>.5)
        if hasattr(self,'known_mode_roles'):
            good|=self.known_mode_roles
        old=self.slots.copy()
        self.slots[~good[np.maximum(0,self.slots)]]=-1
        score=np.where(good,self.evidence,-np.inf);score[self.slots[self.slots>=0]]=-np.inf
        for d in np.flatnonzero(self.slots<0):
            i=int(np.argmax(score))
            if not np.isfinite(score[i]): break
            self.slots[d]=i;score[i]=-np.inf
        return not np.array_equal(old,self.slots)

    def _record(self,b,a,y,bn,x,phase):
        i,c=int(self.code(b)),int(self.code(bn)); self.trans[a,i,c]+=1.;self.trans_total[a,i]+=1.
        r=min(self.B-1,int(y*self.B));n=self.type_count[c]
        pooled=(self.on[c].sum(0)+.5)/(1.+n.sum())
        alt=(self.on[c,r]+.5)/(1.+n[r])
        self.evidence_type[c]+=np.log(np.where(x,alt,1.-alt)/np.where(x,pooled,1.-pooled))
        self.type_count[c,r]+=1.; self.type_sum[c,r]+=y;self.on[c,r]+=x;self.control_mass[c]+=1.
        self.onset[c]+=phase

    def phase(self,b,hist):
        lags=np.stack([b,*[hist[-l] if len(hist)>=l else self._zero for l in range(1,self.L+1)]])>.5
        return (lags[:-1]&~lags[1:]).ravel()

    def learn_joint(self,b,hist,a,y,bn):
        if not np.isfinite(y) or not 0.<=y<=1.:
            raise ValueError('bounded real reward must be in [0,1]')
        x=self.input(b,hist)[1:];phase=self.phase(b,hist);self.total+=1
        self.present+=x;self.onset_total+=phase[:self.J];self.past_on+=b
        self.replay.append((b.copy(),a,y,bn.copy(),x,phase))
        if self._observe_mark(b,a,bn):
            for z in self.type_count,self.type_sum,self.on,self.evidence_type,self.control_mass,self.trans,self.trans_total,self.onset:
                z.fill(0.)
            self._replaying=True
            try:
                for r in self.replay:self._record(*r)
            finally:
                self._replaying=False
        else:
            self._record(*self.replay[-1])

    def rule(self):
        idx,weight,p,mean,p0,base,_=super().rule()
        eligible=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        rarity=1.-(self.present+.5)/(1.+self.total)
        transition=self.onset_total/(.5+self.past_on)
        novelty=rarity*np.tile(transition,self.L)*eligible/(1.+self.onset)
        novelty*=~weight.any(1)[:,None]
        return idx,weight,p,mean,p0,base,novelty

    def event_moment(self,rule,x,optimistic,phase=None):
        vals,q,atoms=super().event_moment(rule,x,False)
        if optimistic:
            gain=np.max(rule[-1]*phase,axis=1)
            vals=vals+(1.-vals)*gain
        return vals,q,atoms


class ProvisionalEventReward(RetractableEventReward):
    """Optimistic local onset hypotheses until the event law is supported."""
    def __init__(self,J,A,lag):
        super().__init__(J,A,lag)
        self.onset_reward=np.zeros_like(self.onset)

    def _record(self,b,a,y,bn,x,phase):
        super()._record(b,a,y,bn,x,phase)
        c=int(self.code(bn)); self.onset_reward[c]+=y*phase

    def _observe_mark(self,b,a,bn):
        revised=super()._observe_mark(b,a,bn)
        if revised:
            self.onset_reward.fill(0.)
        return revised

    def provisional(self,rule):
        known=rule[1].any(1)
        allowed=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        n=self.onset; s=self.onset_reward
        m=(s+.5)/(1.+n)
        # Fully untested onsets retain a large possible novel effect, supported
        # cases retain it in proportion to remaining posterior mean deviation.
        strength=np.tile(self.onset_total/(.5+self.past_on),self.L)
        prevalence=(.5+self.present)/(1.+self.total)
        u=m+(1.-m)*strength*(1.-prevalence)/np.sqrt(1.+n)
        u*=allowed;u*=~known[:,None]
        return np.maximum(0.,u)

    def likelihood(self,rule,x):
        idx,weight,p,atoms,p0,base,_=rule
        law=np.prod(np.where(weight[:,None,:],np.where(x[...,idx][:,:,None,:],p,1.-p),1.),axis=-1)
        mass=(1.+self.control_mass[:,None])*p0
        pr=mass*law; ev=pr.sum(-1);prob=pr/ev[...,None]
        mean=(prob*atoms).sum(-1)
        return mean,prob,law,ev

    def all_transitions(self):
        valid=(self.ci & int(self.binary[self.slots<0].sum()))==0
        prior=(self.control_mass+.5)*valid
        prior/=prior.sum()
        p=self.trans+prior
        return p/p.sum(-1,keepdims=True)


class ElapsedEventReward(ProvisionalEventReward):
    """Do not trade reward for novelty that has not yet been observed."""
    elapsed_only=True

    def all_transitions(self):
        valid=(self.ci & int(self.binary[self.slots<0].sum()))==0
        seen=(self.control_mass>0)&valid
        rest=valid&~seen
        prior=np.zeros(self.C)
        if seen.any():
            prior[seen]=1./seen.sum()
            if rest.any():
                # A learned state dictionary may expand; residual mass is
                # limited by how often an actually new state could still arise.
                mass=1./(1.+self.control_mass.sum())
                prior[seen]*=1.-mass;prior[rest]=mass/rest.sum()
        else:
            prior[valid]=1./valid.sum()
        p=self.trans+prior
        return p/p.sum(-1,keepdims=True)

    def provisional(self,rule):
        n=self.onset; m=(self.onset_reward+.5)/(1.+n)
        eligible=~np.tile(np.isin(np.arange(self.J),self.slots),self.L)
        strength=np.tile(self.onset_total/(.5+self.past_on),self.L)
        rarity=1.-(.5+self.present)/(1.+self.total)
        plausibility=strength*rarity*eligible
        base=rule[5][:,None]
        # When a cause is poorly tested, pull towards the contextual hypothesis
        # in proportion to the transient-role probability, not mere activation.
        local=base+plausibility*(m-base)
        upper=local+plausibility*(1.-m)/np.sqrt(1.+n)
        upper*=~rule[1].any(1)[:,None]
        return upper
