"""Read-only approximate posterior rollout for the factor's successor value head."""
from __future__ import annotations
import numpy as np

from models.factor_projection import factor_current, finite_marginal, scratch_lik, scratch_params, window_marg
from models.factor_reward import conditional_draws


def process_after(d, x, previous, root=None):
    """Conditional HMM/window marginals, with the previous action in likelihood."""
    from models.decompose import Window
    arr=np.asarray(x,float);initial=None if root is None else np.asarray(root,float)
    out=np.full((d.A,d.n),.5)
    for g in d.groups:
        if not g.started and initial is None:
            continue
        s=arr[g.chans]; lw,pp=[],[]
        for weight,h in zip(g.logw,g.hs):
            if isinstance(h,Window):
                current=h.hist if g.started else [tuple(initial[g.chans]>.5)]
                q=np.clip(window_marg(h,previous,current),1e-6,1.-1e-6)
                lw.append(weight+np.log(np.where(s>.5,q,1.-q)).sum())
                hist=(*current,tuple(s>.5))[-h.WIN:]
                pp.append([window_marg(h,a,hist) for a in range(d.A)])
            else:
                if hasattr(h,'_dirty'):
                    t,e=scratch_params(h);l=scratch_lik(s,e)
                else:
                    t,e=np.asarray([h.T(a) for a in range(d.A)]),h.E();l=h._lik(s,e)
                b=h.b
                if not g.started:
                    v=scratch_lik(initial[g.chans],e);b=v/v.sum()
                post=(b@t[previous])*l; p=max(1e-300,float(post.sum()));post/=p
                lw.append(weight+np.log(p));pp.append([(post@t[a])@e for a in range(d.A)])
        weights=np.exp(lw-np.max(lw));weights/=weights.sum()
        out[:,g.chans]=np.einsum('i,iaj->aj',weights,np.asarray(pp))
    return out


def branch_weights(f, b, bn, local, w):
    """Filter a hypothetical sensory outcome; expected, not sampled, regime anchor."""
    k=w.shape[1]; place=f._signature(bn)
    p=f._regime_prior(place,k)
    fast=~f.slow
    quiet=not b[fast].any()
    if place!=f._signature(b) or (quiet and not bn[fast].any()):
        # Real filter treats boundaries as exogenous and silence as clock only.
        wi=np.exp(f.logw-f.logw.max())
    else:
        like=np.where(bn[fast]>.5,local[...,fast],1.-local[...,fast])
        with np.errstate(divide='ignore'):
            ll=np.log(w)+np.log(like).sum(-1)
            clock=f._activity(local,w,b,fast)
            if clock is not None:
                ll+=np.log(clock[1])
        joint=np.exp(ll-ll.max())
        wi=joint.sum(1)
        post=np.divide(joint,wi[:,None],out=p.copy(),where=wi[:,None]>0.)
        # Drop scratch parameter updates to particle counts, keep the posterior
        # anchor and hazard. This is a frozen-parameter continuation.
        p=(1.-f.hz[:,None])*post+f.hz[:,None]*p
    return wi[:,None]*f._cue_prior(p,bn,k)


def next_prob(f, b, action, hist, w, pp):
    """Same conditional emissions as the frozen predictor for a scratch input."""
    fast=~f.slow; quiet=not b[fast].any(); place=f._signature(b)
    mem=None if quiet and f.slow.any() else f._members(b,action,hist,w.shape[1])
    local,_=f._conditional(mem,w.shape[1],place,quiet)
    q = local if pp is None else f.expert*local+(1.-f.expert)*pp
    js=np.flatnonzero(f.slow)
    counts=f.slow_counts[js,b[js].astype(int)]
    slow_p=(counts[:,1]+.5)/(counts.sum(1)+1.)
    return q,slow_p,quiet


def outcome_values(f, x, uniforms, theta=None):
    """Two-step posterior value through a shared observed-successor reward member."""
    b=f._bits(x); hist=(*f.hist,b); head=f.reward
    hybrid=f.reward_hybrid; controlled=f.reward_control
    sensory=uniforms[:,:f.J+3]
    if hybrid:
        g,u=(head.mean,head.context.mean) if theta is None else theta
        param=g[0]; feat=head.context.features(b,f.hist)
    else:
        param=head.mean[0] if theta is None else theta[0]
    scores=np.zeros(f.A)
    for a,(w,mix,slow_p,local) in enumerate(factor_current(f,x)):
        y = conditional_draws(w,mix,f.slow,slow_p,not b[~f.slow].any(),sensory)
        if controlled:
            y=head.refine_draws(b,a,y,uniforms)
        # Duplicate sensory results have identical scratch posterior and value.
        ys,times=np.unique(y,axis=0,return_counts=True)
        for bn,n in zip(ys,times):
            observation=np.full(f.n,.5); observation[f.ch]=bn
            proc=process_after(f.proc,observation,a,x)[:,f.ch] if f.proc is not None else [None]*f.A
            wb=branch_weights(f,b,bn,local,w)
            v = head.features(bn,b,f.hist) if controlled else head.features(bn,())
            raw = param@v
            if hybrid:
                nxt = head.context.features(bn,hist)
                raw += u[a]@feat
                _,gain,context_gain,var = head.joint_posterior(v,feat,a)
            else:
                _,gain,var=head.posterior(v,0)
            values, mass=head.distribution(raw,head.noise[0] if theta is not None else var)
            best=np.empty((len(mass),f.A))
            for a1 in range(f.A):
                q,slow_p,quiet=next_prob(f,bn,a1,hist,wb,proc[a1])
                yn=conditional_draws(wb,q,f.slow,slow_p,quiet,sensory)
                if controlled:
                    yn=head.refine_draws(bn,a1,yn,uniforms)
                phi = head.features(yn,bn,hist) if controlled else head.features(yn,())
                immediate=phi@param
                sensitivity=phi@gain
                if hybrid:
                    immediate += u[a1]@nxt
                    if a1==a:
                        sensitivity += context_gain@nxt
                shifted=immediate[:,None]+sensitivity[:,None]*(values[None,:]-raw)
                best[:,a1]=np.clip(shifted,0.,1.).mean(0)
            scores[a] += n * float(mass@(values+best.max(1))) / len(y)
    return scores


def phase_values(f,x,uniforms,theta=None):
    """Exact continuation integration for the learned finite control factor."""
    b=f._bits(x);head=f.reward; hist=(*f.hist,b)
    theta=head.deterministic() if theta is None else theta
    feat=head.input(b,f.hist); mood,variance,count=head.moment(theta,feat)
    scores=np.zeros(f.A); sensory=uniforms[:,:f.J+3]
    for a,(w,mix,slow_p,_) in enumerate(factor_current(f,x)):
        y=conditional_draws(w,mix,f.slow,slow_p,not b[~f.slow].any(),sensory)
        y=head.refine_draws(b,a,y,uniforms)
        for bn in y:
            c=int(head.code(bn)); opt=head.transitions(bn)
            future=head.input(bn,hist); next_m,_,_=head.moment(theta,future)
            vals,prob=head.distribution(mood[c],variance[c])
            selected=theta[0]
            match=(feat[selected[c]]==future[selected[c]]) if selected.ndim==1 else float(np.sum(selected[c]*(feat==future)))
            q=opt@next_m
            correction=opt[:,c]*match/(2.+count[c])
            continuation=np.clip(q[:,None]+correction[:,None]*(vals[None,:]-mood[c]),0.,1.).max(0)
            scores[a] += float(prob@(vals+continuation))/len(y)
    return scores


def optimistic_values(f,x,uniforms,explore=True):
    """Optimistic *model* for action choice, adaptive forecast inside the tree."""
    b=f._bits(x);head=f.reward; hist=(*f.hist,b)
    theta=head.optimistic_features(explore)
    feat=head.input(b,f.hist); mood,variance,count,which=head.optimistic_moment(theta,feat)
    scores=np.zeros(f.A); sensory=uniforms[:,:f.J+3]
    for a,(w,mix,slow_p,_) in enumerate(factor_current(f,x)):
        y=conditional_draws(w,mix,f.slow,slow_p,not b[~f.slow].any(),sensory)
        y=head.refine_draws(b,a,y,uniforms)
        for bn in y:
            c=int(head.code(bn)); opt=head.transitions(bn)
            future=head.input(bn,hist); next_m,_,_,next_which=head.optimistic_moment(theta,future)
            vals,prob=head.distribution(mood[c],variance[c])
            q=opt@next_m
            correction=opt[:,c]*(which[c]==next_which[c])/(2.+count[c])
            continuation=np.clip(q[:,None]+correction[:,None]*(vals[None,:]-mood[c]),0.,1.).max(0)
            scores[a] += float(prob@(vals+continuation))/len(y)
    return scores


def event_values(f,x,uniforms,optimistic=True):
    """Two-step law with an atom-conditioned, finite continuation correction."""
    if hasattr(f.reward,'provisional'):
        return provisional_values(f,x,uniforms,optimistic)
    b=f._bits(x);head=f.reward; hist=(*f.hist,b)
    rule=head.rule(); feat=head.input(b,f.hist)
    phase_model=hasattr(head,'phase')
    def moments(b,h,feat):
        return head.event_moment(rule,feat,optimistic,head.phase(b,h)) if phase_model else head.event_moment(rule,feat,optimistic)
    mu,rprob,atoms=moments(b,f.hist,feat)
    scores=np.zeros(f.A); sensory=uniforms[:,:f.J+3]
    for a,(w,mix,slow_p,_) in enumerate(factor_current(f,x)):
        y=conditional_draws(w,mix,f.slow,slow_p,not b[~f.slow].any(),sensory)
        y=head.refine_draws(b,a,y,uniforms)
        for bn in y:
            c=int(head.code(bn)); opt=head.transitions(bn)
            future=head.input(bn,hist); next_m,_,_=moments(bn,hist,future)
            nv=opt@next_m
            # First-order moment projection of adding one bounded reward to
            # the future controlled state, not an additional VOI bonus.
            v0=float(rprob[c]@atoms[c]); nf=head.control_mass[c]
            q=nv[:,None]+(opt[:,c]/(2.+nf))[:,None]*(atoms[c,None,:]-v0)
            scores[a] += (mu[c]+rprob[c]@np.clip(q,0.,1.).max(0))/len(y)
    return scores


def provisional_values(f,x,uniforms,optimistic=True):
    """Enumerate the control state; sample only residual sensory observations.

    Atom probabilities in the inner tree add the observed first atom to the
    corresponding real-reward sufficient statistic. The likelihood parents of
    that atom, and the transition factor, remain frozen during this lookahead.
    """
    head=f.reward; b=f._bits(x); input0=head.input(b,f.hist)[1:]
    motion_optimism=optimistic and getattr(head,'motion_optimism',False)
    one_window=optimistic and getattr(head,'one_window',False)
    rule=head.rule(); atoms=rule[3]; trans=head.all_transitions();code=int(head.code(b))
    r,p,_,_=head.likelihood(rule,input0[None,:])
    splits=head.signed_rules(rule) if hasattr(head,'signed_rules') else []
    seen_phase=head.phase(b,f.hist)
    group0=None
    if splits:
        r,p,outcome,_,group0=head.context_moment(splits,input0[None,:],seen_phase[None,:],r,p,atoms)
        atoms0=outcome[0];group0=group0[0]
    else:
        atoms0=atoms
    r,p=r[0],p[0]
    incumbent=r.copy()
    pair_probe=optimistic and hasattr(head,'pair_context')
    bridge,base=head.bridge_table(rule) if optimistic and hasattr(head,'bridge_table') else (None,None)
    if bridge is not None and not bridge.any():
        bridge=None
    if optimistic:
        bound,_=head.provisional(rule)
        incumbent=np.maximum(r,(bound*seen_phase).max(1))
        if pair_probe:
            offer=head.pair_context(rule,input0,seen_phase)
            incumbent=np.maximum(incumbent,offer if group0 is None else np.where(group0==0,offer,0.))
    active=head.slots>=0; cols=head.slots[active]
    states=np.flatnonzero(np.any(trans[:,code,:]>0.,axis=0)) if getattr(head,'sparse_outcomes',False) else head.ci
    marks=(states[:,None]&head.binary[active])!=0
    scores=np.zeros(f.A); sensory=uniforms[:,:f.J+3]
    hist=(*f.hist,b)
    tail=head.input(b,hist)[1+f.J:]
    tailphase=head.phase(b,hist)[f.J:]
    if pair_probe:
        prospective_pair=head.pair_context(rule,np.r_[np.zeros(f.J),tail],np.r_[np.zeros(f.J,bool),tailphase],shift=True)
    quiet=not b[~f.slow].any()
    relevant=None
    if getattr(head,'factored_projection',False):
        known=rule[0][rule[1]]
        relevant=np.unique(np.r_[known[known<f.J],np.array([m for _,m,l,_,_,_ in splits if l==0],dtype=int)])
    if relevant is not None and len(relevant)==0 and bridge is None:
        laws=[None]*f.A
    else:
        laws=factor_current(f,x)
    for a,law in enumerate(laws):
        immediate=incumbent
        if law is None:
            y=np.zeros((1,f.J)); branch_mass=np.ones(1)
        else:
            w,mix,slow_p,_=law
            if bridge is not None:
                activity=f._activity(mix,w,b,~f.slow)
                ww=w if activity is None else w*activity[1]
                forecast=np.einsum('nk,nkj->j',ww,mix)/w.sum();forecast[f.slow]=slow_p
                evidence=np.clip((forecast-base)/(1.-base),0.,1.)
                immediate=np.maximum(incumbent,r+np.max(bridge*evidence,axis=1))
            if relevant is not None and len(relevant)<=3:
                parts,branch_mass=finite_marginal(w,mix,f.slow,slow_p,quiet,relevant)
                y=np.zeros((len(parts),f.J));y[:,relevant]=parts
            else:
                y=conditional_draws(w,mix,f.slow,slow_p,quiet,sensory)
                branch_mass=np.full(len(y),1./len(y))
        # Branch index is (residual sensory draw, controlled outcome).
        by=np.repeat(y,len(states),axis=0)
        by[:,cols]=np.tile(marks,(len(y),1))
        feature=np.column_stack((by,np.broadcast_to(tail,(len(by),len(tail))))).astype(bool)
        m2,p2,l2,z2=head.likelihood(rule,feature)
        onset=(by>.5)&~(b>.5)
        phase=np.column_stack((onset,np.broadcast_to(tailphase,(len(by),len(tailphase)))))
        if splits:
            m2,p2,atoms2,count2,group2=head.context_moment(splits,feature,phase,m2,p2,atoms)
        else:
            atoms2=atoms
        epistemic=optimistic and getattr(head,'epistemic_bonus',False)
        cont=m2.copy()
        if optimistic:
            if getattr(head,'elapsed_only',False):
                phase[:,:f.J]=False  # no prospective novelty credit for unseen triggers
            cont=np.maximum(cont,np.max(phase[:,None,:]*bound,axis=-1))
            if pair_probe:
                cont=np.maximum(cont,prospective_pair if not splits else np.where(group2==0,prospective_pair,0.))
        c=np.tile(states,len(y));idx=np.arange(len(c)); motion=trans[:,c,:].transpose(1,0,2)
        transition_bonus=getattr(head,'root_transition_bonus',0.)
        if epistemic:
            import os as _os
            # One posterior sd of the successor outcome law, weighted by branch
            # mass and the control transition. Zero once a law tightens; keeps a
            # nonzero pull toward under-tested atoms (D-114). Not onset novelty.
            a2d=np.asarray(atoms2)
            if a2d.ndim==2:
                a2d=a2d[None,:,:]
            second=(p2*a2d*a2d).sum(-1)
            sd2=np.sqrt(np.maximum(0.,second-m2*m2))
            bm_tc=np.repeat(branch_mass,len(states))
            trans_a=trans[a,code]
            sd_a=float(np.dot(bm_tc*trans_a[c],sd2[idx,c]))
            if _os.environ.get('AR6_DEBUG'):
                print('      a=%d sd_a=%.4f base=%s' % (a, sd_a, np.round(scores, 4)), flush=True)
        else:
            sd_a=0.
        # Only reward type of the arrived state is updated by the real atom.
        # Atoms retain continuous representative utility (no Bernoulli cast).
        difference=atoms0[c]-m2[idx,c,None]
        delta=l2[idx,c,:]/(z2[idx,c,None]+l2[idx,c,:])*difference
        if splits:
            same=group2[idx,c]==group0[c]
            conditioned=group2[idx,c]>0
            delta=np.where(conditioned[:,None],difference/(1.+count2[idx,c,None]),delta)
            delta*=same[:,None]
        q=np.einsum('baj,bj->ba',motion,cont)
        if motion_optimism:
            legal=head.control_mass>0.
            peak=np.max(np.where(legal,cont,-np.inf),axis=1) if legal.any() else cont.max(1)
            q+=(peak[:,None]-q)/(2.+head.trans_total[:,c].T)
        correction=motion[idx,:,c,None]*delta[:,None,:]
        gain=branch_mass@(np.clip(q[:,:,None]+correction,0.,1.).max(1)*p[c]).sum(1).reshape(len(y),len(states))
        discount=getattr(head,'discount',1.)
        root_bonus = transition_bonus/(2.+head.trans_total[a,code]) if transition_bonus else 0.
        if one_window:
            q0=np.einsum('baj,bj->ba',motion,m2)
            if motion_optimism:
                peak0=np.max(np.where(legal,m2,-np.inf),axis=1) if legal.any() else m2.max(1)
                q0+=(peak0[:,None]-q0)/(2.+head.trans_total[:,c].T)
            gain0=branch_mass@(np.clip(q0[:,:,None]+correction,0.,1.).max(1)*p[c]).sum(1).reshape(len(y),len(states))
            vals=np.stack((immediate[states]+discount*gain0,r[states]+discount*gain))
            means=vals@trans[a,code,states]
            if epistemic:
                means+=sd_a
            if motion_optimism:
                peaks=vals[:,legal[states]].max(1) if legal[states].any() else vals.max(1)
                means+=(peaks-means)/(2.+head.trans_total[a,code])
            scores[a]=means.max()+root_bonus
        else:
            values=immediate[states]+discount*gain
            scores[a]=trans[a,code,states]@values+root_bonus
            if epistemic:
                scores[a]+=sd_a
            if motion_optimism:
                peak=values[legal[states]].max() if legal[states].any() else values.max()
                scores[a]+=(peak-scores[a])/(2.+head.trans_total[a,code])
    return scores
