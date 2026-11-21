"""Anonymous-learner cancellation exposure diagnostic; not a qualification gate.

Only this experimenter sees the fixture's cue, trap and position. The learner
receives ordinary observation/action/reward transitions, never a flag or label.
All arms use the same environment and agent seeds and a fixed prefix duration.
At matched target cue times the payoff and null arms differ by one motor action;
the adverse arm makes the same excursion on negative cues (timing differs).
"""
from __future__ import annotations
import argparse
import importlib
import json
import numpy as np
from envs.tasks import VARIANTS
from models.factor_rollout import provisional_values


def run(name, seed, arm, steps=240, prefix=100, exposures=4, agent='v48'):
    v=VARIANTS[name]; env=v.make(100+seed); x=env.reset()
    b=importlib.import_module('agents.'+agent).Agent(v.n_inputs,v.n_actions,11,seed=seed)
    rows=[]; forced=[]; reward=[]; chosen_targets=0
    for t in range(steps):
        iscue=bool(x[3]>.5); positive=iscue and x[7]<.5
        target=iscue and ((not positive) if arm=='adverse' else positive)
        f=b.factor; h=f.reward
        info=None
        if iscue:
            # Duplicate the controller's upcoming random draw without changing it.
            rg=np.random.default_rng();rg.bit_generator.state=b.rng.bit_generator.state
            u=rg.random((b.draws,f.J+4));q=provisional_values(f,x,u,True)
            plain=provisional_values(f,x,u,False)
            rule=h.rule();bb=f._bits(x); c=int(h.code(bb))
            move=h.all_transitions()[:,c,:]
            info=dict(t=t,positive=positive,pos=int(np.argmax(x[:3])),
                      q=q.tolist(),plain=plain.tolist(),slots=h.slots.tolist(),
                      code=c,rule_indices=rule[0].tolist(),rule_weights=rule[1].tolist(),
                      left_mass=move[:,[int(i) for i in h.ci if any(h.slots[k]==0 and i & h.binary[k] for k in range(h.k))]].sum(1).tolist(),
                      switches=h.switches)
        a=b.step(x)
        scripted=False
        if arm!='free' and t<prefix:
            # Each arm returns to center before the next opportunity; target
            # opportunities and their omissions are yoked for payoff vs null.
            a=1 if x[1]>.5 else (2 if x[0]>.5 else 0)
            if target and chosen_targets<exposures:
                chosen_targets+=1
                a=1 if arm=='null' else 0
                forced.append([t,positive,a])
            f.predict(x,a); b.pending=(x,a);scripted=True
        y,r,_,_=env.step(a);b.learn(y);reward.append(float(r))
        if info is not None:
            info.update(action=a,reward=float(r),scripted=scripted)
            rows.append(info)
        x=y
    cues=[z for z in rows if z['t']>=prefix and not z['scripted']]
    good=[z for z in cues if z['positive']]
    reach=[z for z in good if z['pos']<=1]
    outcomes=[i for i,r in enumerate(reward) if r==1.]
    return dict(name=name,seed=seed,arm=arm,agent=agent,steps=steps,prefix=prefix,
                forced=forced,first_payoff=outcomes[0] if outcomes else None,
                payoffs_prefix=sum(r==1 for r in reward[:prefix]),
                payoffs_after=sum(r==1 for r in reward[prefix:]),
                good_opportunities=len(good),reachable_good=len(reach),
                reached_payoffs=sum(z['reward']==1 for z in good),
                mean_after=float(np.mean(reward[prefix:])),cues=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('name',choices=['canc','canc-noise3']);p.add_argument('seed',type=int,choices=[0,6]);p.add_argument('arm',choices=['free','payoff','null','adverse']);p.add_argument('--steps',type=int,default=240);p.add_argument('--prefix',type=int,default=100);p.add_argument('--exposures',type=int,default=4);p.add_argument('--agent',default='v48');a=p.parse_args()
    print(json.dumps(run(a.name,a.seed,a.arm,a.steps,a.prefix,a.exposures,a.agent),separators=(',',':')),flush=True)

if __name__=='__main__': main()
