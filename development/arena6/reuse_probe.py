"""Exposed diagnostic: separate free discovery from post-exposure reuse.

Arms on the same (variant, seed):
  free   -- act freely throughout.
  expose -- for the first `k` cancellation-cue opportunities (cue=1, lamp=0),
            a scripted prefix plays the *max-reward* action (end on the food
            side); afterwards the agent acts freely.
  yoke   -- same number and timing of scripted interventions, but plays an
            action that cannot pay off (end on the far side). Matched in count
            and timing by construction.

Only ordinary observations and actions reach the learner in every arm; the
script decides an action from the agent's own observation, and no reward,
label or teacher flag is injected.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from envs.tasks import VARIANTS, SPLITS
from agents import agent_class, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', default='v48')
    p.add_argument('--variant', default='canc')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--steps', type=int, default=420)
    p.add_argument('--arm', default='free', choices=('free', 'expose', 'yoke'))
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    v = VARIANTS[a.variant]
    _, _, eb, bb = SPLITS['dev']
    env = v.make(eb + a.seed)
    x = env.reset()
    brain = agent_class(load(a.agent))(v.n_inputs, v.n_actions, 11, seed=bb + a.seed)
    f = brain.factor
    used = 0
    rows = []
    for t in range(a.steps):
        cue = x[3] > .5
        lamp = x[7] > .5
        pos = int(np.argmax(x[:3]))
        scripted = None
        if a.arm != 'free' and cue and not lamp and used < a.k:
            # The action that lands on the food side (position 0) this step.
            target = 0 if a.arm == 'expose' else 2
            scripted = int(np.clip(target - pos + 1, 0, 2))
            used += 1
        if scripted is None:
            act = int(brain.step(x))
        else:
            # Keep the agent's own belief updating as if it had chosen this action.
            brain.pending = (x, scripted)
            f.predict(x, scripted)
            act = scripted
        y, r, _, inf = env.step(act)
        brain.learn(y)
        rows.append(dict(t=t, pos=pos, cue=int(cue), lamp=int(lamp), act=act,
                         scripted=int(scripted is not None), rew=float(r),
                         shock=int(inf['received_shock']), post=int(np.argmax(y[:3]))))
        x = y
    cue_nt = [r for r in rows if r['cue'] and not r['lamp']]
    post = [r for r in cue_nt if r['t'] > max(40, 5 * a.k)]
    rec = dict(k=[a.agent, a.variant, a.seed, a.arm, a.k], m=dict(
        rows=rows, score=float(np.mean([r['rew'] for r in rows[-200:]])),
        exposures=int(used), first_payoff=next((r['t'] for r in rows if r['rew'] == 1.), None),
        n_payoff=sum(1 for r in rows if r['rew'] == 1.),
        late_cue_nt=len(post),
        late_left=sum(1 for r in post if r['act'] == 0),
        late_stay=sum(1 for r in post if r['act'] == 1),
        late_shock=sum(1 for r in rows if r['shock'] and r['t'] > 200)))
    if a.out:
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    else:
        m = rec['m']
        print(a.arm, a.seed, 'score %.3f' % m['score'], 'first_payoff', m['first_payoff'],
              'n_payoff', m['n_payoff'], 'late_left/%d' % m['late_cue_nt'], m['late_left'],
              'late_stay', m['late_stay'], 'late_shock', m['late_shock'])


if __name__ == '__main__':
    main()
