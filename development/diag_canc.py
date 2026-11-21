"""Diagnostic trace for v48 on canc: discovery vs reuse, seed6 vs seed0."""
import sys
sys.path.insert(0, 'development')
import numpy as np
import agents
from envs.tasks import VARIANTS

def run(variant_name, env_seed, brain_seed, steps=600, agent_name='v48', scripted=None):
    v = VARIANTS[variant_name]
    env = v.make(env_seed)
    mod = agents.load(agent_name)
    b = agents.agent_class(mod)(v.n_inputs, v.n_actions, 11, seed=brain_seed)
    obs = env.reset()
    # scripted: dict step->action to force (for exposure fixture)
    scripted = scripted or {}
    log = []
    first_payoff = None
    cue_count = 0
    payoff_opps = 0  # cue & no-trap steps
    for t in range(steps):
        # capture internals before step
        f = b.factor
        head = f.reward
        # NOTE: do not consume b.rng here; Q evaluation uses independent RNG below.
        cue = obs[3] > .5
        trap = obs[7] > .5
        pos = int(np.argmax(obs[:3]))
        if cue:
            cue_count += 1
            if not trap:
                payoff_opps += 1
        # record pre-step internals
        try:
            rule = head.rule()
            bound, _ = head.provisional(rule)
            # pair offer
            input0 = head.input(f._bits(obs), f.hist)[1:]
            seen_phase = head.phase(f._bits(obs), f.hist)
            offer = head.pair_context(rule, input0, seen_phase) if hasattr(head, 'pair_context') else None
            splits = head.signed_rules(rule) if hasattr(head, 'signed_rules') else []
            r, p, _, _ = head.likelihood(rule, input0[None, :])
            r = r[0]
            slots = head.slots.copy()
            anchors = head.anchors.copy() if hasattr(head, 'anchors') else None
            trans_total = head.trans_total.copy()
            code = int(head.code(f._bits(obs)))
            # Q values: use provisional_values directly with fresh uniforms
            from models.factor_rollout import provisional_values
            u2 = np.random.default_rng(12345 + t).random((b.draws, f.J+4))
            q = provisional_values(f, obs, u2, True)
        except Exception as e:
            rule = bound = offer = splits = r = slots = anchors = trans_total = code = q = None
            print('diag err', e)
        if t in scripted:
            a = scripted[t]
        else:
            a = b.step(obs)
        obs2, rew, _, info = env.step(a)
        b.learn(obs2)
        if rew == 1.0 and first_payoff is None:
            first_payoff = t
        if cue:
            log.append(dict(t=t, pos=pos, trap=int(trap), a=a, rew=float(rew),
                            q=[float(x) for x in q] if q is not None else None,
                            code=code, slots=slots.tolist() if slots is not None else None,
                            anchors=anchors.tolist() if anchors is not None else None,
                            r=[float(x) for x in r] if r is not None else None,
                            bound_max=[float(bound[c].max()) for c in range(bound.shape[0])] if bound is not None else None,
                            offer=[float(x) for x in offer] if offer is not None else None,
                            splits=[(int(s[0]), int(s[1]), int(s[2]), float(np.asarray(s[4]).ravel()[0])) for s in splits] if splits else [],
                            rule_idx=rule[0].tolist() if rule is not None else None,
                            rule_w=rule[1].tolist() if rule is not None else None,
                            trans_total=trans_total.tolist() if trans_total is not None else None,
                            ))
        obs = obs2
    return log, first_payoff, cue_count, payoff_opps

if __name__ == '__main__':
    import json
    for (env_seed, brain_seed, label) in [(106, 6, 'seed6'), (100, 0, 'seed0')]:
        print(f'===== {label} env={env_seed} brain={brain_seed} =====')
        log, fp, cc, po = run('canc', env_seed, brain_seed, steps=300)
        print(f'first_payoff={fp} cues={cc} payoff_opps={po}')
        for e in log[:25]:
            print(f"t={e['t']} pos={e['pos']} trap={e['trap']} a={e['a']} r={e['rew']} Q={[round(x,3) for x in e['q']]} code={e['code']} slots={e['slots']} anchors={e['anchors']}")
            print(f"   r={[round(x,3) for x in e['r']]} boundmax={[round(x,3) for x in e['bound_max']]} offer={[round(x,3) for x in e['offer']]} splits={e['splits']}")
            print(f"   rule_idx={e['rule_idx']} rule_w={e['rule_w']}")
