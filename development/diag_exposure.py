"""Matched exposure fixture: free vs payoff-prefix vs adverse-prefix on canc seed6."""
import sys
sys.path.insert(0, 'development')
import numpy as np
import agents
from envs.tasks import VARIANTS, cancellation_oracle

def run(env_seed, brain_seed, agent_name, mode, steps=600, prefix_steps=150):
    v = VARIANTS['canc']
    env = v.make(env_seed)
    mod = agents.load(agent_name)
    b = agents.agent_class(mod)(v.n_inputs, v.n_actions, 11, seed=brain_seed)
    obs = env.reset()
    oracle = cancellation_oracle
    R = []
    first_payoff = None
    payoff_count = 0
    cue_count = 0
    for t in range(steps):
        cue = obs[3] > .5
        if cue:
            cue_count += 1
        if mode == 'free':
            a = b.step(obs)
        elif mode == 'payoff':
            if t < prefix_steps:
                # scripted oracle prefix (real payoff via ordinary action)
                a = oracle(obs)
                # need to advance agent's pending without calling step (which would choose)
                # Use factor directly: predict + learn pattern
                # Agent.step does: q, choose, predict, set pending. We replicate with forced a.
                f = b.factor
                f.predict(obs, a)
                b.pending = (obs, a)
            else:
                a = b.step(obs)
        elif mode == 'adverse':
            if t < prefix_steps:
                a = 1  # stay: misses payoff, gets 0.5 (null prefix, same timing)
                f = b.factor
                f.predict(obs, a)
                b.pending = (obs, a)
            else:
                a = b.step(obs)
        else:
            raise ValueError(mode)
        obs2, rew, _, info = env.step(a)
        b.learn(obs2)
        R.append(float(rew))
        if rew == 1.0:
            payoff_count += 1
            if first_payoff is None:
                first_payoff = t
        obs = obs2
    tail = float(np.mean(R[-200:]))
    return dict(first_payoff=first_payoff, payoff_count=payoff_count, cues=cue_count,
                tail=tail, mean=float(np.mean(R)))

if __name__ == '__main__':
    for agent_name in ['v48', 'v44']:
        for mode in ['free', 'payoff', 'adverse']:
            r = run(106, 6, agent_name, mode)
            print(f"{agent_name} seed6 {mode}: first={r['first_payoff']} payoffs={r['payoff_count']} tail={r['tail']:.4f} mean={r['mean']:.4f}", flush=True)
    # also seed0 free as positive control
    for agent_name in ['v48']:
        r = run(100, 0, agent_name, 'free')
        print(f"{agent_name} seed0 free: first={r['first_payoff']} payoffs={r['payoff_count']} tail={r['tail']:.4f}", flush=True)
