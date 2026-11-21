"""Guard comparison v44 vs v48 on exposed seeds (small probe, not DEV-16)."""
import sys
sys.path.insert(0, 'development')
import numpy as np
import agents
from envs.tasks import VARIANTS

def brain_run(name, env_seed, brain_seed, agent_name):
    v = VARIANTS[name]
    env = v.make(env_seed)
    mod = agents.load(agent_name)
    b = agents.agent_class(mod)(v.n_inputs, v.n_actions, 11, seed=brain_seed)
    obs = env.reset()
    R = []
    for _ in range(v.steps):
        a = b.step(obs)
        obs, r, _, info = env.step(a)
        b.learn(obs)
        R.append(float(r))
    lo = v.steps - 200
    return float(np.mean(R[lo:]))

def stay_oracle(name, env_seed):
    v = VARIANTS[name]
    env = v.make(env_seed)
    obs = env.reset()
    R = []
    for _ in range(v.steps):
        obs, r, _, _ = env.step(1)
        R.append(float(r))
    lo = v.steps - 200
    stay = float(np.mean(R[lo:]))
    # oracle
    env2 = v.make(env_seed)
    obs = env2.reset()
    pol = v.oracle()
    R2 = []
    for _ in range(v.steps):
        obs, r, _, _ = env2.step(pol(obs))
        R2.append(float(r))
    oracle = float(np.mean(R2[lo:]))
    return stay, oracle

if __name__ == '__main__':
    variants = ['canc', 'canc-noise3', 'aut', 'dist', 'aut-noise4', 'dist-noise2']
    seeds = [0, 6]
    for name in variants:
        for s in seeds:
            env_seed = 100 + s
            brain_seed = s
            stay, oracle = stay_oracle(name, env_seed)
            for agent_name in ['v48', 'v44']:
                brain = brain_run(name, env_seed, brain_seed, agent_name)
                score = (brain - stay) / (oracle - stay) if oracle - stay > 1e-9 else float('nan')
                print(f"{name} seed{s} {agent_name}: brain={brain:.4f} stay={stay:.4f} oracle={oracle:.4f} score={score:.3f}", flush=True)
