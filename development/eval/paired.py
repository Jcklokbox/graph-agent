"""Episode identities, plans and raw normalization for paired DEV evidence."""
import math

from envs.tasks import SPLITS
from eval import benchmark


def episode_key(tree, model, variant, seed):
    return tree, seed, f'{model}|DEV|{variant}'


def schedule(names, seeds):
    _, _, env_base, brain_base = SPLITS['dev']
    return [(name, seed, env_base + seed, brain_base + seed)
            for seed in seeds for name in names]


def run_episode(job, *, tree):
    model, name, seed, env_seed, brain_seed = job
    key = episode_key(tree, model, name, seed)
    return dict(k=list(key), m=benchmark.brain_run(name, env_seed, brain_seed, model))


def normalized_payoff(metrics):
    brain, stay, oracle = (metrics[k] for k in ('brain', 'stay', 'oracle'))
    if not all(math.isfinite(x) and 0. <= x <= 1. for x in (brain, stay, oracle)) or not oracle > stay:
        raise ValueError('invalid raw benchmark payoff or gap')
    return (brain - stay) / (oracle - stay)
