"""Transfer benchmark (tier T3): an agent on variations of the base tasks.

Score per variant: normalised tail return (brain - stay) / (oracle - stay),
stay = always action 1. Tail = last 200 steps; for serial reversal every step
after the first phase (adaptation is part of the task). Run from development/:

    python -m eval.benchmark --dev                     # 15 DEV variants x 8 seeds
    python -m eval.benchmark --dev --seeds 16          # DEV with seeds 0..15 (power)
    python -m eval.benchmark --dev --seeds 4           # screening only (first 4 seeds)
    python -m eval.benchmark --held6                   # a held-out split (used once)
    python -m eval.benchmark --dev aut dist            # selected variants
    python -m eval.benchmark --agent v11               # agents/v11.py (default)
    python -m eval.benchmark --dev --kw '{"explore": 1.5}'
    python -m eval.benchmark --dev --set NAME=json     # set an agent module attribute
    python -m eval.benchmark --dev --out results/x.json  # per-seed results (eval.stats)

Episodes run in a process pool (PG_PROCS processes, default: all CPUs).
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager

import numpy as np

import agents
from envs.tasks import SPLITS, VARIANTS, blind_gate_oracle
from eval.parallel import pmap

VALUE_INDEX = 11


@contextmanager
def _module_overrides(module, overrides):
    """Scope legacy --set attributes to one benchmark run (fork workers inherit them)."""
    missing = object()
    before = {key: vars(module).get(key, missing) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in before.items():
            if value is missing:
                vars(module).pop(key, None)
            else:
                setattr(module, key, value)


def _window(name, n):
    if name.startswith("aut-serial"):
        return int(name[len("aut-serial"):]), n
    return n - 200, n


def _shock(info):
    return bool(info.get("received_shock", False) if isinstance(info, dict)
                else info.received_shock)


def _policy_return(name, env_seed, policy):
    v = VARIANTS[name]
    env = v.make(env_seed)
    obs = env.reset()
    R = []
    for _ in range(v.steps):
        obs, r, _, _ = env.step(policy(obs))
        R.append(float(r))
    lo, hi = _window(name, v.steps)
    return float(np.mean(R[lo:hi]))


def _oracle_return(name, env_seed):
    v = VARIANTS[name]
    if v.oracle is not None:
        return _policy_return(name, env_seed, v.oracle())
    # Serial reversal: the oracle acts on the un-relabelled world.
    env = v.make(env_seed)
    env.reset()
    R = []
    for _ in range(v.steps):
        if env.t > 0 and env.t % env.period == 0:
            env.env.reverse()
        env.t += 1
        inner = env.env.env
        _, r, _, _ = inner.step(blind_gate_oracle(inner._observation()))
        R.append(float(r))
    lo, hi = _window(name, v.steps)
    return float(np.mean(R[lo:hi]))


def brain_run(name, env_seed, brain_seed, agent, settings=None):
    v = VARIANTS[name]
    env = v.make(env_seed)
    b = agents.agent_class(agents.load(agent))(v.n_inputs, v.n_actions, VALUE_INDEX,
                                             seed=brain_seed, **(settings or {}))
    obs = env.reset()
    R, S = [], []
    t0 = time.time()
    for _ in range(v.steps):
        a = b.step(obs)
        obs, r, _, info = env.step(a)
        b.learn(obs)
        R.append(float(r))
        S.append(_shock(info))
    lo, hi = _window(name, v.steps)
    return dict(brain=float(np.mean(R[lo:hi])), stay=_policy_return(name, env_seed, lambda o: 1),
                oracle=_oracle_return(name, env_seed), shocks=float(np.mean(S)),
                late_shocks=float(np.mean(S[lo:hi])), sec=time.time() - t0)


def run(split, names=None, verbose=True, n_seeds=None, out=None, *,
        agent="v11", settings=None, module_sets=None):
    settings = {} if settings is None else settings
    module_sets = {} if module_sets is None else module_sets
    all_names, seeds, eb, bb = SPLITS[split]
    names = names or all_names
    if n_seeds is not None:
        seeds = range(n_seeds)
    jobs = [(nm, eb + s, bb + s) for nm in names for s in seeds]
    jobs.sort(key=lambda j: -VARIANTS[j[0]].steps * VARIANTS[j[0]].n_inputs ** 2)  # longest first
    with _module_overrides(agents.load(agent), module_sets):
        metrics = pmap(brain_run, [(*job, agent, settings) for job in jobs])
    res = dict(zip(jobs, metrics))
    rows = {}
    for nm in names:
        rs = [res[(nm, eb + s, bb + s)] for s in seeds]
        norm = [(r["brain"] - r["stay"]) / (r["oracle"] - r["stay"])
                if r["oracle"] - r["stay"] > 1e-9 else float("nan") for r in rs]
        rows[nm] = dict(seeds=list(seeds), per_seed=norm, raw=rs, score=float(np.mean(norm)),
                        wins=f"{sum(r['brain'] > r['stay'] + 1e-9 for r in rs)}/{len(rs)}",
                        **{k: float(np.mean([r[k] for r in rs]))
                           for k in ("brain", "stay", "oracle", "shocks", "late_shocks", "sec")})
    if verbose:
        print(f"{'variant':<26}{'score':>7}{'wins':>7}{'brain':>7}{'stay':>7}{'oracle':>7}"
              f"{'shock':>7}{'late':>7}{'s/ep':>7}")
        for nm, r in rows.items():
            print(f"{nm:<26}{r['score']:7.3f}{r['wins']:>7}{r['brain']:7.3f}{r['stay']:7.3f}"
                  f"{r['oracle']:7.3f}{r['shocks']:7.3f}{r['late_shocks']:7.3f}{r['sec']:7.1f}")
        tr = [r["score"] for nm, r in rows.items() if nm not in ("aut", "canc", "dist")]
        print(f"mean score: all {np.mean([r['score'] for r in rows.values()]):.3f}"
              + (f" | variants only {np.mean(tr):.3f}" if tr else ""))
    if out:
        with open(out, "w") as f:
            json.dump(dict(split=split, agent=agent, settings=settings,
                           module_sets=module_sets, variants=rows), f, indent=1)
        if verbose:
            print("per-seed results ->", out)
    return rows


def main(argv):
    settings = json.loads(argv[argv.index("--kw") + 1]) if "--kw" in argv else {}
    agent = argv[argv.index("--agent") + 1] if "--agent" in argv else "v11"
    module_sets = {}
    for i, a in enumerate(argv):
        if a == "--set":
            k, v = argv[i + 1].split("=", 1)
            module_sets[k] = json.loads(v)
    split = next((a[2:] for a in argv if a[2:] in SPLITS and a.startswith("--")), "dev")
    only = [a for a in argv if a in VARIANTS]
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else None
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    if n_seeds is not None:
        if split != "dev":
            sys.exit("--seeds is a DEV option; held-out splits use all their seeds")
        if n_seeds < len(SPLITS[split][1]):
            print(f"SCREENING ONLY: first {n_seeds} of {len(SPLITS[split][1])} seeds")
    print("split:", split, "agent:", agent, "settings:", settings, module_sets)
    run(split, only or None, n_seeds=n_seeds, out=out,
        agent=agent, settings=settings, module_sets=module_sets)


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        main(sys.argv[1:])
