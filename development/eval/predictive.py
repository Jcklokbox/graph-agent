"""Tier T1: next-observation prediction against the exact Bayes filter.

Every predictor sees the same stream of a family world under the same uniformly
random policy (model learning is separated from control). Score per world and
seed, on the second half of the episode:

    excess log loss  = LL(agent) - LL(filter)              nats / channel / step
    gap closed (GC)  = (LL(marginal) - LL(agent)) / (LL(marginal) - LL(filter))

HGC = (LL(markov4) - LL(agent)) / (LL(markov4) - LL(filter)) isolates the part
of the predictable structure that needs memory beyond a 4-step window.

GC = 1: as good as the ideal observer that knows the true model; GC = 0: no
better than per-channel base rates. The value channel is excluded (constant in
the T1 worlds). Reference predictors: `marginal` (running base rates) and
`markov1` (Laplace counts per (active set, action): exact context, no memory),
`markov4` (the same over the last 4 observations, with back-off: a window).

    python -m eval.predictive                     # v11 + baselines, T1 set v1
    python -m eval.predictive --set v0            # the v0 set (regression floor)
    python -m eval.predictive --worlds hmm3,count3
    python -m eval.predictive --agents markov1 marginal --seeds 4 --steps 2000
    python -m eval.predictive --out results/t1_v11.json
    python -m eval.predictive --set d1 --agents kfact kjoint markov1 --steps 4000   # slice 3
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import agents
from envs.family import Filter, WorldEnv, WORLD_SETS
from eval.benchmark import pmap
from eval.stats import interval

EPS = 1e-4


class Marginal:
    def __init__(self, n, A, v, seed=0):
        self.k = np.ones(n)
        self.n = 2.

    def predict(self, x, a):
        return self.k / self.n

    def learn(self, x, a, y):
        self.k += np.asarray(y) > .5
        self.n += 1.


class Markov1:
    def __init__(self, n, A, v, seed=0):
        self.t = {}
        self.dim = n

    def predict(self, x, a):
        c = self.t.get((tuple(np.asarray(x) > .5), a))
        return np.full(self.dim, .5) if c is None else (c[0] + 1.) / (c[1] + 2.)

    def learn(self, x, a, y):
        key = (tuple(np.asarray(x) > .5), a)
        c = self.t.setdefault(key, [np.zeros(self.dim), 0.])
        c[0] += np.asarray(y) > .5
        c[1] += 1.


class MarkovK:
    """Window baseline: Laplace counts per (last k observations, action) with
    back-off to the longest order seen at least MIN times; k = 4. It recovers
    any hidden state that a 4-step window determines, nothing else."""
    K, MIN = 4, 8

    def __init__(self, n, A, v, seed=0):
        self.t = {}
        self.dim = n
        self.hist = []

    def _keys(self, x, a):
        h = self.hist[-(self.K - 1):] + [tuple(np.asarray(x) > .5)]
        return [(tuple(h[len(h) - j:]), a) for j in range(1, len(h) + 1)]

    def predict(self, x, a):
        best = None
        for key in self._keys(x, a):
            c = self.t.get(key)
            if c is not None and (best is None or c[1] >= self.MIN):
                best = c
        return np.full(self.dim, .5) if best is None else (best[0] + 1.) / (best[1] + 2.)

    def learn(self, x, a, y):
        for key in self._keys(x, a):
            c = self.t.setdefault(key, [np.zeros(self.dim), 0.])
            c[0] += np.asarray(y) > .5
            c[1] += 1.
        self.hist = (self.hist + [tuple(np.asarray(x) > .5)])[-(self.K - 1):]


class StepLearnAdapter:
    """An agent with step / learn / predict_next (e.g. v11); the action is forced
    to the policy's action after step(), so every predictor sees the same data."""

    def __init__(self, module, n, A, v, seed=0):
        self.b = agents.agent_class(module)(n, A, v, seed=seed)

    def predict(self, x, a):
        b = self.b
        b.step(tuple(float(u) for u in x))
        xn = b._pending[0]
        b._pending = (xn, a)
        return np.array(b.predict_next(xn, a))

    def learn(self, x, a, y):
        self.b.learn(tuple(float(u) for u in y))


class KnownStructure:
    """Reference learner that is TOLD the structure of the world (an oracle, not an agent):
    `kfact` — one online HMM (models/online_hmm.py, lag 4) per process, with the true channel
    partition and the true number of states; `kjoint` — one online HMM over all channels with
    K = the size of the joint state space. kfact is the learnable ceiling of a learner that
    has found the decomposition; kjoint pays the product of the state spaces."""

    def __init__(self, world, A, seed=0, joint=False):
        from models.online_hmm import OnlineHMM
        procs = [p for p in world.processes if p.n_states > 1]
        if world.context is not None and world.context.channels:
            procs = [world.context] + procs     # a visible context counts as a process
                                                # (kfact then assumes independence)
        links = getattr(world, "links", [])
        if joint:
            K = int(np.prod([p.n_states for p in procs]))
            chans = sorted([c for p in procs for c in p.channels] + [lk[0] for lk in links])
            self.parts = [(chans, OnlineHMM(len(chans), A, K=K, seed=seed))]
        else:
            # processes joined by a shared-emission channel form one group (K = product)
            idx = {id(p): i for i, p in enumerate(world.processes)}
            groups = [[p] for p in procs]
            extra = [[] for _ in groups]
            for ch, k1, k2, _ in links:
                g1 = next(i for i, g in enumerate(groups) if any(idx[id(q)] == k1 for q in g))
                g2 = next(i for i, g in enumerate(groups) if any(idx[id(q)] == k2 for q in g))
                if g1 != g2:
                    groups[g1] += groups[g2]
                    extra[g1] += extra[g2]
                    del groups[g2], extra[g2]
                    g1 = g1 if g1 < g2 else g1 - 1
                extra[g1].append(ch)
            self.parts = []
            for i, (g, ex) in enumerate(zip(groups, extra)):
                chans = sorted([c for p in g for c in p.channels] + ex)
                K = int(np.prod([p.n_states for p in g]))
                self.parts.append((chans, OnlineHMM(len(chans), A, K=K, seed=seed + 101 * i)))
        self.dim, self.started = world.n_channels, False

    def predict(self, x, a):
        x = np.asarray(x, float)
        if not self.started:
            for ch, h in self.parts:
                h.observe(x[ch])
            self.started = True
        out = np.full(self.dim, .5)
        for ch, h in self.parts:
            out[ch] = h.predict(a)
        return out

    def learn(self, x, a, y):
        y = np.asarray(y, float)
        for ch, h in self.parts:
            h.update(a, y[ch])


BASELINES = {"marginal": Marginal, "markov1": Markov1, "markov4": MarkovK}


def _probe(name):
    """hmm[K]-L[lag][-d[decay]] -> the online HMM probe model (models/online_hmm.py)."""
    from models.online_hmm import make_predictor
    kw = {}
    for part in name.split("-"):
        if part.startswith("hmm"):
            kw["K"] = int(part[3:])
        elif part.startswith("L"):
            kw["lag"] = int(part[1:])
        elif part.startswith("d"):
            kw["decay"] = float(part[1:])
    return make_predictor(name, **kw)


def make(name, n, A, v, seed, world=None):
    if name in ("decomp", "decompw", "decompf", "decompr", "decompc"):
        from models.decompose import Decomposer
        return Decomposer(n, A, v, seed, windows=name != "decomp", fast={"decompf": True, "decompr": "resid"}.get(name, False),
                            conj=name == "decompc")
    if name in ("kfact", "kjoint"):
        return KnownStructure(world, A, seed, joint=name == "kjoint")
    if name in BASELINES:
        return BASELINES[name](n, A, v, seed)
    if name.startswith("hmm"):
        return _probe(name)(n, A, v, seed)
    if name.startswith("factor"):
        from models.factor import make_factor
        return make_factor(name, n, A, v, seed)
    return StepLearnAdapter(agents.load(name), n, A, v, seed)


def run_one(world_name, agent_names, seed, steps, wset="v0"):
    w = WORLD_SETS[wset]()[world_name]
    env, filt = WorldEnv(w, 1000 + seed), Filter(w)
    pol = np.random.default_rng(2000 + seed)
    preds = {nm: make(nm, w.n_channels, w.n_actions, w.value_index, seed, w)
             for nm in agent_names}
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset()
    filt.observe(x)
    ll = {nm: np.zeros(steps) for nm in list(agent_names) + ["filter"]}
    sec = {nm: 0. for nm in agent_names}
    for t in range(steps):
        a = int(pol.integers(w.n_actions))
        p = {"filter": filt.predict(a)}
        for nm, pr in preds.items():
            t0 = time.time()
            p[nm] = pr.predict(x, a)
            sec[nm] += time.time() - t0
        y = np.array(env.step(a))
        for nm, q in p.items():
            q = np.clip(np.asarray(q, float), EPS, 1. - EPS)
            ll[nm][t] = -np.mean((y * np.log(q) + (1 - y) * np.log(1 - q))[ch])
        filt.update(a, y)
        for nm, pr in preds.items():
            t0 = time.time()
            pr.learn(x, a, y)
            sec[nm] += time.time() - t0
        x = tuple(y)
    half = steps // 2
    out = {nm: dict(ll=float(v[half:].mean()),
                    curve=[float(c.mean()) for c in np.array_split(v, 8)]) for nm, v in ll.items()}
    for nm in agent_names:
        out[nm]["sec"] = sec[nm]
        if hasattr(preds[nm], "partition"):
            out[nm]["partition"] = [list(map(int, g)) for g in preds[nm].partition()]
    return world_name, seed, out


def main(argv) -> int:
    names = ["v11", "markov1", "marginal"]
    if "--agents" in argv:
        i = argv.index("--agents") + 1
        names = []
        while i < len(argv) and not argv[i].startswith("--"):
            names.append(argv[i])
            i += 1
    for ref in ("markov4", "marginal"):
        if ref not in names:
            names.append(ref)
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 6
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else 2000
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    wset = argv[argv.index("--set") + 1] if "--set" in argv else "v1"
    worlds = list(WORLD_SETS[wset]())
    if "--worlds" in argv:
        worlds = argv[argv.index("--worlds") + 1].split(",")
    jobs = [(wn, names, s, steps, wset) for wn in worlds for s in range(seeds)]
    t0 = time.time()
    res = pmap(run_one, jobs)
    table = {}
    for wn, s, o in res:
        table.setdefault(wn, {})[s] = o
    summary = {}

    def score(nm, ref):
        return [(o[ref]["ll"] - o[nm]["ll"]) / (o[ref]["ll"] - o["filter"]["ll"])
                for o in table[wn].values()]

    shown = [nm for nm in names if nm != "marginal"]
    for metric, ref, title in [("gc", "marginal", "GC = gap closed between base rates (0) and the exact filter (1)"),
                               ("hgc", "markov4", "HGC = hidden gap closed between the 4-step window (0) and the exact filter (1)")]:
        print(f"T1 {wset}: {seeds} seeds x {steps} steps, random policy, second half; {title}")
        print(f"{'world':<18}" + "".join(f"{nm:>22}" for nm in shown if nm != ref)
              + f"{'filter LL':>11}{'gap':>7}")
        for wn in worlds:
            row = f"{wn:<18}"
            for nm in shown:
                if nm == ref:
                    continue
                v = score(nm, ref)
                m, sd, ci = interval(v)
                summary.setdefault(nm, {}).setdefault(wn, {})[metric] = dict(v=v, mean=m, ci=ci)
                row += f"{m:9.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
            fl = np.mean([o["filter"]["ll"] for o in table[wn].values()])
            gap = np.mean([o[ref]["ll"] - o["filter"]["ll"] for o in table[wn].values()])
            print(row + f"{fl:11.3f}{gap:7.3f}")
        print(f"{'mean':<18}" + "".join(
            f"{np.mean([summary[nm][wn][metric]['mean'] for wn in worlds]):9.2f}{'':13}"
            for nm in shown if nm != ref))
        print()
    secs = {nm: np.mean([o[nm]["sec"] for wo in table.values() for o in wo.values()])
            for nm in names}
    print("sec per episode: " + "  ".join(f"{nm} {v:.1f}" for nm, v in secs.items())
          + f"   wall {time.time() - t0:.0f}s")
    if out:
        with open(out, "w") as f:
            json.dump(dict(set=wset, seeds=seeds, steps=steps, agents=names, table=table,
                           summary=summary), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
