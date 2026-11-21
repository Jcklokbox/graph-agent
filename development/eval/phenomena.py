"""Run the T2 phenomena (envs/phenomena.py) on agents.

Agents: `rw` (Rescorla-Wagner on the US channel; a check of the harness: it is
known to show blocking, conditioned inhibition and contingency effects and to
lack spontaneous recovery and savings) and any agent module (e.g. `v11`).

    python -m eval.phenomena --agents rw v11 --seeds 16 [--out results/t2.json]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import agents
from envs.phenomena import CUES, HOME, N_ACT, SETS, US, VALUE
from eval.benchmark import pmap
from eval.stats import interval


# a reproduced effect: 95 % CI above 0 and mean >= MIN_EFFECT (.02 in v0, .05 in v1: SETS)


class RescorlaWagner:
    """V(US | active cues) = sum of weights (contexts are cues), delta rule."""

    def __init__(self, n, alpha=.2):
        self.w = np.zeros(n)
        self.alpha = alpha
        self.x = None

    def step(self, x):
        self.x = np.asarray(x) > .5
        self.x[[US, VALUE]] = False
        return 0

    def learn(self, y):
        v = self.w[self.x].sum()
        self.w[self.x] += self.alpha * (float(y[US] > .5) - v)

    def predict_us(self, x):
        m = np.asarray(x) > .5
        m[[US, VALUE]] = False
        return float(np.clip(self.w[m].sum(), 0., 1.))


class ModuleAgent:
    def __init__(self, name, seed, n):
        self.b = agents.agent_class(agents.load(name))(n, N_ACT, VALUE, seed=seed)

    def step(self, x):
        return self.b.step(x)

    def learn(self, y):
        self.b.learn(y)

    def predict_us(self, x):
        return float(np.mean([self.b.predict_next(x, a)[US] for a in range(N_ACT)]))


class DecompAgent:
    """T2 adapter for the slice-3 witness (models/decompose.py; a predictor, action 0)."""

    def __init__(self, name, seed, n):
        from models.decompose import Decomposer
        self.d = Decomposer(n, N_ACT, VALUE, seed, windows=name != "decomp",
                            fast={"decompf": True, "decompr": "resid"}.get(name, False),
                            conj=name == "decompc")
        self.x = None

    def step(self, x):
        self.x = x
        return 0

    def learn(self, y):
        self.d.predict(self.x, 0)
        self.d.learn(self.x, 0, y)

    def predict_us(self, x):
        if self.x is None:
            return .5
        return self.d.peek(x, 0, US)


class FactorAgent:
    """Read out the canonical factor predictor at action zero."""

    def __init__(self, name, seed, n):
        from models.factor import make_factor
        self.m = make_factor(name, n, N_ACT, VALUE, seed)
        self.x = None

    def step(self, x):
        self.x = x
        return 0

    def learn(self, y):
        self.m.learn(self.x, 0, y)

    def predict_us(self, x):
        return float(self.m.predict(x, 0, hypothetical=True)[US])


class ConfigCounts:
    """Trivial reference: Laplace counts of the US per complete configuration (cues and
    contexts), no generalisation, no memory."""

    def __init__(self):
        self.t, self.x = {}, None

    @staticmethod
    def _key(x):
        return tuple(v > .5 for i, v in enumerate(x) if i not in (US, VALUE))

    def step(self, x):
        self.x = self._key(x)
        return 0

    def learn(self, y):
        c = self.t.setdefault(self.x, [0., 0.])
        c[0] += y[US] > .5
        c[1] += 1.

    def predict_us(self, x):
        c = self.t.get(self._key(x), [0., 0.])
        return (c[0] + 1.) / (c[1] + 2.)


def make(name, seed, n):
    """rw, cfg, lc (latent-cause witness, hazard learned on a grid), lc-h<h> (fixed hazard),
    lc-f (home steps not emitted by the box context), or an agent module."""
    if name == "rw":
        return RescorlaWagner(n)
    if name == "cfg":
        return ConfigCounts()
    if name.startswith("lc-sub"):
        # D-17: sticky-CRP causes x structure averaging over parent sets of size <= 2;
        # anonymous cues = all other channels, nothing emitted (as lc-anon).
        from models.latent_cause_lin import LatentCauseSub, LatentCauseSubAdd
        others = [c for c in range(n) if c not in (US, VALUE)]
        from models.latent_cause_lin import LatentCauseSubAddM, LatentCauseSubAddP
        cls = LatentCauseSubAdd if name.startswith("lc-sub-add") else LatentCauseSub
        if name.startswith("lc-sub-add-M"):          # F-11: Markov emission of all other channels
            cls = LatentCauseSubAddM
        if name.startswith("lc-sub-add-P"):          # F-12: place factor x regime factor
            cls = LatentCauseSubAddP
        npart = int(name.split("-n")[-1]) if "-n" in name else 1000     # lc-sub[-add]-n<particles>
        emit = others if ("-e" in name.replace("-n", "#") or "-E" in name or "-M" in name or "-P" in name) else []     # lc-sub-add-e: each cause also models all other channels
        hier = "emit" if "-E" in name else False     # lc-sub-add-eE: new cause's emission prior = pooled known causes
        return cls(emit=emit, cues=others, us=US, hazard=None, n=npart, seed=seed, hier=hier)
    if name.startswith("lc-lin"):
        # D-17: additive US model per latent cause; anonymous cues = all other channels;
        # lc-lin emits all other channels (each cause also models them), lc-lin-x emits none.
        from models.latent_cause_lin import LatentCauseLin
        others = [c for c in range(n) if c not in (US, VALUE)]
        return LatentCauseLin(emit=[] if name.endswith("-x") else others, cues=others, us=US,
                              hazard=None, n=1000, seed=seed)
    if name.startswith("lc-anon"):
        # D-16: no role knowledge — no emitted features, no home channel; the US is predicted
        # from the configuration of ALL other channels (what an agent predicting every channel
        # would do for this one); causes are inferred from contingency changes only.
        from models.latent_cause import LatentCauseSMC
        others = [c for c in range(n) if c not in (US, VALUE)]
        return LatentCauseSMC(emit=[], cues=others, us=US, hazard=None, n=1000, seed=seed)
    if name.startswith("lc"):
        # lc[-f][-H][-h<hazard>]: f = home steps not emitted by the box context,
        # H = hierarchical prior over contexts (E: on the emitted features only), h = fixed hazard (default: learned on a grid)
        from models.latent_cause import LatentCauseSMC
        parts = name.split("-")[1:]
        h = next((float(p[1:]) for p in parts if p.startswith("h") and len(p) > 1), None)
        return LatentCauseSMC(emit=[c for c in range(n) if c not in (US, VALUE)],
                              cues=CUES, us=US, hazard=h, n=2000, seed=seed,
                              away=HOME if "f" in parts and n > HOME else None,
                              hier="emit" if "E" in parts else "H" in parts)
    if name.startswith("factor"):
        return FactorAgent(name, seed, n)
    if name in ("decomp", "decompw", "decompf", "decompr", "decompc"):
        return DecompAgent(name, seed, n)
    return ModuleAgent(name, seed, n)


def run_group(agent_name, phen, group, seed, wset="v0", trace=False):
    """trace=True: also return the model's trace() at every probe point (deterministic)."""
    table, n, _ = SETS[wset]
    s = table[phen].build(group, seed)
    ag = make(agent_name, seed, n)
    probes, traces = {}, {}
    tr = trace and hasattr(getattr(ag, "m", None), "trace")
    by_index = {}
    for idx, name, x in s.probes:
        by_index.setdefault(idx, []).append((name, x))
    for idx, name, x in by_index.get(0, []):
        probes[name] = ag.predict_us(x)
    for t in range(len(s.obs) - 1):
        ag.step(s.obs[t])
        ag.learn(s.obs[t + 1])
        for name, x in by_index.get(t + 1, []):
            probes[name] = ag.predict_us(x)
            if tr:
                traces[name] = ag.m.trace()
    if trace:
        return agent_name, phen, group, seed, probes, traces
    return agent_name, phen, group, seed, probes


def main(argv) -> int:
    names = ["rw", "v11"]
    if "--agents" in argv:
        i, names = argv.index("--agents") + 1, []
        while i < len(argv) and not argv[i].startswith("--"):
            names.append(argv[i])
            i += 1
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 16
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    wset = argv[argv.index("--set") + 1] if "--set" in argv else "v0"
    PHENOMENA, _, MIN_EFFECT = SETS[wset]
    only = argv[argv.index("--rows") + 1].split(",") if "--rows" in argv else list(PHENOMENA)
    PHENOMENA = {k: v for k, v in PHENOMENA.items() if k in only}
    jobs = [(nm, ph, g, s, wset) for nm in names for ph, P in PHENOMENA.items()
            for g in P.groups for s in range(seeds)]
    t0 = time.time()
    res = pmap(run_group, jobs)
    probes = {}
    for nm, ph, g, s, pr in res:
        probes.setdefault(nm, {}).setdefault(ph, {}).setdefault(s, {})[g] = pr
    summary = {}
    print(f"T2 {wset}: {seeds} seeds; effect = predicted sign from the literature (> 0); "
          f"verdict: 'yes' if the 95% CI is above 0 and mean >= {MIN_EFFECT}, 'REV' if reversed, '-' otherwise")
    print(f"{'phenomenon':<22}" + "".join(f"{nm:>26}" for nm in names))
    for ph, P in PHENOMENA.items():
        row = f"{ph:<22}"
        for nm in names:
            eff = [P.effect(probes[nm][ph][s]) for s in range(seeds)]
            m, _, ci = interval(eff)
            bar = MIN_EFFECT if P.min_effect is None else P.min_effect
            verdict = ("yes" if ci[0] > 0 and m >= bar else
                       "REV" if ci[1] < 0 and m <= -bar else "-")
            summary.setdefault(nm, {})[ph] = dict(effect=eff, mean=m, ci=ci, verdict=verdict)
            row += f"{m:+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {verdict:>3}"
        print(row)
    print(f"wall {time.time() - t0:.0f}s")
    if out:
        with open(out, "w") as f:
            json.dump(dict(set=wset, seeds=seeds, agents=names, probes=probes, summary=summary), f,
                      indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
