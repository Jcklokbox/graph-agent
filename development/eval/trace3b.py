"""[0] Hot-path & trace localization (SPEC-3b.2): deterministic trace(state) at s = 0 and the
exact divergence terms of M_factor against the witnesses, before models/factor.py is edited.

    python3 -m eval.trace3b --t1 d3 --t2 reinstatement,spont-recovery-sessions,cond-inhibition

Prints, per item, the term-by-term comparison (per-channel log loss on T1, probe values and the
model's own trace on T2).  Read-only: it never writes results/ and never touches the journal.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

EPS = 1e-4


# --------------------------------------------------------------------- T1: per-channel divergence
def t1_channels(world_name, agents, seed=0, steps=4000, wset="d3"):
    """Per-channel excess log loss (nats/channel/step, second half) of each agent vs the exact
    filter and vs the 4-step window, plus the factor model's slots at the end."""
    from envs.family import WORLD_SETS, WorldEnv, Filter
    from eval.predictive import make
    w = WORLD_SETS[wset]()[world_name]
    env, filt = WorldEnv(w, 1000 + seed), Filter(w)
    pol = np.random.default_rng(2000 + seed)
    preds = {nm: make(nm, w.n_channels, w.n_actions, w.value_index, seed, w) for nm in agents}
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset()
    filt.observe(x)
    ll = {(nm, c): np.zeros(steps) for nm in list(agents) + ["filter", "markov4"] for c in ch}
    mk = make("markov4", w.n_channels, w.n_actions, w.value_index, seed, w)
    for t in range(steps):
        a = int(pol.integers(w.n_actions))
        p = {"filter": filt.predict(a)}
        p["markov4"] = mk.predict(x, a)
        for nm, pr in preds.items():
            p[nm] = pr.predict(x, a)
        y = np.array(env.step(a))
        for nm, q in p.items():
            q = np.clip(np.asarray(q, float), EPS, 1. - EPS)
            for c in ch:
                ll[(nm, c)][t] = -(y[c] * np.log(q[c]) + (1 - y[c]) * np.log(1 - q[c]))
        filt.update(a, y)
        mk.learn(x, a, y)
        for nm, pr in preds.items():
            pr.learn(x, a, y)
        x = tuple(y)
    half = steps // 2
    out = {"world": world_name, "seed": seed, "steps": steps, "chan": {}}
    for c in ch:
        row = {}
        for nm in list(agents) + ["filter", "markov4"]:
            row[nm] = round(float(ll[(nm, c)][half:].mean()), 4)
        row["excess"] = {nm: round(row[nm] - row["filter"], 4) for nm in agents}
        row["excess_vs_markov4"] = {nm: round(row[nm] - row["markov4"], 4) for nm in agents}
        out["chan"][c] = row
    for nm in agents:                      # structure the learner ended with
        pr = preds[nm]
        if hasattr(pr, "slots"):
            out["slots_" + nm] = [[list(s) if s else None for s in sl] for sl in pr.slots]
        if hasattr(pr, "partition"):
            out["partition_" + nm] = [list(map(int, g)) for g in pr.partition()]
    return out


# --------------------------------------------------------------------- T2: probe + trace divergence
def t2_trace(row, agents, seed=0, wset="v1"):
    from envs.phenomena import SETS
    from eval.phenomena import make
    table, n, _ = SETS[wset]
    P = table[row]
    out = {"row": row, "seed": seed, "groups": {}}
    for g in P.groups:
        s = P.build(g, seed)
        by_index = {}
        for idx, name, x in s.probes:
            by_index.setdefault(idx, []).append((name, x))
        probes, traces = {}, {}
        for nm in agents:
            ag = make(nm, seed, n)
            m = getattr(ag, "m", None)          # ndarray.m has a .trace() too (matrix trace):
            tr = type(m).__name__ == "FactorModel"   # only the factor model's trace is meant
            for name, x in by_index.get(0, []):
                probes[(nm, name)] = ag.predict_us(x)
            for t in range(len(s.obs) - 1):
                ag.step(s.obs[t])
                ag.learn(s.obs[t + 1])
                for name, x in by_index.get(t + 1, []):
                    probes[(nm, name)] = ag.predict_us(x)
                    if tr:
                        traces[name] = ag.m.trace()
        out["groups"][g] = dict(probes={f"{k[0]}|{k[1]}": round(float(v), 4)
                                        for k, v in sorted(probes.items())},
                                traces=traces, n_obs=len(s.obs))
    out["effect"] = {nm: round(float(P.effect(
        {g: {k.split("|", 1)[1]: v for k, v in out["groups"][g]["probes"].items()
             if k.startswith(nm + "|")} for g in P.groups})), 4) for nm in agents}
    return out


# ------------------------------------------------- T1: which member of M_factor loses on a channel
def t1_members(world_name, seed=0, steps=2000, wset="d3", agent="factor-L4-P-C"):
    """Per channel: log loss of the M_factor mixture vs the log loss of its process member alone
    (the decompw witness it contains) and vs the exact filter; plus the final member weights.
    Isolates whether the divergence is in a member's prediction or in the member weights."""
    from envs.family import WORLD_SETS, WorldEnv, Filter
    import models.factor as F
    from eval.predictive import make

    class Instrumented(F.FactorModel):
        def _members(self, b, a, hist, kk, pp=None):
            mem = F.FactorModel._members(self, b, a, hist, kk, pp)
            self._lastP = mem["P"][:, :, self.S + 1, 0].astype(float)   # proc member, (kk, J)
            return mem

    cached, F.FactorModel = F.FactorModel, Instrumented
    w = WORLD_SETS[wset]()[world_name]
    env, filt = WorldEnv(w, 1000 + seed), Filter(w)
    pol = np.random.default_rng(2000 + seed)
    try:
        pr = make(agent, w.n_channels, w.n_actions, w.value_index, seed, w)
    finally:
        F.FactorModel = cached
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset()
    filt.observe(x)
    J = len(ch)
    ll = np.zeros((steps, J))          # the M_factor mixture
    llp = np.zeros((steps, J))         # its process member alone (the decompw witness inside it)
    llf = np.zeros((steps, J))         # the exact filter
    for t in range(steps):
        a = int(pol.integers(w.n_actions))
        p = np.clip(np.asarray(pr.predict(x, a), float)[ch], EPS, 1. - EPS)
        pf = np.clip(np.asarray(filt.predict(a), float)[ch], EPS, 1. - EPS)
        y = np.array(env.step(a))
        yb = y[ch] > .5
        q = np.clip(pr._lastP, EPS, 1. - EPS)               # (kk, J) proc member predictions
        ll[t] = -(yb * np.log(p) + ~yb * np.log(1. - p))
        llf[t] = -(yb * np.log(pf) + ~yb * np.log(1. - pf))
        llp[t] = -(yb * np.log(q[0]) + ~yb * np.log(1. - q[0]))
        filt.update(a, y)
        pr.learn(x, a, y)
        x = tuple(y)
    half = steps // 2
    out = {"world": world_name, "seed": seed, "steps": steps, "chan": {}}
    wgt = np.exp(pr.ev[0] - pr.ev[0].max(1, keepdims=True))
    pw = np.exp(pr.logw - pr.logw.max())
    pw /= pw.sum()
    E = ((wgt / wgt.sum(1, keepdims=True)) * pw[None, None, :]).sum(2)     # (J, M) weights
    for i, c in enumerate(ch):
        out["chan"][c] = dict(
            ll_mix=round(float(ll[half:, i].mean()), 4),
            ll_proc=round(float(llp[half:, i].mean()), 4),
            ll_filter=round(float(llf[half:, i].mean()), 4),
            dilution=round(float(ll[half:, i].mean() - llp[half:, i].mean()), 4),
            w_sets=round(float(E[i, :pr.S].sum()), 3), w_add=round(float(E[i, pr.S]), 3),
            w_proc=round(float(E[i, pr.S + 1]), 3), w_place=round(float(E[i, pr.S + 2]), 3),
            slots=[list(s) if s else None for s in pr.slots[i]])
    return out


# ------------------------------------------- T2: term-by-term decomposition of one probe value
def t2_probe(row, seed=0, wset="v1", agent="factor-L4-P-C-N25"):
    """At every probe of a T2 row, decompose the US prediction into its terms: the (place, regime)
    posterior the probe induces and, per regime, what each member says about the US. This is the
    divergence equation for the row (SPEC-3b.2 [0])."""
    from envs.phenomena import SETS, US
    from eval.phenomena import make
    import models.factor as F
    table, n, _ = SETS[wset]
    P = table[row]
    out = {"row": row, "seed": seed, "groups": {}}
    for g in P.groups:
        s = P.build(g, seed)
        by_index = {}
        for idx, name, x in s.probes:
            by_index.setdefault(idx, []).append((name, x))
        ag = make(agent, seed, n)
        m = ag.m
        rec = {}

        def probe_dec(m, x, a):
            """Recompute predict(x, a, hypothetical=True) with the terms exposed."""
            b = m._bits(x)
            kk = min(m.K, int(m.used.max()) + 1)
            w = np.exp(m.logw - m.logw.max())
            pp = m.proc.peek_all(x, a)[m.ch] if m.proc is not None else None
            prev = m.hist[-1]
            mem0 = m._members(prev, 0, m.hist[:-1], kk, m._pp)
            pj0 = m._joint(mem0, m._place_pred(prev), kk)
            lik = np.prod(np.where((b > .5)[None, None, None, :], pj0, 1. - pj0), axis=3)
            wy = m._place_prior()[:, :, None] * m._regime_prior(kk) * lik
            mem = m._members(b, a, m.hist, kk, pp)
            pj = m._joint(mem, m._place_pred(b), kk)              # (N, KP, kk, J)
            jw = w[:, None, None] * wy
            tot = jw.sum()
            ui = m.ch.index(US)
            q = np.einsum("npk,npkj->j", jw, pj) / tot
            wz = jw.sum((0, 1)) / tot                             # regime marginal
            wy_ = jw.sum((0, 2)) / tot                            # place marginal
            per = np.einsum("npk,npkj->kj", jw, pj) / np.maximum(jw.sum((0, 1)), 1e-300)[:, None]
            return dict(p_us=round(float(q[ui]), 4),
                        place=[round(float(u), 3) for u in wy_[:5]],
                        regime=[round(float(u), 3) for u in wz[:kk]],
                        us_by_regime=[round(float(u), 3) for u in per[:kk, ui]],
                        dtau=m.dtau, gate=[round(float(u), 2) for u in m.g],
                        slots_us=[list(sl) if sl else None for sl in m.slots[ui]],
                        used=round(float((w * m.used).sum()), 2),
                        usedp=round(float((w * m.usedp).sum()), 2))

        for name, x in by_index.get(0, []):
            rec[name] = probe_dec(m, x, 0)
        for t in range(len(s.obs) - 1):
            ag.step(s.obs[t])
            ag.learn(s.obs[t + 1])
            for name, x in by_index.get(t + 1, []):
                rec[name] = probe_dec(m, x, 0)
        out["groups"][g] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", default="")
    ap.add_argument("--t2", default="")
    ap.add_argument("--agents", default="factor-L4-P-C")
    ap.add_argument("--witness", default="decompw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--set", default="d3")
    ap.add_argument("--members", default="")
    ap.add_argument("--probe", default="")
    a = ap.parse_args(argv)
    res = {}
    if a.probe:
        for row in a.probe.split(","):
            res["probe:" + row] = t2_probe(row, a.seed, "v1", a.agents.split(",")[0])
    if a.members:
        for wn in a.members.split(","):
            res["mem:" + wn] = t1_members(wn, a.seed, a.steps, a.set, a.agents.split(",")[0])
    if a.t1:
        for wn in a.t1.split(","):
            res["t1:" + wn] = t1_channels(wn, [x for x in a.agents.split(",")] + [a.witness],
                                          a.seed, a.steps, a.set)
    if a.t2:
        for row in a.t2.split(","):
            res["t2:" + row] = t2_trace(row, a.agents.split(",") + a.witness.split(","),
                                        a.seed, "v1")
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
