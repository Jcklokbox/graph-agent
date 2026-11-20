"""SPEC-3b.2 [0]: deterministic trace(state) localization (diagnostics, no model change).

For every row at seed 0, read the model's own state at the probe points and compare the two
families' predictions; for d3 compare factor-L4-P-C with decompw on the same stream and read
the state at the steps where the per-step log loss diverges most.

    python3 -m harness.localize            # T2 rows (factor vs lc-sub-add / decompw)
    python3 -m harness.localize d3         # d3-and (factor vs decompw), per-step attribution
"""
import json
import sys

import numpy as np

ROWS = (("reinstatement", ("reinst", "control")), ("spont-recovery-sessions", ("short", "long")),
        ("cond-inhibition", ("one",)), ("d3", ()))


def t2(wset="v1"):
    from eval.phenomena import run_group
    out = {}
    for row, groups in ROWS[:3]:
        for g in groups:
            got = {}
            for ag in ("factor-L4-P-C-N200", "lc-sub-add", "decompw"):
                nm, ph, gr, sd, probes, traces = run_group(ag, row, g, 0, wset, trace=True)
                got[ag] = (probes, traces)
            pf, tf = got["factor-L4-P-C-N200"]
            row_out = {}
            for ag in ("lc-sub-add", "decompw"):
                pw = got[ag][0]
                row_out[ag] = {k: round(pf[k] - pw[k], 4) for k in pf}
            last = tf[list(tf)[-1]]
            out[f"{row}|{g}"] = dict(probes={k: round(v, 4) for k, v in pf.items()},
                                     diff=row_out, trace=last)
    return out


def d3(seed=0, steps=500, world="d3-and", wset="d3", top=4):
    """Per-step, per-channel log loss of factor vs decompw on one stream; the state of both
    models near the worst steps (which term carries the divergence)."""
    from envs.family import WORLD_SETS, WorldEnv, Filter
    from eval.predictive import make, EPS
    w = WORLD_SETS[wset]()[world]
    env = WorldEnv(w, 1000 + seed)
    pol = np.random.default_rng(2000 + seed)
    f = make("factor-L4-P-C", w.n_channels, w.n_actions, w.value_index, seed, w)
    d = make("decompw", w.n_channels, w.n_actions, w.value_index, seed, w)
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset()
    ll = {"factor": np.zeros((steps, len(ch))), "decompw": np.zeros((steps, len(ch)))}
    for t in range(steps):
        a = int(pol.integers(w.n_actions))
        p = {"factor": np.asarray(f.predict(x, a), float),
             "decompw": np.asarray(d.predict(x, a), float)}
        y = np.array(env.step(a))
        for nm, q in p.items():
            q = np.clip(q, EPS, 1. - EPS)
            ll[nm][t] = -(y * np.log(q) + (1 - y) * np.log(1. - q))[ch]
        f.learn(x, a, y)
        d.learn(x, a, y)
        x = tuple(y)
    half = steps // 2
    df = ll["factor"][half:]
    dd = ll["decompw"][half:]
    gap = (df - dd).mean(1)                       # > 0: factor worse on that step
    order = np.argsort(-gap)[:top]
    parts = {}
    if hasattr(d, "partition"):
        parts["decompw"] = [list(map(int, g)) for g in d.partition()]
    return dict(world=world, seed=seed, steps=steps,
                loss={"factor": round(float(df.mean()), 4), "decompw": round(float(dd.mean()), 4)},
                ratio=round(float(df.mean() / dd.mean()), 4),
                worst_steps=[int(half + int(i)) for i in order],
                gap=[round(float(gap[i]), 4) for i in order],
                per_channel={int(c): [round(float(df[:, k].mean()), 4), round(float(dd[:, k].mean()), 4)]
                             for k, c in enumerate(ch)},
                factor_trace=f.trace(),
                partition=parts)


if __name__ == "__main__":
    if "d3" in sys.argv:
        print(json.dumps(d3(), indent=1)[:4000])
    else:
        print(json.dumps(t2(), indent=1)[:6000])
