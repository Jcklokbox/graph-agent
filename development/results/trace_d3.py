"""[0] D3 localization (T1 side): per-channel excess LL of factor-L4-P-C vs decompw on d3."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from envs.family import WORLD_SETS, Filter, WorldEnv
from eval.predictive import make

STEPS = 1200
EPS = 1e-4
OUT = {}
for wn in ("d3-and", "d3-and-ctrl", "d3-ctrl"):
    w = WORLD_SETS["d3"]()[wn]
    env, filt = WorldEnv(w, 1000), Filter(w)
    pol = np.random.default_rng(2000)
    preds = {nm: make(nm, w.n_channels, w.n_actions, w.value_index, 0, w)
             for nm in ("factor-L4-P-C-N100", "decompw")}
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset()
    filt.observe(x)
    ll = {nm: np.zeros((STEPS, len(ch))) for nm in list(preds) + ["filter"]}
    for t in range(STEPS):
        a = int(pol.integers(w.n_actions))
        p = {"filter": filt.predict(a)}
        for nm, pr in preds.items():
            p[nm] = pr.predict(x, a)
        y = np.array(env.step(a))
        for nm, q in p.items():
            q = np.clip(np.asarray(q, float), EPS, 1. - EPS)
            ll[nm][t] = -(y * np.log(q) + (1 - y) * np.log(1 - q))[ch]
        filt.update(a, y)
        for nm, pr in preds.items():
            pr.learn(x, a, y)
        x = tuple(y)
    half = STEPS // 2
    fm = {nm: ll[nm][half:].mean(0) for nm in preds}
    fl = ll["filter"][half:].mean(0)
    out = {}
    for nm in preds:
        gc = (np.log(.5) - fm[nm]) / (np.log(.5) - fl)
        out[nm] = dict(ll=[round(float(v), 4) for v in fm[nm]],
                       gc=[round(float(v), 4) for v in gc])
    out["filter_ll"] = [round(float(v), 4) for v in fl]
    m = preds["factor-L4-P-C-N100"]
    out["factor_trace"] = m.trace()
    out["factor_partition"] = [list(map(int, g)) for g in preds["decompw"].partition()]
    out["decompw_partition"] = [list(map(int, g)) for g in preds["decompw"].partition()]
    OUT[wn] = out
    print(wn, json.dumps(out), flush=True)
with open("results/trace_d3.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("DONE")
