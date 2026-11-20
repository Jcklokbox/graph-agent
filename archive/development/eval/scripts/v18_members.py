"""D-33 instrument: v18 unchanged; record own / lc / mix predictions and weights in learn()."""
import sys, json, os
sys.path.insert(0, '.')
import numpy as np
import agents
from agents.v18 import Agent as V18
from agents.v12 import PosteriorGraph12
from eval.benchmark import pmap
EPS = 1e-4

class Rec(V18):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec = []
    def learn(self, next_obs, **kw):
        x, action = self._pending
        own = np.asarray(PosteriorGraph12._predict_next(self, x, action, 0))
        pl = self._lc_pred(x, action)
        w = self._weights()[1].copy()
        y = np.asarray([float(v) for v in next_obs]) > .5
        self.rec.append((own.copy(), pl.copy(), w, y.copy()))
        return super().learn(next_obs, **kw)

def summarize(rec, ch):
    h = len(rec) // 2
    own = np.array([r[0] for r in rec[h:]])[:, ch]; pl = np.array([r[1] for r in rec[h:]])[:, ch]
    w = np.array([r[2] for r in rec[h:]])[:, ch]; y = np.array([r[3] for r in rec[h:]])[:, ch].astype(float)
    mix = (1 - w) * own + w * pl
    def ll(q):
        q = np.clip(q, EPS, 1 - EPS)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    L = {k: ll(q) for k, q in (("own", own), ("lc", pl), ("mix", mix))}
    return dict(ll={k: float(v.mean()) for k, v in L.items()},
                ll_ch={k: [float(c) for c in v.mean(0)] for k, v in L.items()},
                w=float(w.mean()), w_ch=[float(c) for c in w.mean(0)])

def t1_job(world, seed, steps=2000):
    from envs.family import Filter, WorldEnv, WORLD_SETS
    w = WORLD_SETS["v1"]()[world]
    env, filt = WorldEnv(w, 1000 + seed), Filter(w)
    pol = np.random.default_rng(2000 + seed)
    b = Rec(w.n_channels, w.n_actions, w.value_index, seed=seed)
    ch = [i for i in range(w.n_channels) if i != w.value_index]
    x = env.reset(); filt.observe(x)
    llf, llm = [], []
    cnt1 = np.ones(w.n_channels); cnt = 2 * np.ones(w.n_channels)
    for t in range(steps):
        a = int(pol.integers(w.n_actions))
        pf = np.clip(filt.predict(a), EPS, 1 - EPS); pm = cnt1 / cnt
        b.step(tuple(float(u) for u in x)); b._pending = (b._pending[0], a)
        y = np.array(env.step(a))
        if t >= steps // 2:
            llf.append(-np.mean((y * np.log(pf) + (1 - y) * np.log(1 - pf))[ch]))
            llm.append(-np.mean((y * np.log(pm) + (1 - y) * np.log(1 - pm))[ch]))
        cnt1 += y > .5; cnt += 1
        filt.update(a, y)
        b.learn(tuple(float(u) for u in y))
        x = tuple(y)
    s = summarize(b.rec, ch)
    s["ll"]["filter"], s["ll"]["marginal"] = float(np.mean(llf)), float(np.mean(llm))
    return world, seed, s

def dev_job(name):
    from envs.tasks import VARIANTS
    v = VARIANTS[name]; env = v.make(100)
    b = Rec(v.n_inputs, v.n_actions, 11, seed=0)
    obs = env.reset()
    for _ in range(v.steps):
        a = b.step(obs); obs, r, _, info = env.step(a); b.learn(obs)
    ch = [i for i in range(v.n_inputs) if i != 11]
    return name, summarize(b.rec, ch)

if __name__ == "__main__":
    what, out = sys.argv[1], sys.argv[2]
    if what == "smoke":
        print(t1_job("hmm3", 0, 60)[2]["ll"]); sys.exit()
    if what == "t1":
        from envs.family import WORLD_SETS
        jobs = [(wn, s) for wn in WORLD_SETS["v1"]() for s in (0, 1)]
        res = pmap(t1_job, jobs)
    else:
        from envs.tasks import SPLITS, VARIANTS
        names = sorted(SPLITS["dev"][0], key=lambda n: -VARIANTS[n].steps * VARIANTS[n].n_inputs ** 2)
        res = pmap(dev_job, [(n,) for n in names])
    json.dump(res, open(out, "w"))
    print("done", out)
