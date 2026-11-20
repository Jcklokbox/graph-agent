"""Diagnosis of behavioural deficits: model (prediction) vs control (choice).

In the blind-gate tasks (aut*) the reward of a step is settled on arrival, so
the action that maximises the TRUE immediate reward is optimal (it is what the
oracle does). For every step of the scoring tail we therefore know the correct
action set from simulation (a deep copy of the environment stepped with each
action) and can ask, at the moment v11 acts:

    model right   argmax_a of v11's predicted immediate value  is a correct action
    choice right  the action v11 took                          is a correct action

and split the errors into
    control errors  model right, choice wrong  (optimism, continuation, ties)
    model errors    model wrong                (whatever the choice)

"Food steps" are the steps where the only correct action moves onto the food
side: they carry the score. Run from development/:

    python -m eval.diagnose aut-fooddelay3-noise1 aut-fooddelay2 aut --seeds 16
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np

import agents
from envs.tasks import SPLITS, VARIANTS
from eval.benchmark import pmap, _window

VALUE = 11


def episode(name, env_seed, brain_seed, agent="v11"):
    v = VARIANTS[name]
    env = v.make(env_seed)
    b = agents.agent_class(agents.load(agent))(v.n_inputs, v.n_actions, VALUE, seed=brain_seed)
    obs = env.reset()
    lo, hi = _window(name, v.steps)
    rows = []
    for t in range(v.steps):
        tail = lo <= t < hi
        if tail:
            true_r = []
            for a in range(v.n_actions):
                e2 = copy.deepcopy(env)
                true_r.append(float(e2.step(a)[1]))
        act = b.step(obs)
        if tail:
            x = b._pending[0]
            b._invalidate()
            pred = [b._predict(x, a, b.value_index, 0)[0] for a in range(v.n_actions)]
            q = [b._q(x, a) for a in range(v.n_actions)]
            cont = []
            for a in range(v.n_actions):
                nxt = b._predict_next(x, a, 0)
                vi = b.value_index
                nxt = nxt[:vi] + (min(nxt[vi], .5),) + nxt[vi + 1:]
                cont.append(b.gamma * b._expected_continuation(nxt))
            b._invalidate()
            best = max(true_r)
            correct = [a for a in range(v.n_actions) if true_r[a] >= best - 1e-9]
            rows.append(dict(t=t, act=act, correct=correct, true=true_r, pred=pred, q=q, cont=cont,
                             food=best >= .99 and len(correct) == 1 and correct[0] != 1))
        obs, r, _, _ = env.step(act)
        b.learn(obs)
    return rows


def summarise(rows):
    n = len(rows)
    model_ok = [int(np.argmax(r["pred"])) in r["correct"] for r in rows]
    choice_ok = [r["act"] in r["correct"] for r in rows]
    food = [i for i, r in enumerate(rows) if r["food"]]
    nonfood = [i for i, r in enumerate(rows) if not r["food"]]

    def split(idx):
        if not idx:
            return dict(n=0)
        m = np.array([model_ok[i] for i in idx])
        c = np.array([choice_ok[i] for i in idx])
        return dict(n=len(idx), choice=float(c.mean()), model=float(m.mean()),
                    control_err=float((m & ~c).mean()), model_err=float((~m).mean()),
                    lucky=float((~m & c).mean()))
    # reward lost against the best action, by cause; control errors further split
    # into "bonus" (the optimism term decides) and "continuation" (the look-ahead decides)
    loss = dict(food_model=0., food_control=0., other_model=0., other_control=0.,
                control_by_bonus=0., control_by_continuation=0.)
    for i, r in enumerate(rows):
        lost = max(r["true"]) - r["true"][r["act"]]
        if lost <= 1e-9:
            continue
        where = "food" if r["food"] else "other"
        kind = "model" if not model_ok[i] else "control"
        loss[f"{where}_{kind}"] += lost / n
        if kind == "control":
            mb, a = int(np.argmax(r["pred"])), r["act"]
            bonus = [r["q"][k] - r["pred"][k] - r["cont"][k] for k in range(len(r["q"]))]
            d_bonus = bonus[a] - bonus[mb]
            d_cont = r["cont"][a] - r["cont"][mb]
            loss["control_by_bonus" if d_bonus >= d_cont else "control_by_continuation"] += lost / n
    return dict(all=split(list(range(n))), food=split(food), other=split(nonfood), loss=loss)


def main(argv):
    names = [a for a in argv if a in VARIANTS] or ["aut-fooddelay3-noise1"]
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 16
    agent = argv[argv.index("--agent") + 1] if "--agent" in argv else "v11"
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    _, _, eb, bb = SPLITS["dev"]
    jobs = [(nm, eb + s, bb + s, agent) for nm in names for s in range(n_seeds)]
    res = pmap(episode, jobs)
    report = {}
    print(f"agent {agent}, {n_seeds} seeds, scoring tail; fractions of steps")
    print(f"{'variant':<24}{'steps':<7}{'n':>6}{'choice ok':>10}{'model ok':>10}"
          f"{'ctrl err':>10}{'model err':>10}")
    for nm in names:
        per = [summarise(r) for (j, r) in zip(jobs, res) if j[0] == nm]
        report[nm] = per
        for part in ("food", "other", "all"):
            vals = [p[part] for p in per if p[part]["n"] > 0]
            n = sum(v["n"] for v in vals)
            row = f"{nm if part == 'food' else '':<24}{part:<7}{n:>6}"
            for k in ("choice", "model", "control_err", "model_err"):
                xs = [v[k] for v in vals]
                row += f"{np.mean(xs):10.3f}"
            print(row)
    print("\nreward lost per tail step against the best action, by cause")
    keys = ["food_model", "food_control", "other_model", "other_control",
            "control_by_bonus", "control_by_continuation"]
    print(f"{'variant':<24}" + "".join(f"{k:>16}" for k in keys))
    for nm in names:
        print(f"{nm:<24}" + "".join(f"{np.mean([p['loss'][k] for p in report[nm]]):16.4f}"
                                    for k in keys))
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
