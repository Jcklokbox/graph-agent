"""Staged, seed-major, resumable trials (SPEC-Σ [1]).

Order Seed×Task: seed 0 on every task, then seed 1, ... Each finished task appends one record
<k=(h_tree, seed, task), m> to the journal; a resumed run skips every k already present.
Stages S0={0}, S1={0..3}, S2={0..15}. After S0 and S1 a futility gate: continue iff for every
metric z_m = (mean_m - thr_m) / (sigma_m / sqrt(n)) > c_s (sigma_m: fixed inter-seed sd,
harness/sigma.json). Early stages never accept; the final verdict uses S2 only.

    python3 -m harness.staged t2 --agents A [B] --set v1 [--rows r1,r2] --journal PATH
                              [--stages 0:0-3:0-15] [--procs 2] [--trace]
"""
import json
import math
import os
import sys
import time

from harness import journal

STAGES = ((0,), tuple(range(4)), tuple(range(16)))
C_S = (-2.0, -1.5)          # futility thresholds; harness/sim.py: beta_loss(K=10) .042, null alpha .0252
HERE = os.path.dirname(os.path.abspath(__file__))


def _t975(df):
    z = 1.959964
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def sigma_table():
    with open(os.path.join(HERE, "sigma.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------- T2 task family
def t2_tasks(agents, wset, rows):
    from envs.phenomena import SETS
    table = SETS[wset][0]
    rows = rows or list(table)
    return [f"{a}|{r}|{g}" for a in agents for r in rows for g in table[r].groups]


def t2_job(task, seed, wset, trace):
    from eval.phenomena import run_group
    a, r, g = task.split("|")
    out = run_group(a, r, g, seed, wset, trace=trace)
    return task, seed, out[4], (out[5] if trace else None)


def t2_metrics(recs, wset, h):
    """{(agent,row): {seed: effect}} from the records of tree h."""
    from envs.phenomena import SETS
    table = SETS[wset][0]
    by = {}
    for (hh, s, task), r in recs.items():
        if hh != h:
            continue
        a, row, g = task.split("|")
        by.setdefault((a, row, s), {})[g] = r["m"]
    out = {}
    for (a, row, s), groups in by.items():
        if set(groups) == set(table[row].groups):
            out.setdefault((a, row), {})[s] = float(table[row].effect(groups))
    return out


# ----------------------------------------------------------------- driver
def _pool(procs):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(procs) if procs > 1 else None


def run(kind, agents, wset, rows, jpath, stages=STAGES, procs=2, trace=False, h=None,
        log=print):
    h = h or os.environ.get("PG_TREE") or __import__("harness.snap", fromlist=["x"]).tree_hash(
        os.path.dirname(HERE))
    assert kind == "t2"
    tasks = t2_tasks(agents, wset, rows)
    sig = sigma_table()
    tpath = jpath + ".trace"
    pool = _pool(procs)
    verdict = None
    try:
        for si, seeds in enumerate(stages):
            recs, dup, bad = journal.load(jpath)
            todo = [(t, s) for s in seeds for t in tasks if (h, s, t) not in recs]
            t0 = time.monotonic()
            args = [(t, s, wset, trace and s == 0) for t, s in todo]
            it = pool.imap(_star, args, chunksize=1) if pool else map(_star, args)
            for task, s, m, tr in it:           # imap keeps seed-major order of records
                journal.append(jpath, {"k": [h, s, task], "m": m})
                if tr is not None:
                    journal.append(tpath, {"k": [h, s, task], "trace": tr})
            recs, dup, bad = journal.load(jpath)
            met = t2_metrics(recs, wset, h)
            n = len(seeds)
            rows_out, futile = [], []
            for key in sorted(met):
                v = [met[key][s] for s in seeds if s in met[key]]
                mu = sum(v) / len(v)
                sd = sig.get(key[1], sig["_default"])
                z = mu / (sd / math.sqrt(len(v)))
                rows_out.append((key, len(v), mu, z, v))
                if si < len(stages) - 1 and z <= C_S[min(si, len(C_S) - 1)]:
                    futile.append(key)
            log(f"stage S{si} seeds={seeds[0]}..{seeds[-1]} new={len(todo)} "
                f"{time.monotonic() - t0:.0f}s dup={dup} bad={bad}")
            if si < len(stages) - 1:
                if futile:
                    verdict = ("FUTILE", si, futile, rows_out)
                    break
                continue
            final = []
            for key, nn, mu, z, v in rows_out:
                sd = (sum((x - mu) ** 2 for x in v) / (nn - 1)) ** .5 if nn > 1 else float("nan")
                half = _t975(nn - 1) * sd / math.sqrt(nn) if nn > 1 else float("nan")
                ok = nn > 1 and mu - half > 0 and mu >= .05
                rev = nn > 1 and mu + half < 0
                final.append((key, nn, mu, mu - half, mu + half, "yes" if ok else
                              ("REV" if rev else "-")))
            verdict = ("FINAL", si, final, rows_out)
    finally:
        if pool:
            pool.close()
            pool.join()
    return verdict


def _star(a):
    return t2_job(*a)


def report(verdict, out=print):
    kind, si = verdict[0], verdict[1]
    if kind == "FUTILE":
        out(f"FUTILE at S{si} (c={C_S[si]}): " + ", ".join(f"{a}:{r}" for a, r in verdict[2]))
        for key, n, mu, z, v in verdict[3]:
            out(f"  {key[0]:16s} {key[1]:24s} n={n} mean={mu:+.3f} z={z:+.2f}")
        return
    yes = {}
    for key, n, mu, lo, hi, vd in verdict[2]:
        yes[key[0]] = yes.get(key[0], 0) + (vd == "yes")
        out(f"  {key[0]:16s} {key[1]:24s} n={n} {mu:+.3f} [{lo:+.3f},{hi:+.3f}] {vd}")
    out("FINAL " + " ".join(f"{a}={k}/{sum(1 for x in verdict[2] if x[0][0] == a)}"
                            for a, k in yes.items()))


def main(argv):
    def opt(name, default=None):
        if name not in argv:                    # the last occurrence wins
            return default
        return argv[len(argv) - argv[::-1].index(name)]
    agents = []
    if "--agents" in argv:
        i = argv.index("--agents") + 1
        while i < len(argv) and not argv[i].startswith("--"):
            agents.append(argv[i])
            i += 1
    stages = STAGES
    if opt("--stages"):
        stages = []
        for part in opt("--stages").split(":"):
            a, _, b = part.partition("-")
            stages.append(tuple(range(int(a), int(b or a) + 1)))
        stages = tuple(stages)
    rows = opt("--rows").split(",") if opt("--rows") else None
    v = run(argv[0], agents, opt("--set", "v1"), rows, opt("--journal"), stages,
            int(opt("--procs", "2")), "--trace" in argv, log=lambda s: print(s, flush=True))
    report(v, out=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    sys.exit(main(sys.argv[1:]))
