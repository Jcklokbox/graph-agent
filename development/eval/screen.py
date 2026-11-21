"""Sequential paired screen on DEV: stop as soon as the answer is known.

A candidate is compared with the cached per-seed results of the reference
(default eval/data/dev16_v11.json) on the same environment/brain seeds, so the
reference is never re-run. Episodes are scheduled so that every variant gets
the same wall-clock time (cheap variants accumulate seeds first), and after
every finished episode the stopping rules are checked:

  REGRESSION  a variant's paired 99% interval lies entirely below -delta
              (n >= 4): the candidate breaks something; fix before more runs
  WORSE       the DEV-mean difference (stratified over variants) has its 99%
              interval below 0
  BETTER      ... above 0  (the result is the per-seed file; a full run adds
              nothing the stopping rule did not already establish)
  NO-GAIN     the 99% interval lies inside (-delta, +delta): no effect worth
              the slot, whatever its sign
  BUDGET      the declared wall-clock budget is spent (estimates so far are
              reported; the run is cut, not extended)

99% intervals compensate (conservatively, for <= ~20 looks) for checking after
every episode. Progress is printed every `--every` seconds.

    python -m eval.screen --agent v12 --budget-min 30 --out results/scr_v12.json
    python -m eval.screen --agent v11 --kw '{"explore": 0}' aut dist dist-noise2
    python -m eval.screen --replay results/dev16_v12.json    # rules on a finished run
    python -m eval.screen --agent v12 --resume results/scr_v12.json ...  # keep episodes

Always read the printed plan first: it is the expected time to resolve +-delta
on the DEV mean; if it exceeds the budget, change the question, not the budget.
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

from envs.tasks import SPLITS, VARIANTS
from eval import benchmark

Z99 = 2.5758
DEFAULT_BASE = Path(__file__).with_name("data") / "dev16_v11.json"


def t995(df: int) -> float:
    """Student-t 0.995 quantile (Cornish-Fisher in 1/df; < 1% error for df >= 3)."""
    z = Z99
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def norm_score(r):
    d = r["oracle"] - r["stay"]
    return (r["brain"] - r["stay"]) / d if d > 1e-9 else float("nan")


class Screen:
    def __init__(self, base, names, delta, max_seeds, stop_on_regression=True):
        self.base = {nm: base["variants"][nm] for nm in names}
        self.names = names
        self.delta = delta
        self.max_seeds = min(max_seeds, *(len(b["per_seed"]) for b in self.base.values()))
        self.stop_on_regression = stop_on_regression
        self.cost = {nm: float(np.mean([r["sec"] for r in self.base[nm]["raw"]])) for nm in names}
        self.sd_prior = {nm: sqrt(2.) * float(np.std(self.base[nm]["per_seed"], ddof=1)) + 1e-3
                         for nm in names}
        self.scheduled = {nm: 0 for nm in names}
        self.diffs = {nm: {} for nm in names}
        self.raw = {nm: {} for nm in names}

    def _sd(self, nm):
        d = list(self.diffs[nm].values())
        if len(d) < 3:
            return self.sd_prior[nm]
        k = len(d) - 1                       # shrink the sample sd toward the prior (k : 2)
        return sqrt((k * float(np.var(d, ddof=1)) + 2 * self.sd_prior[nm] ** 2) / (k + 2)) + 1e-3

    def next_job(self):
        """Coverage first (3 seeds of every variant, cheapest first), then the episode that
        removes the most variance of the DEV-mean difference per second (Neyman)."""
        open_ = [nm for nm in self.names if self.scheduled[nm] < self.max_seeds]
        if not open_:
            return None
        thin = [nm for nm in open_ if self.scheduled[nm] < 3]
        if thin:
            nm = min(thin, key=lambda v: (self.scheduled[v], self.cost[v]))
        else:
            nm = max(open_, key=lambda v: self._sd(v) ** 2 /
                     (self.scheduled[v] * (self.scheduled[v] + 1) * self.cost[v]))
        s = self.scheduled[nm]
        self.scheduled[nm] += 1
        return nm, s

    def plan(self, procs, speed=1.):
        """Minutes until the DEV-mean 99% half-width reaches delta (Neyman allocation)."""
        S = sum(self.sd_prior[nm] * sqrt(self.cost[nm] * speed) for nm in self.names)
        return (Z99 * S / (len(self.names) * self.delta)) ** 2 / procs / 60

    def resume(self, path):
        old = json.load(open(path))["variants"]
        for nm, row in old.items():
            if nm in self.diffs:
                for s, r in zip(row["seeds"], row["raw"]):
                    self.add(nm, s, r)
                self.scheduled[nm] = max(self.scheduled[nm], max(row["seeds"]) + 1)

    def add(self, nm, s, raw):
        self.raw[nm][s] = raw
        self.diffs[nm][s] = norm_score(raw) - self.base[nm]["per_seed"][s]

    def variant(self, nm):
        d = list(self.diffs[nm].values())
        n = len(d)
        if n == 0:
            return 0, float("nan"), self.sd_prior[nm], (float("nan"), float("nan"))
        m = float(np.mean(d))
        sd = float(np.std(d, ddof=1)) if n >= 3 else self.sd_prior[nm]
        h = (t995(n - 1) if n >= 4 else 4.) * sd / sqrt(n)
        return n, m, sd, (m - h, m + h)

    def dev_mean(self):
        st = [self.variant(nm) for nm in self.names]
        if any(n < 2 for n, *_ in st):
            return None
        m = float(np.mean([s[1] for s in st]))
        se = sqrt(sum(s[2] ** 2 / s[0] for s in st)) / len(st)
        h = Z99 * se
        return m, (m - h, m + h)

    def decide(self):
        if self.stop_on_regression:
            for nm in self.names:
                n, m, _, ci = self.variant(nm)
                if n >= 4 and ci[1] < -self.delta:
                    return "REGRESSION", f"{nm}: {m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] (n={n})"
        dm = self.dev_mean()
        if dm is not None and all(len(self.diffs[nm]) >= 3 for nm in self.names):
            m, (lo, hi) = dm
            txt = f"DEV-mean diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if hi < 0:
                return "WORSE", txt
            if lo > 0:
                return "BETTER", txt
            if lo > -self.delta and hi < self.delta:
                return "NO-GAIN", txt
        if all(self.scheduled[nm] >= self.max_seeds for nm in self.names) and \
                sum(map(len, self.diffs.values())) == len(self.names) * self.max_seeds:
            return "EXHAUSTED", ""
        return None, ""

    def report(self, elapsed, verdict=None, why=""):
        lines = [f"[{elapsed / 60:5.1f} min] episodes {sum(map(len, self.diffs.values()))}"]
        dm = self.dev_mean()
        if dm is not None:
            lines[0] += f"  DEV-mean diff {dm[0]:+.3f} [{dm[1][0]:+.3f}, {dm[1][1]:+.3f}] (99%)"
        if verdict:
            lines.append(f"STOP {verdict}: {why}")
            lines.append(f"{'variant':<26}{'n':>3}{'diff':>8}   99% interval")
            for nm in sorted(self.names, key=lambda v: self.variant(v)[1]
                             if self.variant(v)[0] else 9):
                n, m, _, ci = self.variant(nm)
                if n:
                    lines.append(f"{nm:<26}{n:>3}{m:+8.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        return "\n".join(lines)

    def dump(self, path, meta):
        rows = {}
        for nm in self.names:
            seeds = sorted(self.raw[nm])
            if seeds:
                ps = [norm_score(self.raw[nm][s]) for s in seeds]
                rows[nm] = dict(seeds=seeds, per_seed=ps, raw=[self.raw[nm][s] for s in seeds],
                                score=float(np.mean(ps)))
        with open(path, "w") as f:
            json.dump(dict(meta, variants=rows), f, indent=1)


def replay(path, scr, procs):
    """Re-run the stopping rules on a finished run with its recorded episode times."""
    cand = json.loads(Path(path).read_text())["variants"]
    free = [0.] * procs
    pending = []                      # (finish time, nm, s)
    while True:
        while len(pending) < procs:
            job = scr.next_job()
            if job is None:
                break
            nm, s = job
            k = int(np.argmin(free))
            t = free[k] + cand[nm]["raw"][s]["sec"]
            free[k] = t
            pending.append((t, nm, s))
        if not pending:
            return scr.decide() + (max(free),)
        pending.sort()
        t, nm, s = pending.pop(0)
        scr.add(nm, s, cand[nm]["raw"][s])
        verdict, why = scr.decide()
        if verdict:
            return verdict, why, t


def live(scr, procs, budget_s, every, out, meta):
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    _, _, eb, bb = SPLITS["dev"]
    t0, last = time.time(), time.time()
    ex = ProcessPoolExecutor(procs, mp_context=mp.get_context("fork"))
    running = {}
    verdict, why = None, ""
    try:
        while True:
            while len(running) < procs:
                job = scr.next_job()
                if job is None:
                    break
                nm, s = job
                running[ex.submit(benchmark.brain_run, nm, eb + s, bb + s)] = (nm, s)
            if not running:
                verdict, why = scr.decide()
                verdict = verdict or "EXHAUSTED"
                break
            done, _ = wait(running, timeout=max(1., every), return_when=FIRST_COMPLETED)
            for f in done:
                nm, s = running.pop(f)
                scr.add(nm, s, f.result())
            verdict, why = scr.decide()
            if verdict is None and time.time() - t0 > budget_s:
                verdict, why = "BUDGET", f"{budget_s / 60:.0f} min spent"
            if verdict:
                break
            if time.time() - last >= every:
                print(scr.report(time.time() - t0), flush=True)
                last = time.time()
    finally:
        for p in list(getattr(ex, "_processes", {}).values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
    print(scr.report(time.time() - t0, verdict, why), flush=True)
    if out:
        scr.dump(out, dict(meta, verdict=verdict, why=why, minutes=(time.time() - t0) / 60))
        print("per-seed results ->", out)
    return verdict


def main(argv):
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    base_path = Path(opt("--base", DEFAULT_BASE))
    base = json.loads(base_path.read_text())
    names = [a for a in argv if a in VARIANTS] or [n for n in SPLITS["dev"][0]]
    procs = int(os.environ.get("PG_PROCS", os.cpu_count() or 1))
    scr = Screen(base, names, float(opt("--delta", .045)), int(opt("--max-seeds", 16)),
                 stop_on_regression="--no-regression-stop" not in argv)
    if "--replay" in argv:
        verdict, why, t = replay(opt("--replay"), scr, procs)
        print(scr.report(t, verdict, why))
        return
    if "--kw" in argv:
        benchmark.BRAIN_KW.update(json.loads(opt("--kw")))
    benchmark.AGENT["name"] = opt("--agent", "v11")
    benchmark.agent_module()
    budget = float(opt("--budget-min", 20)) * 60
    if "--resume" in argv:
        scr.resume(opt("--resume"))
    need = scr.plan(procs, float(opt("--speed", 1.)))
    print(f"plan: ~{need:.0f} min to resolve +-{scr.delta} on the DEV mean at the reference's cost x "
          f"{float(opt('--speed', 1.)):.1f} (regressions usually stop much earlier)", flush=True)
    if need > budget / 60:
        print(f"WARNING: budget {budget / 60:.0f} min < plan; expect BUDGET unless the effect is "
              f"large - raise --delta, restrict variants, or do not run", flush=True)
    print(f"screen: agent {benchmark.AGENT['name']} kw {benchmark.BRAIN_KW} vs {base_path}"
          f" | {len(names)} variants, <= {scr.max_seeds} seeds, delta {scr.delta}, budget {budget / 60:.0f} min",
          flush=True)
    live(scr, procs, budget, float(opt("--every", 120)), opt("--out"),
         dict(split="dev", agent=benchmark.AGENT["name"], settings=benchmark.BRAIN_KW,
              base=str(base_path)))


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        main(sys.argv[1:])
