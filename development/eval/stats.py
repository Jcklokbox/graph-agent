"""Statistics for per-seed benchmark results (written by eval.benchmark --out).

The unit of replication is the seed: one seed runs every variant, and the
15-variant mean of that seed is one observation. Hence

* mean score and its CI: over per-seed means (t interval);
* a comparison of two configurations: paired per-seed differences on the
  common seeds (and the unpaired interval for reference);
* power: the minimal detectable effect (two-sided alpha = .05, power .8) for
  n seeds, and the number of seeds for an effect delta.

    python -m eval.stats results/a.json                    # summary + power table
    python -m eval.stats results/a.json results/b.json     # b - a, paired
"""
from __future__ import annotations

import json
import sys
from math import sqrt

import numpy as np

Z975, Z80 = 1.959964, 0.841621
BASE = ("aut", "canc", "dist")


def t975(df: int) -> float:
    """97.5 % quantile of Student t (Cornish-Fisher expansion; < 1 % error for df >= 3)."""
    z = Z975
    if df <= 0:
        return float("inf")
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def load(path: str):
    with open(path) as f:
        d = json.load(f)
    table = {nm: dict(zip(r["seeds"], r["per_seed"])) for nm, r in d["variants"].items()}
    return d, table


def seed_means(table, names=None, seeds=None):
    names = names or sorted(table)
    seeds = seeds or sorted(set.intersection(*(set(table[n]) for n in names)))
    M = np.array([[table[n][s] for n in names] for s in seeds], dtype=float)
    return seeds, names, np.nanmean(M, axis=1), M


def interval(x):
    x = np.asarray(x, float)
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, float("nan"), (float("nan"), float("nan"))
    sd = float(np.std(x, ddof=1))
    h = t975(n - 1) * sd / sqrt(n)
    return m, sd, (m - h, m + h)


def mde(sd: float, n: int, paired: bool) -> float:
    """Minimal detectable effect for n seeds per arm (sd: per-seed sd of the
    mean score, or of the paired difference when paired)."""
    k = (Z975 + Z80) * sd
    return k / sqrt(n) if paired else k * sqrt(2. / n)


def n_needed(sd: float, delta: float, paired: bool) -> int:
    k = ((Z975 + Z80) * sd / delta) ** 2
    return int(np.ceil(k if paired else 2 * k))


def summary(path: str) -> dict:
    d, table = load(path)
    seeds, names, m, M = seed_means(table)
    mean, sd, ci = interval(m)
    out = dict(agent=d.get("agent"), split=d.get("split"), n_seeds=len(seeds), mean=mean,
               sd_seed_mean=sd, ci=ci,
               per_variant={n: (float(np.nanmean(M[:, i])), float(np.nanstd(M[:, i], ddof=1)))
                            for i, n in enumerate(names)})
    var_only = [i for i, n in enumerate(names) if n not in BASE]
    if var_only and len(var_only) < len(names):
        out["variants_only"] = interval(np.nanmean(M[:, var_only], axis=1))
    return out


def compare(path_a: str, path_b: str) -> dict:
    _, ta = load(path_a)
    _, tb = load(path_b)
    names = sorted(set(ta) & set(tb))
    seeds = sorted(set.intersection(*(set(ta[n]) & set(tb[n]) for n in names)))
    _, _, ma, Ma = seed_means(ta, names, seeds)
    _, _, mb, Mb = seed_means(tb, names, seeds)
    d = mb - ma
    paired = interval(d)
    ea, eb = interval(ma), interval(mb)
    se_u = sqrt(ea[1] ** 2 / len(seeds) + eb[1] ** 2 / len(seeds))
    diff = float(np.mean(d))
    h = t975(2 * len(seeds) - 2) * se_u
    per_var = {}
    for i, n in enumerate(names):
        dv = Mb[:, i] - Ma[:, i]
        per_var[n] = interval(dv)
    return dict(names=names, n_seeds=len(seeds), diff=diff, paired=paired,
                unpaired_ci=(diff - h, diff + h), corr=float(np.corrcoef(ma, mb)[0, 1]),
                per_variant=per_var, sd_a=ea[1], sd_b=eb[1])


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 1
    for p in argv:
        s = summary(p)
        print(f"== {p}: agent {s['agent']}, split {s['split']}, {s['n_seeds']} seeds")
        print(f"mean {s['mean']:.3f}  95% CI {_fmt_ci(s['ci'])}  per-seed sd {s['sd_seed_mean']:.3f}")
        if "variants_only" in s:
            v = s["variants_only"]
            print(f"variants only {v[0]:.3f}  95% CI {_fmt_ci(v[2])}")
        print(f"{'variant':<26}{'mean':>7}{'sd':>7}{'se':>7}")
        for n, (mu, sd) in s["per_variant"].items():
            print(f"{n:<26}{mu:7.3f}{sd:7.3f}{sd / sqrt(s['n_seeds']):7.3f}")
        sd = s["sd_seed_mean"]
        print("power (unpaired, alpha .05, power .8): minimal detectable effect by seeds")
        print("   " + "  ".join(f"n={n}: {mde(sd, n, False):.3f}" for n in (8, 16, 32, 64)))
        print("   seeds needed: " + "  ".join(f"d={d}: {n_needed(sd, d, False)}"
                                              for d in (.1, .05, .03, .02)))
    if len(argv) == 2:
        c = compare(*argv)
        m, sd, ci = c["paired"]
        print(f"== {argv[1]} - {argv[0]} on {c['n_seeds']} common seeds")
        print(f"diff {c['diff']:+.3f}  paired 95% CI {_fmt_ci(ci)} (sd of diff {sd:.3f})  "
              f"unpaired CI {_fmt_ci(c['unpaired_ci'])}  corr of seed means {c['corr']:+.2f}")
        print("paired power: minimal detectable effect " + "  ".join(
            f"n={n}: {mde(sd, n, True):.3f}" for n in (8, 16, 32, 64)))
        print(f"{'variant':<26}{'diff':>7}{'95% CI':>18}")
        for n, (mu, _, vci) in c["per_variant"].items():
            print(f"{n:<26}{mu:+7.3f}  {_fmt_ci(vci)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
