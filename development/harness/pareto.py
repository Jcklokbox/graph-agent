"""SPEC-Σ [3]: accuracy-cost table for the dev mode factor-...-N<k> (particles N_p).
C = T2 v1 rows x seed 0 for factor-L4-P-C; reference N_p = 200. Prints (N_p, T_step, tau_K).
usage: python3 -m harness.pareto REF_JOURNAL DEV_JOURNAL... -- N:T_step ..."""
import sys
from itertools import combinations

from harness import journal
from harness.staged import t2_metrics


def kendall(x, y):
    c = d = 0
    for i, j in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / max(1, c + d)


def effects(paths):
    out = {}
    for p in paths:
        recs = journal.load(p)[0]
        for h in {k[0] for k in recs}:
            for (a, row), by in t2_metrics(recs, "v1", h).items():
                if 0 in by:
                    out.setdefault(a, {})[row] = by[0]
    return out


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        i = sys.argv.index("--")
        E = effects(sys.argv[1:i])
        T = dict(kv.split(":") for kv in sys.argv[i + 1:])
        ref = E["factor-L4-P-C"]
        rows = sorted(ref)
        print("| N_p | T_step s (savings, seed 0) | k = T_full/T_dev | tau_K(dev, full) over 10 rows | max |dEffect| |")
        print("|---|---|---|---|---|")
        for n in sorted(T, key=int):
            a = "factor-L4-P-C" if n == "200" else f"factor-L4-P-C-N{n}"
            x = [E[a][r] for r in rows]
            y = [ref[r] for r in rows]
            print(f"| {n} | {float(T[n]):.1f} | {float(T['200']) / float(T[n]):.2f} | "
                  f"{kendall(x, y):.3f} | {max(abs(u - v) for u, v in zip(x, y)):.3f} |")
