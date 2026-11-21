"""SPEC-3b.2 [0]: dev fidelity N_p' on discriminating rows only (fixes the Σ.2 degenerate C).
C = configs with 16-seed T2 v1 logs on all rows at N_p = 200. Row r is discriminating iff
tau_K(rank C on seeds 4..9, rank C on seeds 10..15) >= TAU_DISC (held-out seeds, disjoint from
the S1 seeds 0..3 used below -> no selection on the test seeds). tau(N) = mean over rows of
tau_K(rank C at N on S1, rank C at 200 on S1); tau* = min(.85, tau_ceil(S1) - .05)."""
import json
import os

from harness import journal
from harness.cset import kendall
from harness.staged import t2_metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = {"factor": "t2v1_factor.json", "factor-L4-P-V": "t2v1_factorF.json",
        "factor-L4-P-O": "t2v1_factorO.json"}
C = list(LOGS)
TAU_DISC = .5
S1 = range(4)


def logged():
    res = os.environ.get("PG_RESULTS", os.path.join(HERE, "results"))
    return {a: {r: v["effect"] for r, v in json.load(open(os.path.join(res, f)))["summary"][a].items()}
            for a, f in LOGS.items()}


def mean(E, a, r, S):
    return sum(E[a][r][s] for s in S) / len(S)


def disc_rows(E):
    rows = sorted(E[C[0]])
    ceil = {r: kendall([mean(E, a, r, range(4, 10)) for a in C],
                       [mean(E, a, r, range(10, 16)) for a in C]) for r in rows}
    return [r for r in rows if ceil[r] >= TAU_DISC], ceil


def analyse(jpath, tree, Ns=(25, 50, 100)):
    E = logged()
    rows, ceil_ho = disc_rows(E)
    recs = journal.load(jpath)[0]
    D = {}
    for (a, r), by in t2_metrics(recs, "v1", tree).items():
        D.setdefault(a, {})[r] = by
    ceil_s1 = sum(kendall([mean(E, a, r, [0, 1]) for a in C], [mean(E, a, r, [2, 3]) for a in C])
                  for r in rows) / len(rows)
    out = []
    for n in Ns:
        if not all(f"{a}-N{n}" in D and all(s in D[f"{a}-N{n}"].get(r, {}) for r in rows for s in S1)
                   for a in C):
            continue
        tau = sum(kendall([sum(D[f"{a}-N{n}"][r][s] for s in S1) / 4 for a in C],
                          [mean(E, a, r, S1) for a in C]) for r in rows) / len(rows)
        dy = max(abs(sum(D[f"{a}-N{n}"][r][s] for s in S1) / 4 - mean(E, a, r, S1))
                 for a in C for r in rows)
        out.append((n, round(tau, 3), round(dy, 3)))
    return dict(rows=rows, ceil_heldout={r: round(v, 2) for r, v in ceil_ho.items()},
                tau_ceil_s1=round(ceil_s1, 3), tau_star=round(min(.85, ceil_s1 - .05), 3),
                table=out)


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        import sys
        print(json.dumps(analyse(sys.argv[1], sys.argv[2])))
