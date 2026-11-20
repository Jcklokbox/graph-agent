"""SPEC-Σ.2 [3.2]: candidate set C (6 factor configurations with 16-seed T2 v1 logs, N_p = 200)."""
import json
import os
from itertools import combinations

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = {"factor": "t2v1_factor.json", "factor-V": "t2v1_factor_abl.json",
        "factor-L4": "t2v1_factor_abl.json", "factor-P": "t2v1_factor_abl.json",
        "factor-L4-P-V": "t2v1_factorF.json", "factor-L4-P-O": "t2v1_factorO.json"}
C = list(LOGS)
ROWS = ["cond-inhibition", "savings", "spont-recovery-sessions"]   # rows logged for all of C


def logged():
    """{cand: {row: [effect per seed 0..15]}} and probes {cand: {row: {seed: groups}}}."""
    eff, prb = {}, {}
    for a, f in LOGS.items():
        d = json.load(open(os.path.join(os.environ.get("PG_RESULTS", os.path.join(HERE, "results")), f)))
        eff[a] = {r: v["effect"] for r, v in d["summary"][a].items()}
        prb[a] = d["probes"][a]
    return eff, prb


def kendall(x, y):
    c = d = 0
    for i, j in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / max(1, c + d)


def score(eff, rows, seeds):
    """candidate score = mean over rows and seeds of the effect (higher = better)."""
    return [sum(eff[a][r][s] for r in rows for s in seeds) / (len(rows) * len(seeds)) for a in C]


def dev_effects(paths):
    """{cand: {row: {seed: effect}}} from staged journals of factor...-N<k> runs."""
    from harness import journal
    from harness.staged import t2_metrics
    out = {}
    for p in paths:
        recs = journal.load(p)[0]
        for h in {k[0] for k in recs}:
            for (a, row), by in t2_metrics(recs, "v1", h).items():
                out.setdefault(a, {}).setdefault(row, {}).update(by)
    return out


def table(np_paths, seeds=range(4)):
    """rows (N_p, tau_K(rank C at N_p, rank C at 200), max per-row |dy|) on seeds S1."""
    eff, _ = logged()
    ref = score(eff, ROWS, list(seeds))
    ceil = kendall(score(eff, ROWS, [0, 1]), score(eff, ROWS, [2, 3]))
    res = []
    for n, paths in sorted(np_paths.items()):
        D = dev_effects(paths)
        sc = [sum(D[f"{a}-N{n}"][r][s] for r in ROWS for s in seeds) / (len(ROWS) * len(seeds))
              for a in C]
        dy = max(abs(sum(D[f"{a}-N{n}"][r][s] for s in seeds) / len(seeds)
                     - sum(eff[a][r][s] for s in seeds) / len(seeds)) for a in C for r in ROWS)
        res.append((n, kendall(sc, ref), dy))
    return ceil, res
