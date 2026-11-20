"""probe_t2.py AGENT ROW[,ROW] [SEED] [--set v1|h1|h2]  — T2 probes + births / switches / hazard, one seed."""
import sys
sys.path.insert(0, '.')
from envs.phenomena import SETS
from eval.phenomena import make
table, n, _ = SETS[sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "v1"]
sys.argv = [a for i, a in enumerate(sys.argv) if a != "--set" and (i == 0 or sys.argv[i - 1] != "--set")]
agent, rows = sys.argv[1], sys.argv[2].split(",")
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
for row in rows:
    P = table[row]
    for g in P.groups:
        s = P.build(g, seed)
        ag = make(agent, seed, n)
        by = {}
        for idx, name, x in s.probes:
            by.setdefault(idx, []).append((name, x))
        pr, ev = {}, []
        b = getattr(ag, "b", None)
        last = (0, 0)
        for t in range(len(s.obs) - 1):
            ag.step(s.obs[t]); ag.learn(s.obs[t + 1])
            if b is not None:
                cur = (getattr(b, "births", 0), getattr(b, "context_switches", 0))
                if cur != last:
                    ev.append(("B" if cur[0] != last[0] else "S") + f"@{t}")
                    last = cur
            for name, x in by.get(t + 1, []):
                pr[name] = round(ag.predict_us(x), 3)
        extra = ""
        if b is not None and hasattr(b, "_hz"):
            extra = f" h={b._h_mean():.4f}"
        print(f"{row:24s} {g:10s} {pr} {' '.join(ev[:12])}{' …' if len(ev) > 12 else ''} n_ev={len(ev)}{extra}")
