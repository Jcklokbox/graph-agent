"""[0] D3 localization (T2 side): deterministic trace at s=0 on the failing rows vs witnesses."""
import json
import sys
import time

sys.path.insert(0, ".")
from eval.phenomena import run_group

OUT = {}
ROWS = ["reinstatement", "spont-recovery-sessions", "cond-inhibition"]
AG = ["factor-L4-P-C-N25", "decompw", "lc-sub-add"]
for r in ROWS:
    from envs.phenomena import SETS
    for g in SETS["v1"][0][r].groups:
        for a in AG:
            t = time.monotonic()
            _, _, _, _, probes, traces = run_group(a, r, g, 0, "v1", trace=True)
            OUT[f"{r}|{g}|{a}"] = dict(probes=probes, traces=traces,
                                       dt=round(time.monotonic() - t, 1))
            print(r, g, a, json.dumps(probes), flush=True)
with open("results/trace_t2.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("DONE")
