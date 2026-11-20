"""Logged N_p = 200 records (C, seed 0, savings) reproduced bitwise by the current tree?"""
import json
from eval.phenomena import run_group
from harness.cset import C, logged
_, prb = logged()
out = {}
for a in C:
    m = run_group(a, "savings", "one", 0, "v1")[4]
    ref = prb[a]["savings"]["0"]["one"]
    out[a] = max(abs(float(m[k]) - float(ref[k])) for k in ref)
print(json.dumps(out))
