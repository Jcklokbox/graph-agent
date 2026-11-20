import time, sys, json
from eval.phenomena import run_group
out = {}
for n in (25, 50, 100, 200):
    a = "factor-L4-P-C" if n == 200 else f"factor-L4-P-C-N{n}"
    t = time.process_time(); run_group(a, "savings", "one", 0, "v1"); out[n] = round(time.process_time() - t, 2)
print(json.dumps(out))
