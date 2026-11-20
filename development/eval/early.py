"""Calibration with few data: excess log loss over the exact filter in the
first eighth of each T1 episode (250 of 2000 steps), from a results file of
eval.predictive. Lower is better; plain Laplace counts (markov1) are the
reference a calibrated learner should not lose to.

    python -m eval.early results/t1v1_v12a.json
"""
import json
import sys

import numpy as np

from eval.stats import interval


def main(path):
    d = json.load(open(path))
    names = [n for n in d["agents"] if n not in ("marginal",)]
    print(f"{path}: excess log loss over the filter, first 1/8 of the episode (nats/ch/step)")
    print(f"{'world':<18}" + "".join(f"{n:>20}" for n in names))
    wins = {n: 0 for n in names}
    for wn, seeds in d["table"].items():
        row = f"{wn:<18}"
        vals = {}
        for n in names:
            v = [o[n]["curve"][0] - o["filter"]["curve"][0] for o in seeds.values()]
            m, _, ci = interval(v)
            vals[n] = m
            row += f"{m:8.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        if "markov1" in vals:
            for n in names:
                wins[n] += vals[n] <= vals["markov1"] + 1e-12
        print(row)
    if "markov1" in names:
        print("worlds with excess <= markov1: " + "  ".join(f"{n} {w}" for n, w in wins.items()))


if __name__ == "__main__":
    main(sys.argv[1])
