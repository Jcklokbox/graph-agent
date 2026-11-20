"""T2 effects relative to the feasibility witness (test-audit proposal 4), from saved runs.

    python -m eval.t2_relative results/t2v1_v12_v13.json results/t2v1_v14.json

Prints, per row, every agent's mean effect, its verdict, and the effect as a fraction of the
reference witness (`lc-f-E`, results/t2v1_witness_fE.json), plus the share of seeds whose
effect has the predicted sign. No verdict changes: the frozen criterion of T2 v1 stays the
one that decides; this table is for reading the size of an effect.
"""
from __future__ import annotations

import json
import sys

import numpy as np

REF = "results/t2v1_witness_fE.json"


def load(path):
    d = json.load(open(path))
    return {a: d["summary"][a] for a in d["agents"] if a in d["summary"]}


def main(argv):
    ref = load(REF)
    ref = ref.get("lc-f-E", next(iter(ref.values())))
    runs = {}
    for p in argv:
        runs.update(load(p))
    rows = list(ref)
    print(f"{'row':<24}{'ref':>7}" + "".join(f"{a:>26}" for a in runs))
    for r in rows:
        line = f"{r:<24}{ref[r]['mean']:+7.3f}"
        for a, s in runs.items():
            if r not in s:
                line += f"{'':>26}"
                continue
            e = np.asarray(s[r]["effect"])
            rel = s[r]["mean"] / ref[r]["mean"] if abs(ref[r]["mean"]) > 1e-3 else float("nan")
            line += f"{s[r]['mean']:+7.3f} {s[r]['verdict']:>3} {rel:+6.2f}x {np.mean(e > 0):4.0%}  "
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
