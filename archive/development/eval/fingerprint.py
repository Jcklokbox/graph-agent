"""Bit-exact behavioural fingerprint of PosteriorGraph on the DEV variants.

For one episode per variant (env seed 100+s, brain seed s) it hashes every
root Q value (repr of the float, all 17 significant digits) and every chosen
action. Equal fingerprints mean that two code versions made exactly the same
computations at every decision, so a pure refactor is verified in ~90 s
instead of a full DEV run.

    python -m eval.fingerprint                                 # print fingerprints
    python -m eval.fingerprint --save eval/fingerprint_v11.txt # store a reference
    python -m eval.fingerprint --check eval/fingerprint_v11.txt
    python -m eval.fingerprint --seeds 0,1                     # more episodes per variant
    python -m eval.fingerprint --agent v11                     # agent (default v11)
"""
from __future__ import annotations

import hashlib
import sys
import time

import agents
from envs.tasks import DEV, VARIANTS
from eval.benchmark import VALUE_INDEX, pmap

AGENT = {"name": "v11"}


def fingerprint(name: str, s: int) -> tuple[str, int, str, float]:
    v = VARIANTS[name]
    env = v.make(100 + s)
    b = agents.agent_class(agents.load(AGENT["name"]))(v.n_inputs, v.n_actions,
                                                       VALUE_INDEX, seed=s)
    h = hashlib.sha256()
    q = b._q

    def traced(obs, action):
        val = q(obs, action)
        h.update(repr(val).encode())
        return val

    b._q = traced
    obs = env.reset()
    t0 = time.time()
    for _ in range(v.steps):
        a = b.step(obs)
        h.update(bytes([a]))
        obs, _, _, _ = env.step(a)
        b.learn(obs)
    return name, s, h.hexdigest()[:16], time.time() - t0


def main() -> int:
    seeds = [0]
    if "--agent" in sys.argv:
        AGENT["name"] = sys.argv[sys.argv.index("--agent") + 1]
    if "--seeds" in sys.argv:
        seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    jobs = [(nm, s) for nm in DEV for s in seeds]
    jobs.sort(key=lambda j: -VARIANTS[j[0]].steps * VARIANTS[j[0]].n_inputs ** 2)
    t0 = time.time()
    rows = sorted(pmap(fingerprint, jobs))
    lines = [f"{nm} {s} {fp}" for nm, s, fp, _ in rows]
    for (nm, s, fp, sec) in rows:
        print(f"{nm:<24}{s:>3} {fp} {sec:6.1f}s")
    print(f"wall {time.time() - t0:.1f}s")
    if "--save" in sys.argv:
        with open(sys.argv[sys.argv.index("--save") + 1], "w") as f:
            f.write("\n".join(lines) + "\n")
    if "--check" in sys.argv:
        with open(sys.argv[sys.argv.index("--check") + 1]) as f:
            ref = set(f.read().split("\n")) - {""}
        bad = [ln for ln in lines if ln not in ref]
        print("IDENTICAL" if not bad else f"MISMATCH: {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
