"""Paired canonical-vs-relabelled benchmark sweep; no parameter fitting.

Run from repository root after installing development dependencies:
    python research/permutation_equivariance_sweep.py --pairs 300 --steps 1000

For each environment seed, canonical and randomly relabelled runs use the same
underlying environment seed. The learner is reset between them. CSV rows are
written incrementally so an interrupted long run preserves completed pairs.
This is a robustness measurement of the current learner, not a test of a
specific proposed mechanism.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "development"))
from blind_gate import BlindGateEnv  # noqa: E402
from blind_gate_graph import BlindGateGraphBrain  # noqa: E402


class RelabeledEnv:
    def __init__(self, env: BlindGateEnv, sensor_perm: np.ndarray,
                 action_perm: np.ndarray) -> None:
        self.env = env
        self.sensor_perm = np.asarray(sensor_perm, dtype=int)
        self.action_perm = np.asarray(action_perm, dtype=int)

    def reset(self) -> np.ndarray:
        return self._map(self.env.reset())

    def _map(self, obs: np.ndarray) -> np.ndarray:
        mapped = np.asarray(obs, dtype=float).copy()
        mapped[:8] = np.asarray(obs, dtype=float)[self.sensor_perm]
        return mapped

    def step(self, action: int):
        obs, reward, done, info = self.env.step(int(self.action_perm[action]))
        return self._map(obs), reward, done, info


def run_one(seed: int, steps: int, *, relabel: bool,
            permutation_seed: int) -> tuple[float, float]:
    base_env = BlindGateEnv(seed=seed)
    if relabel:
        rng = np.random.default_rng(permutation_seed)
        env = RelabeledEnv(base_env, rng.permutation(8), rng.permutation(3))
    else:
        env = RelabeledEnv(base_env, np.arange(8), np.arange(3))
    brain = BlindGateGraphBrain()
    obs = env.reset()
    rewards = np.empty(steps, dtype=float)
    shocks = 0
    for t in range(steps):
        action = brain.forward(obs)
        obs, reward, _, info = env.step(action)
        brain.learn(obs)
        rewards[t] = reward
        shocks += int(info.received_shock)
    tail_start = max(0, steps - min(300, steps))
    return float(rewards[tail_start:].mean()), shocks / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research" / "permutation_sweep_results.csv")
    args = parser.parse_args()
    if args.pairs < 1 or args.steps < 1:
        parser.error("pairs and steps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "seed", "steps", "canonical_tail", "canonical_shock",
            "relabel_tail", "relabel_shock", "tail_difference",
            "shock_difference"))
        if new_file:
            writer.writeheader()
            handle.flush()
        for i in range(args.pairs):
            seed = args.start_seed + i
            canonical = run_one(seed, args.steps, relabel=False,
                                permutation_seed=seed + 1_000_000)
            relabeled = run_one(seed, args.steps, relabel=True,
                                permutation_seed=seed + 1_000_000)
            writer.writerow({
                "seed": seed, "steps": args.steps,
                "canonical_tail": canonical[0], "canonical_shock": canonical[1],
                "relabel_tail": relabeled[0], "relabel_shock": relabeled[1],
                "tail_difference": relabeled[0] - canonical[0],
                "shock_difference": relabeled[1] - canonical[1],
            })
            handle.flush()
            if (i + 1) % 10 == 0 or i + 1 == args.pairs:
                print(f"completed_pairs={i + 1}/{args.pairs}", flush=True)


if __name__ == "__main__":
    main()
