"""Fast regression suite for autonomous control, cancellation and delay learning."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from blind_gate import BlindGateEnv
from blind_gate_graph import BlindGateGraphBrain, make_observation
from test_blind_gate_graph import (
    chunking_transitions,
    run_random,
    run_test_1,
    train_transitions,
)


class CancellationGateEnv:
    """A deterministic gate with a 50% red cancellation lamp on each V3 cue."""

    def __init__(self, seed: int = 0, period: int = 5) -> None:
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 1
        self.wellbeing = 0.5
        self.precursor = None
        self.trap = False
        self.food = None
        self.next_cue = 3
        return self.observation()

    def observation(self):
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.0
        if self.precursor == 0:
            obs[3] = 1.0
        if self.food == 0:
            obs[5] = 1.0
        obs[7] = float(self.trap)
        obs[11] = self.wellbeing
        return obs

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(action)
        current_cue = self.precursor
        current_trap = self.trap
        current_food = self.food
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1

        if current_cue == 0:
            self.food = None if current_trap else 0
        else:
            self.food = None

        if current_cue == 0 and not current_trap and self.position == 0:
            self.wellbeing = 1.0
        elif self.position == 1:
            self.wellbeing = 0.5
        else:
            self.wellbeing = 0.0

        self.precursor = None
        self.trap = False
        if self.t >= self.next_cue:
            self.precursor = 0
            self.trap = bool(self.rng.random() < 0.5)
            self.next_cue += self.period

        return self.observation(), self.wellbeing, False, {
            "received_shock": self.wellbeing == 0.0,
            "got_food": self.wellbeing == 1.0,
        }


class VariableDelayEnv:
    """V3 pulses followed by V5 after a uniformly selected 2..4 steps."""

    def __init__(self, seed: int = 0, period: int = 8) -> None:
        self.rng = np.random.default_rng(seed)
        self.period = int(period)
        self.reset()

    def reset(self):
        self.t = 0
        self.next_cue = 2
        self.food_due: int | None = None
        return self.observation()

    def observation(self):
        obs = np.zeros(12, dtype=float)
        obs[1] = 1.0
        if self.t == self.next_cue - self.period:
            obs[3] = 1.0
        if self.food_due == self.t:
            obs[5] = 1.0
        obs[11] = 0.5
        return obs

    def step(self, action: int = 1):
        del action
        if self.t == self.next_cue - self.period:
            delay = int(self.rng.integers(2, 5))
            self.food_due = self.t + delay
        self.t += 1
        if self.t >= self.next_cue:
            self.next_cue += self.period
        return self.observation(), 0.5, False, {}


def test_1_compatibility() -> dict:
    base = run_test_1()
    random100 = run_random(19, 20, 21, 100, compact_learning=True)
    random1000 = run_random(19, 20, 21, 1000, compact_learning=True)

    brain = BlindGateGraphBrain(
        seed=11, temporal_enabled=False, consume_trace=False, compact_learning=True
    )
    train_transitions(brain, chunking_transitions(), 72)
    left = brain.chunk_by_components.get((1, 8))
    right = brain.chunk_by_components.get((1, 10))
    left_q = brain.q_pred((left, 3), 11) if left is not None else 0.0
    right_q = brain.q_pred((right, 4), 11) if right is not None else 0.0
    simple_q = brain.q_pred((1, 8), 11)

    pass_noise = (
        random100["v7_edges_Q_ge_0_5"] < 8
        and random100["v7_max_Q"] < 0.5
        and random1000["v7_edges_Q_ge_0_5"] == 0
        and random1000["v7_max_Q"] < 0.5
    )
    return {
        "base_links": base["table"],
        "random_noise": {
            "100_steps": {
                "surviving_edges": random100["surviving_pred_edges"],
                "V7_max_Q": random100["v7_max_Q"],
                "V7_edges_Q_ge_0_5": random100["v7_edges_Q_ge_0_5"],
            },
            "1000_steps": {
                "surviving_edges": random1000["surviving_pred_edges"],
                "V7_max_Q": random1000["v7_max_Q"],
                "V7_edges_Q_ge_0_5": random1000["v7_edges_Q_ge_0_5"],
            },
            "pass": pass_noise,
        },
        "chunks": brain.chunk_snapshot(),
        "chunk_links": {
            "((V1,V8),V3)->V11": left_q,
            "((V1,V10),V4)->V11": right_q,
            "(V1,V8)->V11": simple_q,
        },
        "pass": (
            base["all_pass"] and pass_noise
            and left_q > 0.99 and right_q > 0.99 and simple_q < 0.01
        ),
    }


def test_2_autonomous() -> dict:
    rows = []
    for seed in range(5):
        env = BlindGateEnv(seed=seed)
        observation = env.reset()
        brain = BlindGateGraphBrain(
            seed=100 + seed,
            temporal_enabled=False,  # S_fast stays active; only delay vessels are skipped.
            compact_learning=True,
            max_chunks=0,
        )
        rewards: list[float] = []
        for _ in range(1000):
            action = brain.forward(observation, action=None)
            next_observation, reward, _, info = env.step(action)
            brain.learn(next_observation)
            observation = next_observation
            rewards.append(float(reward))
        tail = rewards[-300:]
        rows.append({
            "seed": seed,
            "mean_W_last_300": float(np.mean(tail)),
            "shock_rate_last_300": float(np.mean(np.asarray(tail) == 0.0)),
            "pass": float(np.mean(tail)) > 0.55
                    and float(np.mean(np.asarray(tail) == 0.0)) < 0.05,
        })
    return {
        "runs": rows,
        "mean_W_across_seeds": float(np.mean([r["mean_W_last_300"] for r in rows])),
        "max_shock_rate": float(max(r["shock_rate_last_300"] for r in rows)),
        "pass": all(r["pass"] for r in rows),
    }


def test_3_cancellation() -> dict:
    env = CancellationGateEnv(seed=31)
    observation = env.reset()
    brain = BlindGateGraphBrain(
        seed=31, temporal_enabled=False, compact_learning=True
    )
    for _ in range(900):
        action = brain.forward(observation, action=None)
        next_observation, _, _, _ = env.step(action)
        brain.learn(next_observation)
        observation = next_observation

    q_plus = brain.q_pred(3, 5)
    q_minus_v7_v5 = brain.q_inhib(7, 5)
    q_minus_pair_v5 = brain.q_inhib((3, 7), 5)
    q_minus_v7_v11 = brain.q_inhib(7, 11)
    safe = make_observation(1, precursor=0, noise=0.0)
    trap = make_observation(1, precursor=0, noise=1.0)
    safe_action = brain.select_action(safe)
    trap_action = brain.select_action(trap)
    return {
        "Q_plus_V3_to_V5": q_plus,
        "Q_minus_V7_to_V5": q_minus_v7_v5,
        "Q_minus_(V3,V7)_to_V5": q_minus_pair_v5,
        "Q_minus_V7_to_V11": q_minus_v7_v11,
        "action_safe_precursor": safe_action,
        "action_cancelled_precursor": trap_action,
        "pass": q_plus > 0.85
                and max(q_minus_v7_v5, q_minus_pair_v5, q_minus_v7_v11) > 0.50
                and safe_action == 0 and trap_action == 1,
        "inhibitory_links": brain.inhibitory_snapshot(),
    }


def test_4_variable_delay() -> dict:
    env = VariableDelayEnv(seed=47)
    observation = env.reset()
    brain = BlindGateGraphBrain(
        seed=47, temporal_enabled=True, compact_learning=True
    )
    for _ in range(900):
        brain.forward(observation, action=1)
        next_observation, _, _, _ = env.step(1)
        brain.learn(next_observation)
        observation = next_observation

    temporal = brain.q_pred_temporal(3, 5)
    predictions = {delay: brain.expected_at_delay(3, 5, delay) for delay in range(1, 5)}
    return {
        "Q_temporal_V3_to_V5": temporal,
        "expected_by_delay": predictions,
        "pass": temporal["fast"] < 0.15
                and temporal["mid"] > 0.20
                and temporal["slow"] > 0.20
                and predictions[1] < 0.20
                and min(predictions[d] for d in (2, 3, 4)) > 0.15,
    }



class DistantGateEnv:
    """Two-step food route: V2 -> V1 -> V0, with a one-tick food window."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 2
        self.next_cue = 5
        self.food_time: int | None = None
        self.last_cue = -100
        return self.observation()

    def observation(self):
        obs = np.zeros(12, dtype=float)
        obs[self.position] = 1.0
        if self.t == self.last_cue:
            obs[3] = 1.0
        if self.food_time == self.t and self.position == 0:
            obs[5] = 1.0
        if self.position == 2:
            obs[11] = 0.50
        elif self.position == 1:
            obs[11] = 0.45
        else:
            obs[11] = 1.0 if self.food_time == self.t else 0.0
        return obs

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(action)
        self.position = int(np.clip(self.position + action - 1, 0, 2))
        self.t += 1
        reward = 0.50 if self.position == 2 else 0.45 if self.position == 1 else 0.0
        if self.position == 0 and self.food_time == self.t:
            reward = 1.0
        if self.t >= self.next_cue and self.position == 2:
            self.last_cue = self.t
            self.food_time = self.t + 2
            self.next_cue = self.t + int(self.rng.integers(5, 7))
        obs = self.observation()
        # Observation owns the authoritative W value.
        obs[11] = reward
        return obs, reward, False, {"got_food": reward == 1.0, "received_shock": reward == 0.0 and self.position == 0}


def test_5_distant_gate() -> dict:
    env = DistantGateEnv(seed=59)
    observation = env.reset()
    brain = BlindGateGraphBrain(
        seed=59, temporal_enabled=True, compact_learning=True
    )
    rewards: list[float] = []
    cue_actions: list[int] = []
    corridor_after_cue_actions: list[int] = []
    corridor_without_cue_actions: list[int] = []
    for _ in range(1000):
        cue_trace_before = max(brain.S_fast[3], brain.S_mid[3])
        action = brain.forward(observation, action=None)
        cue_active = observation[3] > 0.5 and observation[2] > 0.5
        if cue_active:
            cue_actions.append(action)
        elif observation[1] > 0.5 and cue_trace_before > 0.80:
            corridor_after_cue_actions.append(action)
        elif observation[1] > 0.5 and cue_trace_before <= 0.20:
            corridor_without_cue_actions.append(action)
        next_observation, reward, _, _ = env.step(action)
        brain.learn(next_observation)
        observation = next_observation
        rewards.append(float(reward))

    tail = rewards[-300:]
    # Read the temporal conjunction value directly; it is a value feature,
    # not a Pred antecedent, so the public 1/2-node Q API remains unchanged.
    key = (tuple(sorted((1, 3, 8, 20))), 11)
    stats = brain.q_val_stats.get(key, [0.0, 0.0])
    corridor_value = float(stats[0] / stats[1]) if stats[1] else 0.0
    q_plus_action0_w = brain.q_pred(8, 11)
    mean_w = float(np.mean(tail))
    shock_rate = float(np.mean([r == 0.0 for r in tail]))
    return {
        "mean_W_last_300": mean_w,
        "shock_rate_last_300": shock_rate,
        "secondary_value_V1_after_V3": corridor_value,
        "Q_plus_action0_to_V11": q_plus_action0_w,
        "cue_action_rate_left": float(np.mean(np.asarray(cue_actions) == 0)) if cue_actions else 0.0,
        "corridor_after_cue_action_rate_left": float(np.mean(np.asarray(corridor_after_cue_actions) == 0)) if corridor_after_cue_actions else 0.0,
        "corridor_without_cue_action_rate_right": float(np.mean(np.asarray(corridor_without_cue_actions) == 2)) if corridor_without_cue_actions else 0.0,
        "cue_decisions": len(cue_actions),
        "corridor_after_cue_decisions": len(corridor_after_cue_actions),
        "corridor_without_cue_decisions": len(corridor_without_cue_actions),
        "pass": corridor_value > 0.70 and q_plus_action0_w > 0.85
                and mean_w > 0.53 and shock_rate < 0.05
                and bool(cue_actions) and float(np.mean(np.asarray(cue_actions) == 0)) > 0.60
                and bool(corridor_after_cue_actions) and float(np.mean(np.asarray(corridor_after_cue_actions) == 0)) > 0.60
                and bool(corridor_without_cue_actions) and float(np.mean(np.asarray(corridor_without_cue_actions) == 2)) > 0.50,
    }

def run() -> dict:
    started = time.perf_counter()
    tests = {
        "test_1_compatibility": test_1_compatibility(),
        "test_2_autonomous": test_2_autonomous(),
        "test_3_cancellation": test_3_cancellation(),
        "test_4_variable_delay": test_4_variable_delay(),
        "test_5_distant_gate": test_5_distant_gate(),
    }
    tests["runtime_seconds"] = time.perf_counter() - started
    tests["all_pass"] = all(
        tests[name]["pass"] for name in (
            "test_1_compatibility", "test_2_autonomous",
            "test_3_cancellation", "test_4_variable_delay", "test_5_distant_gate",
        )
    )
    return tests


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("autonomous_generalization_test.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["all_pass"]:
        raise SystemExit("At least one autonomous/generalization test failed")
    if result["runtime_seconds"] >= 12.0:
        raise SystemExit("Test suite exceeded the 12-second runtime budget")
