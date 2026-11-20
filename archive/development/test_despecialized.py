"""Five falsifiable behavioral experiments for the learning principles.

These tests judge learned predictions and behavior in controlled tasks. They do
not inspect source text, private tables, node-pool sizes, or compatibility APIs.
"""
from __future__ import annotations

import numpy as np

from benchmark_envs import CancellationGateEnv, DistantGateEnv, VariableDelayEnv
from blind_gate import BlindGateEnv, stay_policy, evaluate
from blind_gate_graph import BlindGateGraphBrain


def make_observation(position: int, precursor: int | None = None,
                    food: int | None = None, noise: float = 0.0,
                    wellbeing: float = 0.5) -> np.ndarray:
    observation = np.zeros(12, dtype=float)
    observation[position] = 1.0
    if precursor in (0, 1):
        observation[3 + precursor] = 1.0
    if food in (0, 1):
        observation[5 + food] = 1.0
    observation[7] = noise
    observation[11] = wellbeing
    return observation


class RelabeledEnv:
    """Apply bijective sensor/action wiring around an environment."""

    def __init__(self, env, sensor_permutation: np.ndarray, action_permutation: np.ndarray):
        self.env = env
        self.sensor_permutation = np.asarray(sensor_permutation, dtype=int)
        self.action_permutation = np.asarray(action_permutation, dtype=int)

    def reset(self) -> np.ndarray:
        return self._map_observation(self.env.reset())

    def _map_observation(self, observation: np.ndarray) -> np.ndarray:
        mapped = np.asarray(observation, dtype=float).copy()
        mapped[:8] = np.asarray(observation, dtype=float)[self.sensor_permutation]
        return mapped

    def step(self, action: int):
        environment_action = int(self.action_permutation[action])
        observation, reward, done, info = self.env.step(environment_action)
        return self._map_observation(observation), reward, done, info

    def reverse_wiring(self) -> None:
        self.action_permutation[[0, 2]] = self.action_permutation[[2, 0]]
        left = int(np.flatnonzero(self.sensor_permutation == 3)[0])
        right = int(np.flatnonzero(self.sensor_permutation == 4)[0])
        self.sensor_permutation[left], self.sensor_permutation[right] = (
            self.sensor_permutation[right], self.sensor_permutation[left]
        )


def train_step(brain: BlindGateGraphBrain, observation: np.ndarray,
               action: int, next_observation: np.ndarray) -> None:
    brain.forward(observation, action)
    brain.learn(next_observation)


def run_autonomous(env, brain: BlindGateGraphBrain, steps: int):
    observation = env.reset()
    rewards: list[float] = []
    shocks = 0
    for _ in range(steps):
        action = brain.forward(observation)
        observation, reward, _, _info = env.step(action)
        brain.learn(observation)
        rewards.append(float(reward))
        shocks += int(float(reward) == 0.0)
    return np.asarray(rewards), shocks


def test_1_informative_prediction_and_parsimonious_compounds() -> None:
    brain = BlindGateGraphBrain()
    transitions = []
    # Informative precursors predict distinct later food channels.
    for _ in range(40):
        transitions.extend((
            (make_observation(1, precursor=0), 1,
             make_observation(1, food=0, wellbeing=0.5)),
            (make_observation(1, precursor=1), 1,
             make_observation(1, food=1, wellbeing=0.5)),
            # An uninformative noise bit has the same outcome in both cases.
            (make_observation(1, noise=0.0), 1, make_observation(1, wellbeing=0.5)),
            (make_observation(1, noise=1.0), 1, make_observation(1, wellbeing=0.5)),
        ))
    for observation, action, outcome in transitions:
        train_step(brain, observation, action, outcome)

    assert brain.predictive_strength(3, 5) > 0.20
    assert brain.predictive_strength(4, 6) > 0.20
    assert brain.predictive_strength(7, 11) < 0.05
    for edge in brain.hyperedges.values():
        for name in ("expectation", "confidence", "positive", "inhibition", "trace"):
            assert 0.0 <= edge[name] <= 1.0
        assert all(0.0 <= probability <= 1.0 for probability in edge["delays"].values())

    # A compound with two individually uninformative cues earns predictive
    # value only when their conjunction distinguishes the rewarded outcome.
    conjunctive = BlindGateGraphBrain()
    cases = (
        (make_observation(1), 0.0),
        (make_observation(1, precursor=0), 0.0),
        (make_observation(1, noise=1.0), 0.0),
        (make_observation(1, precursor=0, noise=1.0), 1.0),
    )
    for _ in range(30):
        for context, wellbeing in cases:
            result = context.copy()
            result[11] = wellbeing
            train_step(conjunctive, context, 1, result)
    assert conjunctive.predictive_strength((3, 7), 11) > 0.10


def test_2_autonomous_learning_survives_sensor_and_action_permutations() -> None:
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed)
        env = RelabeledEnv(
            BlindGateEnv(seed=seed),
            rng.permutation(8),
            rng.permutation(3),
        )
        rewards, shocks = run_autonomous(env, BlindGateGraphBrain(), 1000)
        tail = rewards[-300:]
        rows.append((float(np.mean(tail)), shocks / len(rewards)))

    mean_value = float(np.mean([row[0] for row in rows]))
    maximum_shock_rate = max(row[1] for row in rows)
    # Relative to the neutral/stay baseline, the agent should learn useful
    # behavior under arbitrary consistent relabeling, not just canonical IDs.
    stay = evaluate(stay_policy, seeds=list(range(5)), steps=1000)["mean_W"]
    assert mean_value > stay + 0.04, (rows, stay)
    assert maximum_shock_rate < 0.10, rows


def test_3_conditional_inhibition_and_variable_delay_learning() -> None:
    env = CancellationGateEnv(seed=31)
    brain = BlindGateGraphBrain()
    run_autonomous(env, brain, 900)

    safe = make_observation(1, precursor=0, noise=0.0)
    cancelled = make_observation(1, precursor=0, noise=1.0)
    safe_action = brain.select_action(safe)
    cancelled_action = brain.select_action(cancelled)

    delay_env = VariableDelayEnv(seed=47)
    delay_brain = BlindGateGraphBrain()
    observation = delay_env.reset()
    for _ in range(900):
        delay_brain.forward(observation, action=1)
        observation, _, _, _ = delay_env.step(1)
        delay_brain.learn(observation)
    observed_delays = delay_brain.observed_delay_distribution(3, 5)
    assert observed_delays.get(1, 0.0) < 0.10
    assert all(observed_delays.get(delay, 0.0) > 0.10 for delay in (2, 3, 4))
    assert safe_action == 0
    assert cancelled_action == 1, (safe_action, cancelled_action)


def test_4_delayed_value_guides_a_distant_two_step_route() -> None:
    env = DistantGateEnv(seed=59)
    rewards, shocks = run_autonomous(env, BlindGateGraphBrain(), 1200)
    tail = rewards[-300:]
    assert float(np.mean(tail)) > 0.53
    assert shocks / len(rewards) < 0.10


def test_5_reversal_is_learned_without_resetting_the_agent() -> None:
    rng = np.random.default_rng(77)
    env = RelabeledEnv(
        BlindGateEnv(seed=77),
        rng.permutation(8),
        rng.permutation(3),
    )
    brain = BlindGateGraphBrain()
    observation = env.reset()
    first_phase: list[float] = []
    reversal_phase: list[float] = []
    for step in range(1200):
        if step == 600:
            env.reverse_wiring()
        action = brain.forward(observation)
        observation, reward, _, _ = env.step(action)
        brain.learn(observation)
        if 400 <= step < 600:
            first_phase.append(float(reward))
        elif 900 <= step < 1200:
            reversal_phase.append(float(reward))

    assert float(np.mean(first_phase)) > 0.53
    assert float(np.mean(reversal_phase)) > 0.53
