"""Local predictive hypergraph learner for anonymous input and motor channels.

Transition expectations, relative associations, and eligibility traces live on
hyperedges. Environment channel identities are supplied as topology, not
interpreted as sensor semantics by the learning rules.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np

from ops import A, D, M, N, update_Q

Antecedents = tuple[int, ...]
EdgeKey = tuple[Antecedents, int]


class BlindGateGraphBrain:
    """Learn local predictions and choose among internally generated actions.

    `action_nodes` reserves graph vertices for candidate motor events. Every
    other observation slot is treated uniformly as an input, including the
    wellbeing channel. The task declares which input vertex represents
    wellbeing so candidate actions can be ranked by its predicted activation.

    The defaults match `BlindGateEnv`; alternate channel/action wiring can be
    supplied without changing the learner's association rules.
    """

    def __init__(self, *, observation_size: int = 12,
                 action_nodes: Iterable[int] = (8, 9, 10),
                 wellbeing_node: int = 11, expectation_rate: float = 0.1,
                 association_rate: float = 0.1,
                 extinction_rate: float = 0.01, trace_decay: float = 0.8) -> None:
        self.action_nodes = tuple(int(node) for node in action_nodes)
        self.observation_size = int(observation_size)
        self.wellbeing_node = int(wellbeing_node)
        if self.observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if not self.action_nodes or len(set(self.action_nodes)) != len(self.action_nodes):
            raise ValueError("action_nodes must contain distinct graph vertices")
        if any(node < 0 or node >= self.observation_size for node in self.action_nodes):
            raise ValueError("action nodes must lie within the graph observation width")
        if self.wellbeing_node < 0 or self.wellbeing_node >= self.observation_size:
            raise ValueError("wellbeing_node must lie within the graph observation width")
        if self.wellbeing_node in self.action_nodes:
            raise ValueError("wellbeing_node cannot also be an action node")
        for name, rate in (("expectation_rate", expectation_rate),
                           ("association_rate", association_rate),
                           ("extinction_rate", extinction_rate)):
            if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(trace_decay) or not 0.0 <= trace_decay < 1.0:
            raise ValueError("trace_decay must be in [0, 1)")

        self.input_nodes = tuple(node for node in range(self.observation_size)
                                 if node not in self.action_nodes)
        self.expectation_rate = float(expectation_rate)
        self.association_rate = float(association_rate)
        self.extinction_rate = float(extinction_rate)
        self.trace_decay = float(trace_decay)
        self.hyperedges: dict[EdgeKey, dict[str, Any]] = {}
        self._pending = False
        self._pending_source = np.zeros(self.observation_size, dtype=float)
        self._pending_observation = np.zeros(self.observation_size, dtype=float)

    def _validate_observation(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=float)
        if value.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), got {value.shape}"
            )
        if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("Observation values must be finite and in [0, 1]")
        if any(value[node] != 0.0 for node in self.action_nodes):
            raise ValueError("Environment must not feed motor nodes back as observations")
        return value.copy()

    @staticmethod
    def _subsets(antecedents: Antecedents) -> Iterable[Antecedents]:
        for size in range(1, len(antecedents)):
            yield from combinations(antecedents, size)

    def _source(self, observation: np.ndarray, action: int) -> np.ndarray:
        source = np.zeros(self.observation_size, dtype=float)
        source[list(self.input_nodes)] = observation[list(self.input_nodes)]
        source[self.action_nodes[action]] = 1.0
        return source

    @staticmethod
    def _active_hyperedges(source: np.ndarray) -> list[tuple[Antecedents, float]]:
        active = [int(node) for node, value in enumerate(source) if value > 0.0]
        edges: list[tuple[Antecedents, float]] = []
        for size in range(1, len(active) + 1):
            for members in combinations(active, size):
                antecedents = tuple(members)
                activation = min(float(source[node]) for node in antecedents)
                edges.append((antecedents, activation))
        return edges

    def _reference(self, antecedents: Antecedents, target: int) -> float:
        background = self.hyperedges.get(((), target))
        values = [float(background["expectation"]) if background is not None else 0.5]
        for subset in self._subsets(antecedents):
            edge = self.hyperedges.get((subset, target))
            if edge is not None and edge["confidence"] > 0.0:
                values.append(float(edge["expectation"]))
        return max(values)

    def _new_edge(self) -> dict[str, Any]:
        return {"expectation": 0.5, "confidence": 0.0,
                "positive": 0.0, "inhibition": 0.0,
                "trace": 0.0, "age": 0, "delays": {}}

    def _update_edge(self, antecedents: Antecedents, target: int,
                     activation: float, outcome: float) -> None:
        key = (antecedents, target)
        edge = self.hyperedges.get(key)
        if edge is None:
            edge = self._new_edge()
            self.hyperedges[key] = edge
        alpha = self.expectation_rate * activation
        edge["expectation"] = float(A(
            M(N(alpha), float(edge["expectation"])), M(alpha, outcome)
        ))
        reference = self._reference(antecedents, target)
        confirmation = float(M(activation, D(float(edge["expectation"]), reference)))
        falsification = float(M(activation, D(reference, float(edge["expectation"]))))
        edge["positive"] = float(update_Q(
            float(edge["positive"]), confirmation, falsification,
            self.association_rate, self.extinction_rate,
        ))
        edge["inhibition"] = float(update_Q(
            float(edge["inhibition"]), falsification, confirmation,
            self.association_rate, self.extinction_rate,
        ))
        evidence = float(M(self.association_rate, activation))
        edge["confidence"] = float(A(
            float(edge["confidence"]), M(evidence, N(float(edge["confidence"])))
        ))

    def _update_background(self, target: int, outcome: float) -> None:
        """Update the antecedent-free background relation in the same graph."""
        key: EdgeKey = ((), target)
        edge = self.hyperedges.get(key)
        if edge is None:
            edge = self._new_edge()
            self.hyperedges[key] = edge
        alpha = self.expectation_rate
        edge["expectation"] = float(A(
            M(N(alpha), float(edge["expectation"])), M(alpha, outcome)
        ))
        edge["confidence"] = float(A(
            float(edge["confidence"]), M(self.association_rate, N(float(edge["confidence"])))
        ))

    def _predict_wellbeing(self, source: np.ndarray) -> float:
        """Combine bounded excitation and inhibition over active graph edges."""
        background = self.hyperedges.get(((), self.wellbeing_node))
        prediction = float(background["expectation"]) if background is not None else 0.5
        excitation = 0.0
        inhibition = 0.0
        for antecedents, activation in self._active_hyperedges(source):
            edge = self.hyperedges.get((antecedents, self.wellbeing_node))
            if edge is None:
                continue
            confidence = float(edge["confidence"])
            positive_support = M(activation, M(confidence, float(edge["positive"])))
            inhibitory_support = M(activation, M(confidence, float(edge["inhibition"])))
            excitation = max(excitation, float(positive_support))
            inhibition = max(inhibition, float(inhibitory_support))
        return float(D(A(prediction, excitation), inhibition))

    def _local_novelty(self, source: np.ndarray, action_node: int) -> float:
        """Strongest locally active uncertainty for one candidate action."""
        novelty = 0.0
        for antecedents, activation in self._active_hyperedges(source):
            if action_node not in antecedents:
                continue
            edge = self.hyperedges.get((antecedents, self.wellbeing_node))
            confidence = 0.0 if edge is None else float(edge["confidence"])
            novelty = max(novelty, float(M(activation, N(confidence))))
        return novelty

    def select_action(self, observation: np.ndarray) -> int:
        if self._pending:
            raise RuntimeError("Call learn() after each forward()")
        obs = self._validate_observation(observation)
        scores = []
        for action, action_node in enumerate(self.action_nodes):
            source = self._source(obs, action)
            estimate = self._predict_wellbeing(source)
            novelty = self._local_novelty(source, action_node)
            # Optimism is bounded by unresolved wellbeing and driven by local
            # edge evidence, not elapsed time or an unrelated random generator.
            scores.append(float(A(estimate, M(novelty, N(estimate)))))
        return int(np.argmax(scores))

    def forward(self, observation: np.ndarray, action: int | None = None) -> int:
        if self._pending:
            raise RuntimeError("Call learn() after each forward()")
        obs = self._validate_observation(observation)
        if action is None:
            action = self.select_action(obs)
        if not isinstance(action, (int, np.integer)) or int(action) not in range(len(self.action_nodes)):
            raise ValueError(f"action must be an index in 0..{len(self.action_nodes) - 1}, or None")
        action = int(action)
        self._pending_observation = obs
        self._pending_source = self._source(obs, action)
        self._pending = True
        return action

    def learn(self, next_observation: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("Call forward() before learn()")
        target = self._validate_observation(next_observation)
        source = self._pending_source
        active_edges = self._active_hyperedges(source)

        # Merge current activity with eligibility from each edge's own trace.
        updates: dict[EdgeKey, float] = {}
        for antecedents, activation in active_edges:
            for node in self.input_nodes:
                updates[(antecedents, node)] = activation
        for key, edge in self.hyperedges.items():
            eligibility = float(edge["trace"])
            if eligibility > 0.0:
                updates[key] = max(updates.get(key, 0.0), eligibility)

        # Simple antecedents update first; compounds compare with the best
        # lower-order explanation, so redundant conjunctions add no evidence.
        for (antecedents, target_node), activation in sorted(
                updates.items(), key=lambda item: len(item[0][0])):
            self._update_edge(antecedents, target_node, activation, float(target[target_node]))

        for node in self.input_nodes:
            self._update_background(node, float(target[node]))

        self._update_temporal(target, self._pending_observation, active_edges)
        self._pending = False

    def _update_temporal(self, target: np.ndarray, previous_observation: np.ndarray,
                         active_edges: list[tuple[Antecedents, float]]) -> None:
        for edge in self.hyperedges.values():
            if edge["trace"] > 0.0:
                edge["trace"] *= self.trace_decay
                edge["age"] += 1
        for antecedents, activation in active_edges:
            for target_node in self.input_nodes:
                edge = self.hyperedges.get((antecedents, target_node))
                if edge is not None:
                    edge["trace"] = max(float(edge["trace"]), activation)
                    # Activity on the preceding transition makes the next
                    # observation one step away.
                    if edge["age"] == 0:
                        edge["age"] = 1
        for (antecedents, target_node), edge in self.hyperedges.items():
            strength = float(edge["trace"])
            if strength <= 0.0:
                continue
            if target_node == self.wellbeing_node:
                outcome_occurred = target[target_node] != previous_observation[target_node]
            else:
                outcome_occurred = target[target_node] > 0.0 and previous_observation[target_node] <= 0.0
            if outcome_occurred:
                delay = int(edge["age"])
                learning = float(M(self.association_rate, strength))
                retention = float(N(learning))
                for previous_delay in tuple(edge["delays"]):
                    edge["delays"][previous_delay] = float(
                        M(retention, float(edge["delays"][previous_delay]))
                    )
                old_probability = float(edge["delays"].get(delay, 0.0))
                edge["delays"][delay] = float(A(M(retention, old_probability), learning))
                edge["trace"] = 0.0
                edge["age"] = 0

    def _normalize_antecedents(self, antecedents: int | Iterable[int]) -> Antecedents:
        values = ((int(antecedents),) if isinstance(antecedents, (int, np.integer))
                  else tuple(int(node) for node in antecedents))
        normalized = tuple(sorted(set(values)))
        if not normalized or any(node < 0 or node >= self.observation_size for node in normalized):
            raise ValueError(f"Antecedents must be graph nodes in 0..{self.observation_size - 1}")
        return normalized

    def predictive_strength(self, antecedents: int | Iterable[int], consequent: int) -> float:
        """Return bounded positive association strength for one graph edge."""
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        return float(edge["positive"]) if edge else 0.0

    def inhibitory_strength(self, antecedents: int | Iterable[int], consequent: int) -> float:
        """Return bounded inhibitory association strength for one graph edge."""
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        return float(edge["inhibition"]) if edge else 0.0

    def observed_delay_distribution(
            self, antecedents: int | Iterable[int], consequent: int) -> dict[int, float]:
        """Return the bounded, recency-weighted distribution of event delays."""
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        return {} if edge is None else dict(edge["delays"])

    def observed_delay_probability(
            self, antecedents: int | Iterable[int], consequent: int, delay: int) -> float:
        if delay < 1:
            raise ValueError("delay must be positive")
        return self.observed_delay_distribution(antecedents, consequent).get(delay, 0.0)


__all__ = ["BlindGateGraphBrain"]
