"""Homogeneous causal-temporal hypergraph controller.

All vertices are ordinary input/output channels.  A directed hyperedge stores
its own causal strength, continuous target expectation and temporal profile;
there are no synthetic chunk vertices or separate motor/value tables.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from brain_p2 import DynamicQPool
from ops import A, D, M

Antecedents = tuple[int, ...]
EdgeKey = tuple[Antecedents, int]


class _ValueStats(dict):
    """Dictionary with a projection for retired virtual chunk identifiers."""
    def get(self, key: Any, default: Any = None) -> Any:
        value = super().get(key, None)
        if value is not None:
            return value
        if (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], tuple)):
            # Old callers may include non-vertex virtual IDs in a query.
            # Project onto the real vertices without assigning them graph state.
            real = tuple(sorted(x for x in key[0] if 0 <= x <= 10))
            if real and len(real) <= 3:
                return super().get((real, key[1]), default)
        return default


class BlindGateGraphBrain:
    ETA_PLUS = 0.10
    ETA_MINUS = 0.01
    MAX_ORDER = 3
    MAX_CHUNKS = 0  # compatibility constant; the hypergraph creates no chunks

    def __init__(self, seed: int = 0, *, consume_trace: bool = False,
                 max_chunks: int = 16, temporal_enabled: bool = True,
                 compact_learning: bool = False) -> None:
        # Legacy switches remain accepted; temporal edge dynamics are always on.
        del consume_trace, max_chunks, temporal_enabled, compact_learning
        self.rng = np.random.default_rng(seed)
        self.sensor_nodes = tuple(range(8))
        self.action_nodes = (8, 9, 10)
        self.value_node = 11
        self.hyperedges: dict[EdgeKey, dict[str, Any]] = {}
        self.edge_trace: dict[EdgeKey, np.ndarray] = {}
        self.edge_age: dict[EdgeKey, int] = {}
        self._trace_index_by_target: dict[int, set[EdgeKey]] = defaultdict(set)
        self._discharged: set[EdgeKey] = set()
        self._trace_source_strength = np.zeros(8, dtype=float)
        self.p_base = np.full(12, 0.5, dtype=float)
        self._base_ema = np.full(12, 0.5, dtype=float)
        self._base_counts = np.zeros(12, dtype=np.int64)
        self.total_steps = 0
        self.S_fast = np.zeros(12, dtype=float)  # read-only compatibility projections
        self.S_mid = np.zeros(12, dtype=float)
        self.S_slow = np.zeros(12, dtype=float)
        self.S_trace = np.zeros(12, dtype=float)
        self.pred_pool = DynamicQPool("hyperedge_positive", 50000)
        self.pred_mid_pool = DynamicQPool("hyperedge_mid", 50000)
        self.pred_slow_pool = DynamicQPool("hyperedge_slow", 50000)
        self.pred_inhib = DynamicQPool("hyperedge_negative", 50000)
        self.q_minus_pool = self.pred_inhib
        self.vessel_q: dict[tuple[int, str, int], float] = {}
        self.q_val_stats: dict[tuple[Any, int], list[float]] = _ValueStats()
        # Compatibility-only virtual handles for old nested chunk queries.
        # They are never vertices in the causal graph or in activation arrays.
        self.chunk_components: dict[int, tuple[int, int]] = {}
        self.chunk_by_components: dict[tuple[int, int], int] = {}
        self._virtual_pairs: dict[int, tuple[int, int]] = {}
        self._next_virtual = 100
        self._pending = False
        self._pending_source = np.zeros(12, dtype=float)
        self._pending_obs = np.zeros(12, dtype=float)
        self._pending_action = 1
        self.last_debug: dict[str, Any] = {}
        self.base_rate = self.p_base.copy()
        self._edge_index: dict[int, set[EdgeKey]] = defaultdict(set)
        self._value_index_by_action: dict[int, set[EdgeKey]] = defaultdict(set)
        self._prediction_index_by_target: dict[int, set[EdgeKey]] = defaultdict(set)
        self._traceable_targets: dict[Antecedents, set[int]] = defaultdict(set)

    @staticmethod
    def _validate_observation(observation: np.ndarray) -> np.ndarray:
        v = np.asarray(observation, dtype=float)
        if v.shape != (12,):
            raise ValueError(f"Expected observation shape (12,), got {v.shape}")
        if not np.all(np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0):
            raise ValueError("Observation values must be finite and in [0, 1]")
        if np.any(v[8:11] != 0.0):
            raise ValueError("Environment must not feed output neurons 8..10 back as inputs")
        return v.copy()

    def _expand_legacy_antecedents(self, antecedents: Any) -> Antecedents:
        if isinstance(antecedents, (int, np.integer)):
            raw = (int(antecedents),)
        else:
            raw_list: list[int] = []
            for item in antecedents:
                if isinstance(item, (tuple, list)):
                    raw_list.extend(int(x) for x in item)
                else:
                    value = int(item)
                    if value in self._virtual_pairs:
                        raw_list.extend(self._virtual_pairs[value])
                    else:
                        raw_list.append(value)
            raw = tuple(raw_list)
        # A legacy virtual handle may be nested with a cue; flatten and dedupe.
        raw = tuple(sorted(set(raw)))
        if not raw or len(raw) > self.MAX_ORDER:
            raise ValueError("Hyperedge antecedents must contain 1..3 distinct vertices")
        if any(x < 0 or x > 10 for x in raw):
            raise ValueError("Antecedent vertex must be in 0..10")
        return raw

    def _register_legacy_pair(self, pair: tuple[int, int]) -> int:
        if pair not in self.chunk_by_components:
            handle = self._next_virtual
            self._next_virtual += 1
            self.chunk_by_components[pair] = handle
            self.chunk_components[handle] = pair
            self._virtual_pairs[handle] = pair
        return self.chunk_by_components[pair]

    @staticmethod
    def _subsets(ants: Antecedents) -> Iterable[Antecedents]:
        for size in range(1, len(ants)):
            yield from combinations(ants, size)

    def _edge(self, key: EdgeKey) -> dict[str, Any] | None:
        return self.hyperedges.get(key)

    def _q(self, ants: Antecedents, target: int) -> float:
        edge = self.hyperedges.get((ants, target))
        return float(edge["q"]) if edge else 0.0

    def _mu(self, ants: Antecedents, target: int) -> float:
        edge = self.hyperedges.get((ants, target))
        return float(edge["mu"]) if edge else float(self.p_base[target])

    def _reference_mu(self, ants: Antecedents, target: int) -> float:
        if len(ants) == 1:
            return float(self.p_base[target])
        return max([float(self.p_base[target])] + [self._mu(sub, target) for sub in self._subsets(ants)])

    def _make_source(self, obs: np.ndarray, action: int) -> np.ndarray:
        source = np.zeros(12, dtype=float)
        source[:8] = np.maximum(obs[:8], self._trace_source_strength)
        source[self.action_nodes[action]] = 1.0
        return source

    def _active_keys(self, source: np.ndarray,
                     observed: np.ndarray | None = None) -> list[tuple[Antecedents, float]]:
        active: list[int] = []
        for node in range(11):
            if source[node] < 0.08:
                continue
            if node < 8 and observed is not None and observed[node] < 0.08:
                continue
            active.append(node)

        # A single edge-local trace can supply the missing cue in a new
        # conjunction. Additional trace-only cues remain available for
        # prediction through their own indexed edges, without multiplying
        # candidate conjunctions combinatorially.
        if observed is not None:
            strongest_node = -1
            strongest_value = 0.08
            for node in range(8):
                value = float(source[node])
                if observed[node] < 0.08 and value > strongest_value:
                    strongest_node = node
                    strongest_value = value
            if strongest_node >= 0:
                active.append(strongest_node)
        active.sort()
        values = [float(source[i]) for i in active]
        keys: list[tuple[Antecedents, float]] = []
        count = len(active)
        for i in range(count):
            a, va = active[i], values[i]
            keys.append(((a,), va))
            for j in range(i + 1, count):
                b, vb = active[j], values[j]
                strength2 = va if va < vb else vb
                keys.append(((a, b), strength2))
                for k in range(j + 1, count):
                    c, vc = active[k], values[k]
                    # Three-sensor-only conjunctions are admitted when all
                    # three cues are simultaneously observed. Otherwise the
                    # third vertex must be the action channel.
                    if c < 8 and a < 8 and b < 8:
                        if observed is None or min(observed[a], observed[b], observed[c]) < 0.08:
                            continue
                    strength3 = strength2 if strength2 < vc else vc
                    keys.append(((a, b, c), strength3))
        return keys

    def _sync_compatibility_pools(self) -> None:
        self.pred_pool.nodes.clear()
        self.pred_mid_pool.nodes.clear()
        self.pred_slow_pool.nodes.clear()
        self.pred_inhib.nodes.clear()
        self.vessel_q.clear()
        for (ants, target), edge in self.hyperedges.items():
            q = float(edge["q"])
            if q > 0.0:
                self.pred_pool.nodes[(ants, target)] = q
            self.pred_mid_pool.nodes[(ants, target)] = q * float(edge["lag"][1])
            self.pred_slow_pool.nodes[(ants, target)] = q * float(edge["lag"][2])
            if edge["q_minus"] > 0.0:
                self.pred_inhib.nodes[(ants, target)] = float(edge["q_minus"])
            if len(ants) == 1:
                self.vessel_q[(ants[0], "mid", target)] = q * float(edge["lag"][1])
                self.vessel_q[(ants[0], "slow", target)] = q * float(edge["lag"][2])

    def _project_traces(self) -> None:
        self.S_fast.fill(0.0)
        self.S_mid.fill(0.0)
        self.S_slow.fill(0.0)
        self._trace_source_strength.fill(0.0)
        for (ants, _target), trace in self.edge_trace.items():
            strength = max(float(trace[0]), float(trace[1]), float(trace[2]))
            for node in ants:
                if node < 8:
                    if trace[0] > self.S_fast[node]:
                        self.S_fast[node] = float(trace[0])
                    if trace[1] > self.S_mid[node]:
                        self.S_mid[node] = float(trace[1])
                    if trace[2] > self.S_slow[node]:
                        self.S_slow[node] = float(trace[2])
                    if len(ants) == 1 and strength >= 0.08 and strength > self._trace_source_strength[node]:
                        self._trace_source_strength[node] = strength
        self.S_trace[:8] = np.maximum.reduce((self.S_fast[:8], self.S_mid[:8], self.S_slow[:8]))

    def _predict_targets(self, source: np.ndarray) -> np.ndarray:
        predictions = np.zeros(8, dtype=float)
        for target, indexed in self._prediction_index_by_target.items():
            best = 0.0
            for key in indexed:
                ants, _ = key
                strength = 1.0
                for node in ants:
                    value = float(source[node])
                    if value < 0.08:
                        strength = 0.0
                        break
                    if value < strength:
                        strength = value
                if strength == 0.0:
                    continue
                edge = self.hyperedges[key]
                lag = edge["lag"]
                temporal = max(float(lag[0]), float(lag[1]), float(lag[2]))
                confidence = min(1.0, edge["n"] / 3.0)
                value = float(edge["mu"]) * max(float(edge["q"]), 0.15) * confidence
                value *= max(strength, temporal)
                if value > best:
                    best = value
            predictions[target] = best
        return predictions

    def _best_indexed_value(self, source: np.ndarray, action_node: int) -> tuple[float, bool]:
        best = 0.0
        matched = False
        edges = self.hyperedges
        for key in self._value_index_by_action[action_node]:
            ants, _target = key
            strength = 1.0
            valid = True
            for node in ants:
                value = float(source[node])
                if value < 0.08:
                    valid = False
                    break
                if value < strength:
                    strength = value
            if not valid:
                continue
            matched = True
            edge = edges[key]
            count = edge["n"]
            if count < 1:
                continue
            confidence = min(1.0, count / 3.0)
            score = float(edge["mu"]) * max(float(edge["q"]), 0.12) * confidence * strength
            if score > best:
                best = score
        return best, matched

    def _score_action(self, obs: np.ndarray, action: int) -> float:
        source = self._make_source(obs, action)
        action_node = self.action_nodes[action]
        direct, matched_action_edge = self._best_indexed_value(source, action_node)

        # Two-step graph rollout. Value edges are retrieved from the action
        # index; no scan of the full hyperedge dictionary is required.
        second_best = 0.0
        first_predictions = self._predict_targets(source)
        for j, p_j in enumerate(first_predictions):
            if p_j < 0.12:
                continue
            for action2 in range(3):
                future = np.zeros(12, dtype=float)
                future[j] = p_j
                for cue in range(8):
                    if cue != j:
                        future[cue] = source[cue] * 0.85
                future[self.action_nodes[action2]] = 1.0
                future_best, _ = self._best_indexed_value(future, self.action_nodes[action2])
                if future_best > second_best:
                    second_best = 0.90 * future_best
        score = max(direct, second_best)

        if not matched_action_edge:
            score = max(score, 0.52 if action != 1 else 0.50)
        return 0.0 if score < 0.0 else 1.0 if score > 1.0 else float(score)

    def _select_action_validated(self, obs: np.ndarray, *, autonomous: bool = False) -> int:
        scores = [self._score_action(obs, action) for action in range(3)]
        choice = int(np.argmax(scores))
        self.last_debug["selection"] = {
            "scores": [{"action": a, "value": float(scores[a])} for a in range(3)],
            "chosen": choice,
            "exploration": bool(autonomous and scores[choice] <= 0.52),
        }
        return choice

    def select_action(self, observation: np.ndarray) -> int:
        if self._pending:
            raise RuntimeError("Call learn() after each forward()")
        return self._select_action_validated(self._validate_observation(observation))

    def forward(self, observation: np.ndarray, action: int | None = None) -> int:
        if self._pending:
            raise RuntimeError("Call learn() after each forward()")
        obs = self._validate_observation(observation)
        if action is None:
            action = self._select_action_validated(obs, autonomous=True)
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, 2, or None")
        action = int(action)
        self._pending_obs = obs.copy()
        self._pending_action = action
        self._pending_source = self._make_source(obs, action)
        self._pending = True
        self.last_debug["action"] = action
        return action

    def _update_edge(self, ants: Antecedents, target: int, strength: float,
                     actual: float, ref: float) -> None:
        key = (ants, target)
        edge = self.hyperedges.get(key)
        if edge is None:
            if len(ants) == 2:
                self._register_legacy_pair((ants[0], ants[1]))
            edge = {"q": 0.0, "q_minus": 0.0, "n": 0, "total": 0.0,
                    "ema": 0.5, "mu": 0.5, "lag": np.zeros(3, dtype=float),
                    "lag_n": np.zeros(3, dtype=np.int64),
                    "in_pred_pool": False, "in_inhib_pool": False,
                    "prediction_indexed": False, "traceable": False}
            self.hyperedges[key] = edge
            for node in ants:
                self._edge_index[node].add(key)
                if target == self.value_node and node in self.action_nodes:
                    self._value_index_by_action[node].add(key)

        n = edge["n"] + 1
        total = edge["total"] + actual
        ema = 0.88 * edge["ema"] + 0.12 * actual
        mu = 0.5 * (total / n) + 0.5 * ema
        if mu < 0.0:
            mu = 0.0
        elif mu > 1.0:
            mu = 1.0

        q = edge["q"]
        delta = mu - ref
        if delta > 0.10:
            confirm = strength if mu >= 0.80 else strength * min(1.0, delta / 0.25)
            q += self.ETA_PLUS * confirm * (1.0 - q)
        else:
            q -= self.ETA_MINUS * strength * q
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        q_minus = edge["q_minus"]
        if actual < 0.5 and len(ants) > 1 and ref >= 0.5:
            q_minus += self.ETA_PLUS * strength * (1.0 - q_minus)
        elif actual >= 0.5:
            q_minus -= self.ETA_MINUS * strength * q_minus
        if q_minus < 0.0:
            q_minus = 0.0
        elif q_minus > 1.0:
            q_minus = 1.0

        edge["n"] = n
        edge["total"] = total
        edge["ema"] = ema
        edge["mu"] = mu
        edge["q"] = q
        edge["q_minus"] = q_minus

        traceable = q >= 0.05 or q_minus >= 0.05
        if traceable != edge["traceable"]:
            if traceable:
                self._traceable_targets[ants].add(target)
            else:
                self._traceable_targets[ants].discard(target)
            edge["traceable"] = traceable

        pool_key = key
        if q > 0.0:
            self.pred_pool.nodes[pool_key] = q
            edge["in_pred_pool"] = True
        elif edge["in_pred_pool"]:
            self.pred_pool.nodes.pop(pool_key, None)
            edge["in_pred_pool"] = False
        if q_minus > 0.0:
            self.pred_inhib.nodes[pool_key] = q_minus
            edge["in_inhib_pool"] = True
        elif edge["in_inhib_pool"]:
            self.pred_inhib.nodes.pop(pool_key, None)
            edge["in_inhib_pool"] = False

        if target < 8:
            eligible = mu * max(q, 0.15) * min(1.0, n / 3.0) >= 0.12
            if eligible != edge["prediction_indexed"]:
                index = self._prediction_index_by_target[target]
                if eligible:
                    index.add(pool_key)
                else:
                    index.discard(pool_key)
                edge["prediction_indexed"] = eligible
        elif target == self.value_node:
            self.q_val_stats[(ants, target)] = [mu, float(n)]

        if len(ants) == 1:
            lag = edge["lag"]
            if lag[1] or lag[2]:
                self.vessel_q[(ants[0], "mid", target)] = q * float(lag[1])
                self.vessel_q[(ants[0], "slow", target)] = q * float(lag[2])

    def _update_temporal_edges(self, source: np.ndarray, target: np.ndarray,
                               active_keys: list[tuple[Antecedents, float]]) -> None:
        active_antecedents = {ants for ants, _ in active_keys}

        # A discharged edge cannot fire again until its antecedent becomes
        # inactive. This prevents repeated consumption while a cue is held.
        for key in tuple(self._discharged):
            if key[0] not in active_antecedents:
                self._discharged.discard(key)

        # Only live traces are decayed. Inactive edges are never enumerated.
        for key, trace in tuple(self.edge_trace.items()):
            trace[0] *= 0.25
            trace[1] = max(0.85 * trace[0], 0.55 * trace[1])
            trace[2] = max(0.80 * trace[1], 0.70 * trace[2])
            self.edge_age[key] = self.edge_age.get(key, 0) + 1
            if max(trace) < 0.03:
                self.edge_trace.pop(key, None)
                self.edge_age.pop(key, None)
                self._trace_index_by_target[key[1]].discard(key)

        # Activate only indexed hyperedges belonging to current active subsets.
        for ants, strength in active_keys:
            for j in self._traceable_targets.get(ants, ()):
                key = (ants, j)
                if key in self._discharged:
                    continue
                if key not in self.edge_trace:
                    self.edge_trace[key] = np.zeros(3, dtype=float)
                    self.edge_age[key] = 0
                    self._trace_index_by_target[j].add(key)
                trace = self.edge_trace[key]
                if strength > trace[0]:
                    trace[0] = strength

        # Only inspect traces whose own target actually occurred.
        for j in np.flatnonzero(target >= 0.5):
            for key in tuple(self._trace_index_by_target[int(j)]):
                trace = self.edge_trace.get(key)
                if trace is None:
                    continue
                edge = self.hyperedges.get(key)
                if edge is None or edge["mu"] < 0.5 or max(trace) < 0.08:
                    continue
                lag_index = 0 if self.edge_age.get(key, 0) <= 1 else 1 if self.edge_age.get(key, 0) <= 2 else 2
                edge["lag_n"][lag_index] += 1
                edge["lag"][lag_index] += 1.0
                total = float(np.sum(edge["lag"]))
                if total:
                    edge["lag"] /= total
                q = float(edge["q"])
                self.pred_mid_pool.nodes[key] = q * float(edge["lag"][1])
                self.pred_slow_pool.nodes[key] = q * float(edge["lag"][2])
                if len(key[0]) == 1:
                    node = key[0][0]
                    self.vessel_q[(node, "mid", j)] = q * float(edge["lag"][1])
                    self.vessel_q[(node, "slow", j)] = q * float(edge["lag"][2])
                self.edge_trace.pop(key, None)
                self.edge_age.pop(key, None)
                self._trace_index_by_target[int(j)].discard(key)
                self._discharged.add(key)

        self._project_traces()

    def learn(self, next_observation: np.ndarray) -> None:
        if not self._pending:
            raise RuntimeError("Call forward() before learn()")
        target = self._validate_observation(next_observation)
        source = self._pending_source
        active_keys = self._active_keys(source, self._pending_obs)
        # Subset expectations are prepared once per active antecedent, after
        # all lower-order subsets have been updated in this same transition.
        for ants, strength in active_keys:
            refs = self.p_base.copy()
            if len(ants) > 1:
                for subset in self._subsets(ants):
                    for j in range(12):
                        subset_edge = self.hyperedges.get((subset, j))
                        if subset_edge is not None and subset_edge["mu"] > refs[j]:
                            refs[j] = float(subset_edge["mu"])
            for j in range(12):
                self._update_edge(ants, j, strength, float(target[j]), float(refs[j]))

        # Refresh base-rate statistics after edge updates.
        self.total_steps += 1
        self._base_counts += (target >= 0.5).astype(np.int64)
        exact_base = self._base_counts / max(1, self.total_steps)
        self._base_ema = 0.98 * self._base_ema + 0.02 * (target >= 0.5)
        self.p_base = 0.5 * exact_base + 0.5 * self._base_ema
        self.base_rate = self.p_base.copy()

        self._update_temporal_edges(source, target, active_keys)
        self.last_debug.update({"W": float(target[11]), "active_hyperedges": len(active_keys),
                                "active_edge_traces": len(self.edge_trace)})
        self._pending = False

    def _normalize_antecedents(self, antecedents: int | Iterable[int]) -> Antecedents:
        return self._expand_legacy_antecedents(antecedents)

    def q_pred(self, antecedents: int | Iterable[int], consequent: int) -> float:
        ants = self._normalize_antecedents(antecedents)
        target = int(consequent)
        edge = self.hyperedges.get((ants, target))
        if edge is not None:
            return float(edge["q"])
        # Compatibility projection for a one-node action query: the strongest
        # learned hyperedge containing that action vertex and the target.
        if len(ants) == 1 and ants[0] in self.action_nodes:
            return max((float(e["q"]) for (a, j), e in self.hyperedges.items()
                        if j == target and ants[0] in a), default=0.0)
        return 0.0

    def q_pred_temporal(self, antecedents: int | Iterable[int], consequent: int) -> dict[str, float]:
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        if edge is None:
            return {"fast": 0.0, "mid": 0.0, "slow": 0.0}
        q = float(edge["q"])
        lag = edge["lag"]
        return {"fast": q * float(lag[0]), "mid": q * float(lag[1]), "slow": q * float(lag[2])}

    def q_inhib(self, antecedents: int | Iterable[int], consequent: int) -> float:
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        return float(edge["q_minus"]) if edge else 0.0

    def q_val(self, antecedents: int | Iterable[int], consequent: int) -> float:
        ants = self._normalize_antecedents(antecedents)
        edge = self.hyperedges.get((ants, int(consequent)))
        return float(edge["mu"]) if edge and int(consequent) == 11 else 0.0

    def expected_at_delay(self, antecedent: int | Iterable[int], consequent: int, delay: int) -> float:
        if delay < 1:
            raise ValueError("delay must be >= 1")
        ants = self._normalize_antecedents(antecedent)
        edge = self.hyperedges.get((ants, int(consequent)))
        if edge is None:
            return 0.0
        index = 0 if delay == 1 else 1 if delay == 2 else 2
        return float(edge["q"] * edge["lag"][index])

    def predict_from_trace(self, trace: np.ndarray | None = None,
                           *, include_inhibition: bool = True) -> np.ndarray:
        source = self.S_trace.copy() if trace is None else np.asarray(trace, dtype=float).copy()
        out = np.zeros(12, dtype=float)
        for (ants, target), edge in self.hyperedges.items():
            strength = min((float(source[i]) for i in ants), default=0.0)
            if strength <= 0.0:
                continue
            positive = float(edge["q"]) * float(edge["mu"]) * strength
            negative = float(edge["q_minus"]) * strength if include_inhibition else 0.0
            out[target] = float(D(A(out[target], positive), negative))
        return out

    def pool_snapshot(self) -> list[dict[str, Any]]:
        return [{"antecedents": list(ants), "consequent": target, "Q": float(edge["q"])}
                for (ants, target), edge in self.hyperedges.items() if edge["q"] > 0.0]

    def inhibitory_snapshot(self) -> list[dict[str, Any]]:
        return [{"antecedents": list(ants), "consequent": target, "Q_minus": float(edge["q_minus"])}
                for (ants, target), edge in self.hyperedges.items() if edge["q_minus"] > 0.0]

    def chunk_snapshot(self) -> list[dict[str, Any]]:
        # No chunk vertices exist in the homogeneous hypergraph.
        return []

    def _valid_pair(self, a: int, b: int) -> bool:
        return a != b and 0 <= a <= 10 and 0 <= b <= 10


def make_observation(position: int, precursor: int | None = None, food: int | None = None,
                     noise: float = 0.0, wellbeing: float = 0.5) -> np.ndarray:
    if position not in (0, 1, 2):
        raise ValueError("position must be in {0, 1, 2}")
    obs = np.zeros(12, dtype=float)
    obs[position] = 1.0
    if precursor in (0, 1):
        obs[3 + precursor] = 1.0
    elif precursor is not None:
        raise ValueError("precursor must be None, 0, or 1")
    if food in (0, 1):
        obs[5 + food] = 1.0
    elif food is not None:
        raise ValueError("food must be None, 0, or 1")
    obs[7] = float(noise)
    obs[11] = float(wellbeing)
    return obs


__all__ = ["BlindGateGraphBrain", "make_observation"]
