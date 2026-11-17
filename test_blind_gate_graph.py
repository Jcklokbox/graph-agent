from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blind_gate import BlindGateEnv
from blind_gate_graph import BlindGateGraphBrain, make_observation
from ops import D


def controlled_transitions():
    """Seven requested links represented by a compact deterministic cycle."""
    return [
        # Center -> left; the precursor also predicts food V5.
        (make_observation(1, precursor=0), 0,
         make_observation(0, food=0, wellbeing=0.5)),
        # Center -> right; precursor predicts food V6.
        (make_observation(1, precursor=1), 2,
         make_observation(2, food=1, wellbeing=0.5)),
        # Right -> center with a left action.
        (make_observation(2), 0, make_observation(1, wellbeing=0.5)),
        # Left -> center with a right action.
        (make_observation(0), 2, make_observation(1, wellbeing=0.5)),
        # Background transition keeps V1 and output nodes from being constants.
        (make_observation(1), 1, make_observation(1, wellbeing=0.5)),
        # Precursor-only examples make the food links independent of movement.
        (make_observation(0, precursor=0), 1,
         make_observation(0, food=0, wellbeing=0.5)),
        (make_observation(2, precursor=1), 1,
         make_observation(2, food=1, wellbeing=0.5)),
    ]


def train_transitions(brain: BlindGateGraphBrain, transitions, repeats: int) -> None:
    for _ in range(repeats):
        for observation, action, next_observation in transitions:
            brain.forward(observation, action)
            brain.learn(next_observation)


def run_test_1() -> dict:
    brain = BlindGateGraphBrain(seed=0)
    transitions = controlled_transitions()
    train_transitions(brain, transitions, 80)

    rows = [
        ("V3 -> V5", brain.q_pred(3, 5), True),
        ("V4 -> V6", brain.q_pred(4, 6), True),
        ("(V1, V8) -> V0", brain.q_pred((1, 8), 0), True),
        ("(V1, V8) -> V1", brain.q_pred((1, 8), 1), False),
        ("(V2, V8) -> V1", brain.q_pred((2, 8), 1), True),
        ("(V1, V10) -> V2", brain.q_pred((1, 10), 2), True),
        ("(V0, V10) -> V1", brain.q_pred((0, 10), 1), True),
    ]
    table = [
        {
            "link": name,
            "Q": float(q),
            "expected": "> 0.99" if expected else "< 0.01",
            "pass": bool(q > 0.99 if expected else q < 0.01),
        }
        for name, q, expected in rows
    ]
    return {
        "repetitions": 80,
        "table": table,
        "all_pass": all(row["pass"] for row in table),
        "pred_pool": brain.pred_pool.stats(),
        "chunks": brain.chunk_snapshot(),
    }


def run_random(seed_env: int, seed_brain: int, seed_policy: int, steps: int, *, compact_learning: bool = False) -> dict:
    env = BlindGateEnv(seed=seed_env)
    brain = BlindGateGraphBrain(
        seed=seed_brain, compact_learning=compact_learning,
        max_chunks=0 if compact_learning else BlindGateGraphBrain.MAX_CHUNKS,
    )
    rng = np.random.default_rng(seed_policy)
    observation = env.reset()
    for _ in range(steps):
        action = int(rng.integers(0, 3))
        brain.forward(observation, action)
        next_observation, _, _, _ = env.step(action)
        brain.learn(next_observation)
        observation = next_observation

    v7_edges = [edge for edge in brain.pool_snapshot() if 7 in edge["antecedents"]]
    return {
        "steps": steps,
        "surviving_pred_edges": len(brain.pred_pool.nodes),
        "v7_edge_count": len(v7_edges),
        "v7_max_Q": max((edge["Q"] for edge in v7_edges), default=0.0),
        "v7_edges_Q_ge_0_5": sum(edge["Q"] >= 0.5 for edge in v7_edges),
        "pred_pool_stats": brain.pred_pool.stats(),
        "v7_edges": v7_edges,
    }


def chunking_transitions():
    """V11 requires position + action chunk + precursor."""
    transitions = [
        # True motor-position contexts.
        (make_observation(1, precursor=0), 0,
         make_observation(0, food=0, wellbeing=1.0)),
        (make_observation(1, precursor=1), 2,
         make_observation(2, food=1, wellbeing=1.0)),
        # The same motor-position combinations without a precursor cause shock.
        *[(make_observation(1), 0,
           make_observation(0, wellbeing=0.0)) for _ in range(3)],
        *[(make_observation(1), 2,
           make_observation(2, wellbeing=0.0)) for _ in range(3)],
        # Same output action from the opposite side leads to the center.
        *[(make_observation(2, precursor=0), 0,
           make_observation(1, wellbeing=0.5)) for _ in range(6)],
        *[(make_observation(0, precursor=1), 2,
           make_observation(1, wellbeing=0.5)) for _ in range(12)],
        # Precursor alone is insufficient: wrong action gives shock.
        *[(make_observation(1, precursor=0), 2,
           make_observation(2, wellbeing=0.0)) for _ in range(4)],
        *[(make_observation(1, precursor=1), 0,
           make_observation(0, wellbeing=0.0)) for _ in range(4)],
        # Safe neutral contexts make the single precursor links uninformative.
        (make_observation(1, precursor=0), 1,
         make_observation(1, wellbeing=0.5)),
        (make_observation(1, precursor=1), 1,
         make_observation(1, wellbeing=0.5)),
    ]
    return transitions


def train_chunking_brain(consume_trace: bool) -> BlindGateGraphBrain:
    brain = BlindGateGraphBrain(seed=11, consume_trace=consume_trace)
    train_transitions(brain, chunking_transitions(), 120)
    return brain


def run_test_3() -> dict:
    soft = train_chunking_brain(consume_trace=False)
    hard = train_chunking_brain(consume_trace=True)

    def rows_for(brain: BlindGateGraphBrain) -> list[dict]:
        chunks = brain.chunk_by_components
        left_chunk = chunks.get((1, 8))
        right_chunk = chunks.get((1, 10))
        return [
            {
                "link": "((V1, V8), V3) -> V11",
                "Q": brain.q_pred((left_chunk, 3), 11) if left_chunk is not None else 0.0,
                "expected": "high",
            },
            {
                "link": "((V1, V10), V4) -> V11",
                "Q": brain.q_pred((right_chunk, 4), 11) if right_chunk is not None else 0.0,
                "expected": "high",
            },
            {
                "link": "(V1, V8) -> V11",
                "Q": brain.q_pred((1, 8), 11),
                "expected": "low",
            },
            {
                "link": "(V3) -> V11",
                "Q": brain.q_pred(3, 11),
                "expected": "low / insufficient",
            },
            {
                "link": "(V4) -> V11",
                "Q": brain.q_pred(4, 11),
                "expected": "low / insufficient",
            },
        ]

    soft_rows = rows_for(soft)
    hard_rows = rows_for(hard)

    def chunk_constraints(brain: BlindGateGraphBrain) -> dict:
        chunk_ids = sorted(brain.chunk_components)
        component_pairs_blocked = all(
            not brain._valid_pair(chunk_id, component)
            for chunk_id, components in brain.chunk_components.items()
            for component in components
        )
        chunk_pairs_blocked = all(
            not brain._valid_pair(a, b)
            for index, a in enumerate(chunk_ids)
            for b in chunk_ids[index + 1:]
        )
        return {
            "count": len(chunk_ids),
            "within_max_chunks": len(chunk_ids) <= brain.max_chunks,
            "chunk_with_own_component_blocked": component_pairs_blocked,
            "chunk_with_chunk_blocked": chunk_pairs_blocked,
        }

    soft_constraints = chunk_constraints(soft)
    hard_constraints = chunk_constraints(hard)

    # A final identical sample is learned in both modes. Verify that links for
    # transition and wellbeing update in the same learn() call.
    probes = {}
    for name, brain in (("soft_decay", soft), ("hard_consumption", hard)):
        before_transition = brain.q_pred((1, 8), 0)
        chunk_id_before = brain.chunk_by_components.get((1, 8))
        before_wellbeing = (
            brain.q_pred((chunk_id_before, 3), 11) if chunk_id_before is not None else 0.0
        )
        obs = make_observation(1, precursor=0)
        nxt = make_observation(0, food=0, wellbeing=1.0)
        brain.forward(obs, 0)
        trace_before_learn = brain.S_trace.copy()
        brain.learn(nxt)
        chunk_id = brain.chunk_by_components[(1, 8)]
        probes[name] = {
            "transition_Q_before": before_transition,
            "transition_Q_after": brain.q_pred((1, 8), 0),
            "wellbeing_Q_before": before_wellbeing,
            "wellbeing_Q_after": brain.q_pred((chunk_id, 3), 11),
            "active_trace_before_learn_V1": float(trace_before_learn[1]),
            "active_trace_before_learn_V8": float(trace_before_learn[8]),
            "trace_after_learn_V1": float(brain.S_trace[1]),
            "trace_after_learn_V8": float(brain.S_trace[8]),
        }

    return {
        "soft_decay": {
            "rows": soft_rows,
            "chunks": soft.chunk_snapshot(),
            "trace_mode": "no destructive consumption; normal decay",
        },
        "hard_consumption": {
            "rows": hard_rows,
            "chunks": hard.chunk_snapshot(),
            "trace_mode": "confirmed antecedent traces are zeroed after learning",
        },
        "same_step_probe": probes,
        "chunk_constraints": {
            "soft_decay": soft_constraints,
            "hard_consumption": hard_constraints,
        },
        "pass": (
            soft_rows[0]["Q"] > 0.99
            and soft_rows[1]["Q"] > 0.99
            and soft_rows[2]["Q"] < 0.01
            and hard_rows[0]["Q"] > 0.99
            and hard_rows[1]["Q"] > 0.99
            and hard_rows[2]["Q"] < 0.01
            and probes["soft_decay"]["trace_after_learn_V1"] > 0.0
            and probes["hard_consumption"]["trace_after_learn_V1"] == 0.0
            and soft_constraints["within_max_chunks"]
            and hard_constraints["within_max_chunks"]
            and soft_constraints["chunk_with_own_component_blocked"]
            and hard_constraints["chunk_with_own_component_blocked"]
            and soft_constraints["chunk_with_chunk_blocked"]
            and hard_constraints["chunk_with_chunk_blocked"]
        ),
    }


def run() -> dict:
    test1 = run_test_1()
    random100 = run_random(19, 20, 21, 100)
    random1000 = run_random(19, 20, 21, 1000)
    test3 = run_test_3()

    # The 100-step historical result is retained as a supplied comparison row.
    random_table = [
        {
            "steps": 100,
            "old_surviving_edges": 702,
            "new_surviving_edges": random100["surviving_pred_edges"],
            "old_V7_edges_Q_ge_0_5": 8,
            "new_V7_edges_Q_ge_0_5": random100["v7_edges_Q_ge_0_5"],
            "new_V7_max_Q": random100["v7_max_Q"],
        },
        {
            "steps": 1000,
            "old_surviving_edges": None,
            "new_surviving_edges": random1000["surviving_pred_edges"],
            "old_V7_edges_Q_ge_0_5": None,
            "new_V7_edges_Q_ge_0_5": random1000["v7_edges_Q_ge_0_5"],
            "new_V7_max_Q": random1000["v7_max_Q"],
        },
    ]

    # Trace decay check remains the same D operation used by the graph.
    decay = [1.0]
    value = 1.0
    for _ in range(8):
        value = float(D(value, 0.05))
        decay.append(value)

    return {
        "test_1_compatibility": test1,
        "test_2_noise_filtering": {
            "comparison_table": random_table,
            "random_100": random100,
            "random_1000": random1000,
            "pass": (
                random100["v7_edges_Q_ge_0_5"] < 8
                and random100["v7_max_Q"] < 0.5
                and random1000["v7_edges_Q_ge_0_5"] == 0
            ),
        },
        "test_3_chunking_and_trace": test3,
        "trace_decay_reference": decay,
        "all_pass": test1["all_pass"]
        and (random100["v7_edges_Q_ge_0_5"] < 8)
        and (random100["v7_max_Q"] < 0.5)
        and (random1000["v7_edges_Q_ge_0_5"] == 0)
        and test3["pass"],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("blind_gate_graph_test.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["all_pass"]:
        raise SystemExit("At least one benchmark criterion failed")
