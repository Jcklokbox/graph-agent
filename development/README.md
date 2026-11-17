# Development workspace

This directory contains the executable project and its current behavioral experiments. The theoretical workspace is [`../research/`](../research/README.md).

## Files

- `blind_gate.py` — delayed-food environment and baseline policies.
- `blind_gate_graph.py` — topology-configurable predictive hypergraph learner/controller.
- `ops.py` — bounded algebra operations.
- `benchmark_envs.py` — cancellation, variable-delay, and distant-goal environments used by behavioral experiments.
- `test_despecialized.py` — five principle-level experiments: informative prediction and conjunctions, autonomous behavior under channel permutations, conditional inhibition and variable-delay learning, distant value seeking, and reversal learning.

## Running

From this directory:

```bash
python -m py_compile ops.py blind_gate.py blind_gate_graph.py benchmark_envs.py test_despecialized.py
python -m pytest -q
```

The code requires NumPy; pytest runs the behavioral suite. The thresholds are falsifiable research criteria, not a release checklist. A red result should trigger analysis of the failed principle and experiment, not a task-specific patch. The previous implementation-coupled tests and their stale JSON outputs were removed because they measured virtual chunks, pools, and compatibility structures rather than the learning principles now being evaluated.

## Current model boundary

Bounded expectations, confidence, separate excitation/inhibition, and temporal eligibility are stored on directed hyperedges. Action selection combines predicted wellbeing with local novelty from active edge evidence. The model has no task policy, separate action-value table, fixed-order pruning, or legacy API path. Known limitations include exponentially many active subsets, deterministic tie-breaking, weak autonomous results, and shallow value propagation. See `../research/current-development.md` and `../research/foundational-principles.md`.
