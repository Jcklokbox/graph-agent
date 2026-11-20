# Archive

Superseded material, kept for reference. Nothing in `development/` imports it, and it is not
maintained.

- `development/` — the original `BlindGateGraphBrain` hypergraph prototype (`blind_gate_graph.py`,
  `blind_gate.py`, `ops.py`, `test_despecialized.py`), the earlier benchmark scripts of the
  posterior graph (`benchmark_envs.py`, `run_posterior_scenarios.py`, `transfer_suite.py`;
  replaced by `development/tasks.py` and `development/benchmark.py`) and the last pre-
  consolidation agent `posterior_graph_v10.py` (Round 12; imports nothing from here, runs with
  the current `tasks.py` / `benchmark.py`).
- `research/` — theory notes of the hypergraph prototype and of the early posterior-graph
  design, the component probe and its results, the permutation-equivariance sweep, and
  `round-log-r1-r12.md`, the full log of Rounds 1–12.
