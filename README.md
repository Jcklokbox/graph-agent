# Posterior Graph

An online, sparse, hierarchical posterior-predictive agent: it learns from anonymous
observation channels which channels matter, what the actions control and when delayed
effects arrive, and plans two steps ahead with Bayesian optimism. v11 / v12 use the standard library only;
v18's latent-cause predictor, the environments and the harnesses use NumPy.

- [`development/`](development/README.md) — the agents (agent of record `agents/v18.py`, see
  [`research/results.md`](research/results.md); history v11 → v12 → v18), the environments
  (behavioural tasks, the FHP family of factored hidden-process worlds with an exact filter,
  conditioning phenomena), the evaluation harnesses for the four evidence tiers and the tests.
- [`research/`](research/README.md) — the goal and evidence tiers ([`goal.md`](research/goal.md)),
  the decision log with pre-registered predictions ([`decisions.md`](research/decisions.md)),
  durable lessons ([`insights.md`](research/insights.md)), the mathematical model
  ([`model.md`](research/model.md)), measured results ([`results.md`](research/results.md))
  and the theoretical orientation documents.
- [`archive/`](archive/) — superseded code and notes (the original `BlindGateGraphBrain`
  prototype, earlier benchmarks, the round-by-round log of Rounds 1–12). Kept for reference;
  nothing in `development/` imports it.

Research is not a frozen specification of the code: theory may precede the implementation
and the implementation may expose weaknesses in the theory.
