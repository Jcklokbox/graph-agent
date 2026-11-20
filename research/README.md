# Research workspace

The theoretical side of the project. These notes are context for agents and contributors,
not a binding specification; where it matters they separate established observations,
modelling assumptions, hypotheses and open questions.

## Start here

**Checkpoint:** [checkpoint-ru.md](checkpoint-ru.md) — goal, working pipeline, standing rules,
current state and known drifts in one page; read it first. Full inventory of agents,
prototypes, test sets, results and speeds: [revision-2026-11-20-ru.md](revision-2026-11-20-ru.md).

0. [Goal, environment family and evidence tiers](goal.md) — the central question of the
   current research programme (slices 0–3), the FHP environment family, tiers T0–T4 and the
   statistical rules. [Decision log](decisions.md) — every run with its prediction written
   beforehand and its outcome. [Insights](insights.md) — durable lessons.
1. [The posterior-graph model](model.md) — the mathematical model of the current agent as
   one system: one statistic (the discounted conjugate cell), one structure test (the split
   Bayes factor), one way to combine context (a chain of shrinkage), expectation windows,
   latent regimes and the planner.
2. [Results](results.md) — measured behaviour (DEV selection, held-out sets run once),
   ablations of the consolidation round and the open failures.
3. [Foundational research principles](foundational-principles.md) — scientific orientation,
   architectural aspirations and standards for honest evidence.
4. [First-principles derivation](first-principles-derivation.md) — necessary conditions
   derived from the adaptive learning problem, with limits on what follows.
5. [Biological core](biological-core.md) — biological motivation and limits of the analogy.

Superseded notes (the earlier hypergraph theory, equation audits, the posterior-graph probe,
the full Round 1–12 log) are in [`../archive/research/`](../archive/research/).
