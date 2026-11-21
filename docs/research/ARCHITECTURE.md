# Current architecture

[v60](../../development/agents/v60.py) is the accepted controller. It attaches
`JointReward` directly to the canonical factor model through
[`FactorController`](../../development/agents/factor_controller.py); it does not
construct earlier numbered agents. [v12](../../development/agents/v12.py) is the
benchmark comparator, not a v60 dependency.

## One real transition

1. **Observe.** The environment supplies an observation `x`, including the
   designated value channel. The [agent interface](../../development/agents/__init__.py)
   does not pass task labels, oracle output, or benchmark metrics to the controller.
2. **Choose.** `step(x)` draws from the controller RNG and asks the factor model
   for two-step action values. Projection and rollout use hypothetical branches
   without updating live learned state, opening a real prediction cache, or
   advancing the controller RNG. The controller selects an action, calls the
   real `factor.predict(x, action)` to prepare the learning cache, and records
   `(x, action)` as pending.
3. **Act.** The environment applies the chosen action and returns the next
   observation `y`. The [controller benchmark](../../development/eval/benchmark.py)
   keeps the separately returned reward and oracle data for scoring, not as
   additional inputs to `step` or `learn`.
4. **Learn.** `learn(y)` consumes the pending transition once. `FactorModel.learn`
   updates the reward member from the real transition (using `y[value_index]`)
   and updates sensory inference, then the controller clears its pending state.
   A second `step` while a transition is pending, or a `learn` without one, is
   an error.

## State ownership

| Component | Owns or computes | Implementation |
| --- | --- | --- |
| Controller | Action selection RNG and pending transition | [Factor controller](../../development/agents/factor_controller.py), [v60 defaults](../../development/agents/v60.py) |
| Sensory inference | Learned local dependencies, latent state, predictive emissions, and the real prediction cache | [Factor model](../../development/models/factor.py), [process discovery](../../development/models/discovery.py) |
| Action-conditioned reward | Reward statistics and a bounded window of real transitions; compares pooled, unary, and compound partitions on the same evidence | [Reward members](../../development/models/factor_reward.py), [context](../../development/models/factor_context_reward.py), [regimes](../../development/models/factor_regime_reward.py), [signed event selection](../../development/models/factor_event_reward.py), [joint selection](../../development/models/factor_joint_reward.py) |
| Planning | Temporary projections and branch values; no ownership of live learned state | [Sensory projection](../../development/models/factor_projection.py), [two-step rollout](../../development/models/factor_rollout.py), [information value](../../development/models/factor_information.py) |

## Roles of retained code and data

| Role | Current consumers or entry points |
| --- | --- |
| Accepted controller | `v60`, with joint reward and isolation regression coverage |
| Comparator and legacy benchmark baseline | `v12` is the DEV comparator and depends on `v11`. `v11` also remains the default of several existing evaluation CLIs. |
| Retained reference agents | `v15`, `v17`, and `v18` have dedicated regression tests. `v21`, `v22`, `v23`, `v28`, `v37`, and `v38` are exercised by [control isolation tests](../../development/tests/test_factor_control.py); `v21` is also the default in the exposed controller runner. These are not in the v60 construction path. |
| Evaluation and operations | [Environments](../../development/envs/) generate observations and transitions. [Evaluation](../../development/eval/) drives episodes and computes metrics. The [harness](../../development/harness/) handles journals, snapshots, and orchestration, not agent learning. |
| Accepted evidence | [D65 and AR-7 replay](../EVIDENCE.md) validates saved evidence with no new episodes. The freeze identifies the original code, not the current checkout. |
| Retired experiments | Recover original drivers and configurations through the recorded [Git history](../EVIDENCE.md), not by treating spent cohorts as fresh evidence. |

## Regression checks

The [CI regression job](../../.github/workflows/regression.yml) runs the full
suite with `FACTOR_ISOLATION_CANDIDATE=v60`, explicitly including v60 in the
no-legacy-import and planning purity checks. It also compares both accepted
verdicts byte for byte. Passing CI does not constitute a new qualification.
The current acceptance result is scoped to [the tested suites](AR7.md); blind
T4 has not been opened (see the [roadmap](ROADMAP.md)).
