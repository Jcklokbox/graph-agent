# Current architecture

| Component | Implementation |
| --- | --- |
| Accepted controller | [v60](../../development/agents/v60.py) attaches the bounded joint reward member to the canonical factor model. The [controller lifecycle](../../development/agents/factor_controller.py) attaches the final reward member directly, with no historical agent construction chain. |
| Sensory inference | [Factor model](../../development/models/factor.py) and [discovery](../../development/models/discovery.py) learn local dependencies, latent process state and predictive emissions. |
| Action-conditioned reward | [Reward members](../../development/models/factor_reward.py), [context](../../development/models/factor_context_reward.py), [regimes](../../development/models/factor_regime_reward.py), [signed event selection](../../development/models/factor_event_reward.py), and [joint selection](../../development/models/factor_joint_reward.py). The joint selector compares pooled, unary and compound partitions on the same bounded real transition window. |
| Planning | [Sensory projection](../../development/models/factor_projection.py), [two-step rollout](../../development/models/factor_rollout.py), and [information value](../../development/models/factor_information.py). Hypothetical updates never alter live counts or RNG. |
| Benchmarks and orchestration | [Environments](../../development/envs/), [evaluation](../../development/eval/), [harness](../../development/harness/). [v12](../../development/agents/v12.py) is a comparator, not a v60 dependency. |

The acceptance result is scoped to [the tested suites](AR7.md). T4 on blind
combinations is still open; see the [roadmap](ROADMAP.md).
