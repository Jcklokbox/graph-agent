# AR-5: conditional 3b-2 execution spec

Not an accepted agent. Do not dispatch a candidate against DEV until an immutable 3b-1 freeze passes C, A, h1, B and new holdout. The historical `v20` is archived: prediction at depth 0 still included v12 reward and training; depth 1 and bonus used v12 tables. Reusing that path would invalidate isolation.

## Forecast interface

1. Add a **bounded reward member inside the factor model** on designated value parents chosen by local prequential/conditional residual evidence. The designated value is allowed by the interface; no food, cue, position, task or context labels. In DEV the actual observed wellbeing has atoms such as .45 and .5 and can be remapped. Accumulate positive-weight bounded-real moments or use a proper bounded distribution; do not turn a fractional reward into one Bernoulli trial. A Dirichlet random-measure prior gives `m=(λm0+Σr)/(λ+n)`, epistemic `Var(E[R])=(E[R²]−m²)/(λ+n+1)`, and prior-seeded empirical expectations with bounded sufficient statistics. A score for real-outcome structure needs a real predictive density/mass or a proper mean score—not a fictitious Bernoulli evidence ratio. Full tabulation over active history would reintroduce the rejected architecture.
2. Draw imagined observations once per root action from the **shared** branch latents of `M_factor`: common particle/regime, one selected family state per process, and that family's emissions. If a mixture chooses separate member hypotheses per coordinate, represent that in the generator. Reproduce marginal readouts on a tiny reference and track clock handling: its closed-form zero/nonempty mixture must not be silently replaced by independent marginals. A sampled branch or exact finite summation is permitted; use the same model train/read/rollout semantics. Offer reward-parent projections when only a bounded number of channels is required; do not enumerate all sensory configurations.
3. A hypothetical observation is read with the **previous action** in its likelihood, then the **next action** selects the successor prediction. Distinguish sensory inference from prospective parameter learning. Use ephemeral beliefs/read-only learned banks, no cloning the full N×K×channels arrays, no mutation of real counts, RNG, anchor or pending state. Existing same-action `GroupModel.peek` is not this interface.

## Choice and compute

Compute `Q2(a)=E[r1|b,a]+γ E_{x′,r1}[max_{a′} E[r2|b′,x′,a′]]` from the factor. A direct learned reward transition conditioned on x′ can forecast the second reward without predicting every coordinate of x″. Varying observations prompt learning if tentative posteriors are included. A separately reported epistemic contribution is the improvement vs a frozen-parameter evaluation on the **same** branch draws; do not add both the Bayes-adaptive max and the same VOI again. Separate aleatoric variation from uncertainty in action-value differences. Bound action count, sample count and reward-parent footprint independent of J; cost all forecasts, candidate work, and learning. Use a scratch RNG for planning and fixed tie-noise semantics.

## Progression

| Check | Falsification test | Advance only when |
|---|---|---|
| Execution isolation | Forbid imports/calls into v11/v12 at evaluation; snapshot state before and after identical imagined updates | Model-only predictions, no side effects, no label assumptions |
| Exact finite fixture | Two actions, delayed reward, shared observation latent; enumerate the two-step value and marginal/epistemic terms | Match enumerated value and distinguish perfectly coupled vs independent bits with the same marginals |
| Mechanism screen | Same delayed food/position units D-33 isolated and same independent causal null | Preserve action timing and no systematic loss against the comparator; unit results not DEV evidence |
| Scaling | Warm pinned act+learn on fixed widths, separate cold start, allocations and dense false-merge fixture | Invariant O(J) with bounded local support and ≤2.50 pair at N200; no age-dependent shortcut |
| Paired judge | Fixed, hashed DEV-16 cache from v12 and normalization replay, 240 expected tuples | Lower bound >=−.02 at registered two-sided 99%; exposed predictor guards unchanged |
| Next capability (only after closure) | Blind procedural held-out T4 under new graph/parameter combinations before type-sharing | Then propose C3 latent types with transfer between contexts, separate learnable vs oracle ceilings and finite local inference. No claim about C3 from a C2 pass. |

If a powered paired comparison cannot fit the remaining allocation, report the available coverage and inconclusive result. Do not change sample size or CI after seeing performance; fresh pairs require new preregistration and the actual v12 run on those seeds. Neither aggregate v12 .891 nor T1/T2 pass substitutes for paired action evidence.
