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

### Finite-work joint rollout detail

The existing marginal readout defines a joint law by mixing the regime-conditional product emissions that `_activity` actually uses. For quiet input, draw the shared zero-event clock once, then (when nonempty) draw the particle/regime from its pre-clock mass and draw the product conditional on nonempty. **No rejection loop:** choose the first success with mass `p_i × ∏_{j<i}(1−p_j)/(1−∏_j(1−p_j))`, then sample only suffix Bernoullis. This is exact in linear work even for a nearly silent model. Conditional process heads currently enter that product as marginals; adding covariance via shared process-state draws would change the quiet law and needs its own calibration and predictor validation rather than being called exact. At a real root, evaluate alternative actions from the *current* process belief; `predict(..., hypothetical=True)` currently means a probe after assimilating another observation, and is not the right current-root API. A trial implementation may only change a temporary readout/conditioning state until real `learn`.

### D-108 candidate model (not itself gate acceptance)

`v48` is the cause-scoped, signed event candidate. An actuated dwell witness
must predict that alternative actions can retain or exit a local sensory
mark; weak action correlations alone are insufficient. Its compound control
scope has at most four bits. The finite Markov state statistics are learned
online (not supplied transition/reward entries). A bounded transition replay
reprojects sufficient statistics when the scope changes. Actual persistent
transition surprise can recall one previous action/reward mode or start a
new one; guesses do not touch that state.

For a reached control state the reward model has at most sixteen outcome
classes. Every real reward adds its actual magnitude, and the within-class
mean carries that magnitude rather than converting `.45` to a success/failure.
One sparse unary temporal witness is compared to the action-selected pooled
null. A sensory-moderator witness can split a supported parent on both sides
of a concurrent observation, with signed estimates and actual outcome
likelihoods. There is no favorable-state/path oracle or inherited v12 readout.

Planning enumerates bounded control outcomes and, for at most three active
sensory parents, their exact marginal joint law (including the quiet-clock
atom); otherwise it uses five common-random-number draws from the same law.
When no live reward parent reads a new sensory emission, integrate that view
out exactly. The inner action maximum is computed after a virtual real-atom
update of the selected reward scope, with fixed structure and motion. The
expected two-step value can be computed without epistemic exploration
(`provisional_values(..., optimistic=False)`). Root selection with optimism
also considers bounded unresolved-onset and untried-transition hypotheses,
which are exploratory estimates, **not calibrated expected rewards**. The
procedure is a finite-depth posterior projection; it is not a full Bayes
search over all future graph structures or counterfactual intervention history.

Control cardinality (≤16), replay length (≤2·16·A), actions, temporal lags,
reward bins, moderator anchors and support enumeration are bounded. The
contextual bank scans at most four onset sources against J potential
moderators; it does not maintain all pairs. The scope filter, replay,
projected sensor rows and real update visit O(J) entries at fixed N and those
bounds, so full act+learn retains the passive O(Σ_g |P_g|) construction.
Small-width ratios validate overhead, not the asymptotic proof itself.

### Frozen result

The v48 DEV-16 qualification completed but failed the registered
noninferiority rule: 240 matched pairs, mean difference −0.003298,
99% CI [−0.043362, +0.036767] against lower limit −0.02. See
`development/results/ar5/gate-dev16-v48/DEV.json` and `validation.json`.
The passing isolation and scaling checks do not make v48 an accepted
agent. The passive 3b-1 result is unchanged.
