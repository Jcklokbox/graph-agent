# Equation audit of the checked-in learner

**Purpose:** tie theory to actual source without treating field names as explanations. This is a source-level audit of `development/ops.py` and `development/blind_gate_graph.py`, not a validation of behavioral performance. Every claim below is directly checkable in those files. Runtime benchmark results are not newly measured here: `pytest` is unavailable in the current shell (`pytest: command not found`).

## 1. Exact operation semantics

For inputs in `[0,1]`, the current primitives are

`N(x)=1−x`, `A(x,y)=min(1,x+y)`, `M(x,y)=xy`, `D(x,y)=max(0,x−y)`.

The Q helper is

`Q' = min(1, max(0, Q − η₋ f) + η₊ c)`

for confirm `c`, falsify `f`, learning parameters in `[0,1]`. This is a bounded accumulation/decay rule. It is not a likelihood update, a probability calibration, or RW's linear signed weight update. Saturation means effects depend on update order and accumulated history.

These primitives also do not, by name alone, establish fuzzy/probabilistic logic. `N(x)=1−x` is the complement of a probability only when `x` denotes a probability of a specified event. `M(x,y)=xy` is the joint probability only under independence (or some separately justified factorization). `A(x,y)=min(1,x+y)` is **not** probability union in general: for independent events of probability `.5` each, union is `.75`, while `A(.5,.5)=1`.  `D(x,y)=max(0,x−y)` is positive-part difference, not Bayesian conditioning. Consequently, names such as AND/OR/NOT require an operational derivation from the quantity semantics. The current source's hyperedge activation uses `min` across antecedents, yet `min` is not the same conjunction operator as `M` or a calibrated joint probability. Nor do the primitives form familiar probability algebra: the counterexample `x=.25,y=.25,z=1` gives `M(x,A(y,z))=.25`, but `A(M(x,y),M(x,z))=.3125`; distributivity fails. Also `N(A(.25,.25))=.5`, while `M(N(.25),N(.25))=.5625`; this complement does not obey the corresponding De Morgan identity for these operations. These are exact counterexamples, not a numerical proof of broader properties.

## 2. Source representation and combinatorial growth

Given an observation `o`, the source for candidate action `a` copies every non-action input (including wellbeing) and sets the chosen action vertex to `1`. Every non-empty subset of positive source channels is enumerated. For `k` active channels there are `2^k−1` antecedents, with activation `min_i source_i`. There is no antecedent-order cap in the code. Thus runtime and possible edge updates grow exponentially with active-channel count. The minimum is a fuzzy-AND choice, not a probabilistic conjunction unless a specific dependence model justifies it.

For each active antecedent, the learner creates/update edges to **every** input target, including targets currently at zero. Previously traced edges also update, even when not active now. Therefore absence can contribute a negative comparison only if zero is a meaningful observed target, not merely a missing/sparse sensor report. There is no explicit observation mask. Action vertices are allowed in antecedents but excluded from the target loop (`input_nodes` excludes all action nodes); there are no learned action-node consequents. Thus motor activity is not fully symmetric with sensory/value activity in this implementation. The action set is supplied externally and selected by a separate score operation.

## 3. Expectations, reference, association and confidence

An edge expectation starts at `0.5`; with `α = expectation_rate × activation`, it updates as

`E' = (1−α)E + α y`,

where `y` is the current target value. This is an exponential moving average, not a Bayesian posterior; `expectation_rate` is not derived from sample count or observation noise.

Reference `R` is the maximum of background expectation (default `0.5`) and expectations of existing strict non-empty subsets having positive confidence. Importantly, `_update_edge` first changes `E` and **then** calculates `R`. Thus compounds compare their post-update estimate to the strongest lower-order estimate. For activation `x`,

`c = x max(0,E'−R)`, `f = x max(0,R−E')`.

If the edge expectation is near its fixed reference `R` and a fractional activation `x` produces a positive residual, then `E'−R ≈ expectation_rate × x × (y−R)`, so confirmation is approximately `expectation_rate × x² × (y−R)`. Thus graded feature activity is effectively weighted twice in association evidence; for binary `x∈{0,1}`, this reduces to a single activity factor. This is an algebraic consequence of updating the EMA with activation and then multiplying its deviation by activation again. It is not a standard linear eligibility/gradient weighting unless that quadratic response is intended.

Positive strength is `Q'(Q⁺, c, f; association_rate, extinction_rate)`; inhibition strength swaps `c` and `f`. These are two separately retained nonnegative accumulators, not one signed causal parameter. Because only one of `c,f` is nonzero in a single update, both can nevertheless persist across training histories and later co-activate in prediction. The max-reference rule is a heuristic competition criterion; it is not algebraically equivalent to RW, Mackintosh, Pearce–Hall, or a causal model comparison.

Confidence update is `C'=C + (association_rate × x)(1−C)`. At full activation with the default rate `.1`, after `n` exposures `C_n=1−.9^n`: confidence reaches about `.90` after 22 visits and `.99` after 44. It is accumulated exposure, independent of whether outcomes were predictable, surprising, accurate, or noisy. It is therefore not calibrated certainty and cannot by itself distinguish reliable evidence from repeated contradictory evidence. It also does not reset on reversal; any exploratory effect that depends on `1−C` fades with exposure even when the relation has become wrong.

The empty-antecedent background also starts at `0.5` and applies the same exponential average to each next-observation target. Its confidence increments by `association_rate(1−C)` on every transition. It estimates a marginal mean only under stationarity and appropriate sampling.

## 4. Action evaluation and local “novelty”

For active wellbeing edges, support is `activation × confidence × strength`. The predictor keeps the **maximum** positive support and maximum inhibitory support across edges, then returns

`P̂ = max(0, min(1, P_background + max_support⁺) − max_support⁻)`.

This does not sum independent causes, represent uncertainty in a probabilistic sense, or preserve the scale of the learned expectation. The use of max can ignore a second useful independent cause; the background plus excitation can saturate. Separate max inhibition can cancel unrelated excitation.

For a candidate action, “novelty” is the maximum over action-containing active edges of `activation × (1−confidence)`. Score is

`S = min(1, P̂ + novelty × (1−P̂))`.

This is optimism toward wellbeing's upper bound, not expected information gain: it does not estimate possible observations, posterior change, or whether learning will improve later decisions. Ties are resolved by `numpy.argmax` in action order, so deterministic label bias can break permutation symmetry before evidence differentiates actions.

## 5. Eligibility and timing

At each transition, current active antecedents get trace `max(old_trace, activation)` for each input target. Existing positive traces decay by `trace_decay` and increment integer age. For an inactive edge at the default decay `.8`, the trace half-life is `ln(.5)/ln(.8) ≈ 3.11` steps; residual trace after 10 and 20 steps is about `.1074` and `.0115`. Repeated antecedent activation refreshes the trace, so these numbers describe an unrefreshed trace, not an absolute prediction horizon. For each traced edge, a target occurrence is:

- wellbeing: exact value inequality `target != previous_observation`;
- other inputs: target `>0` while previous target `<=0`.

On occurrence, the event delay is updated by a recency-weighted saturating rule: every existing delay value is multiplied by `1−learning`, then the current integer-age bin becomes `(1−learning)old + learning`, with `learning=association_rate × trace_strength`. The resulting values are bounded but not normalized to sum to one. The trace is then reset.

Traces are edge-local, but their update targets include every input. The separate delay dictionary does not affect action prediction or learning credit; it is observational metadata only. Thus observing a delay distribution is not equivalent to using a learned temporal expectation to time behavior. Exact wellbeing inequality is also sensitive to any numerical change, and repeated levels/continuous signals need explicit event semantics.

## 6. Consequences for five intended capabilities

1. **Conjunction:** all combinations are represented, while higher-order strengths compare their target means with a max lower-order baseline. This may admit residual conjunctions, but saturated/max aggregation and confidence-as-exposure make its selectivity unproven.
2. **Conditional NOT:** an edge can accumulate inhibition when its estimate sits below a reference. Prediction does not require a separately validated conditioned inhibitor; the benchmark action change could instead reflect correlations or deterministic tie order. Summation and retardation are not presently tested.
3. **OR/generalization:** prediction aggregates with max, not a normalized mixture or sum. It may support alternative routes, but broad convergence and independent evidence combination are not established.
4. **TIME:** traces can update delayed event metadata and edge associations, but explicit delay estimates do not enter action scoring. Variable-delay learning is currently not equivalent to temporal prediction/discharge.
5. **VALUE CHAIN:** candidate score only reads active hyperedges directly targeting wellbeing. No recursive rollout, successor representation, replay, or learned value propagation across intermediate events exists in this class. The distant-goal challenge is therefore a structural gap, not simply an insufficiently tuned trace.

## 7. Invariance and reversal implications

The learner uses integer nodes symmetrically within its declared input/action roles, but roles are not interchangeable: action vertices are omitted from observed inputs and wellbeing is a designated target for scoring. Permutation tests should preserve these roles and remap action outputs. Full invariance is further broken by ordered `argmax` tie-breaking. Reversal currently updates one shared set of edges; no regime posterior or context-indexed version of a relation exists. Old and new contingencies can average together, with adaptation governed by fixed rates. This is ordinary recency adaptation, not explicit change-point inference.

## 8. Research decisions required before redesign

- Are inputs binary events, continuous state features, or both? Which zeros are observed absences, and which are unavailable data?
- Should outcomes be event probabilities, continuous expected values, or a distribution over next observations? Define losses and calibration tests accordingly.
- Should compound selection optimize held-out conditional predictive gain with complexity control, or be a biologically motivated attentional rule? Compare rather than conflate.
- Should inhibitory relations encode signed expectation, explicit outcome absence, or a latent cause? Choose tests that distinguish them.
- Is temporal learning meant to predict event time, bridge delayed reinforcement, or both? Define distinct observables and repeated-event controls.
- Does distant-route behavior require model-based planning, a successor map, conditioned reinforcement, or simple repeated transition credit? Use reward and transition revaluation to separate these.
- What constitutes change: shifted reward distribution, transition structure, or sensor/action wiring? The current reversal test combines multiple changes; split them in diagnosis while retaining the five top-level scenarios.

These are prerequisites to principled algorithm choice. Rewriting parameters in the present score cannot resolve a representation/objective mismatch.
