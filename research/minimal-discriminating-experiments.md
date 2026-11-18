# Minimal experiments to choose graph mechanisms

## Design principle

An experiment is useful when competing mechanisms predict different observables. A benchmark that merely succeeds or fails cannot identify why. The designs below are deliberately small, synthetic and mechanism-neutral. They are not new acceptance thresholds and should not be tuned to make a candidate mechanism win. Before running an experiment, implement each candidate in the smallest form consistent with its equations, freeze all hyperparameters on separate calibration cases, and state predictions. Report held-out prediction, choice, adaptation, and resource metrics.

## 1. Does the learner need an interaction representation?

### Worlds

Use binary features `A,B` and binary outcome `Y` with balanced presentations of all four combinations. Compare:

- **Additive:** `Y=A XOR B` (no additive independent main effects can fit it exactly).
- **Conjunctive:** `Y=A AND B`.
- **Redundant:** `Y=A`; provide `B` correlated with `A` during training but independently randomized at test.
- **Context reversal:** train `Y=A` in context `C=0`, then `Y=1−A` in `C=1`, with contexts omitted in one transfer test.

All combinations must be sampled to estimate conditional relationships. For the redundancy case, split the test data by the counterbalanced `B` values; do not train only on one perfect correlation and then interpret the unseen combination as evidence for causal understanding.

### Model contrasts

1. Additive predictor with no explicit interaction.
2. Full interaction predictor / all compounds.
3. Sparse interaction selected only on training predictive gain.
4. Context-conditioned versions of 1–3.

### Predictions and metrics

- XOR: model 1 has irreducible error; models with a represented interaction can reach low Bayes risk, subject to data/model correctness.
- AND: additive model approximates but cannot represent the exact truth table as a sum of independent main effects under a fixed logistic/linear-link setup without interaction. Measure held-out log loss and calibration, not edge weight.
- Redundant cue: unrestricted conjunction can memorize training correlation; test recombination determines whether it learned conditional structure or merely co-occurrence. Penalized sparse models may reject B if it adds no conditional information.
- Context reversal: a single shared association incurs interference; context-conditioned memory can recover both mappings only when context is observed and correctly used.

**Question answered:** is graph conjunction necessary for these dependencies, and does its selection rule generalize? A win on a single conjunction alone is not enough.

## 2. What does prediction error contribute beyond recency averaging?

### Worlds

Present a cue `X` and binary outcome `Y` under: (a) stable `P(Y=1|X)`; (b) random outcomes with unchanged probability; (c) a brief outlier; (d) abrupt reversal; (e) gradual drift. Include a control cue whose marginal outcome rate matches X but is independent of it.

### Model contrasts

- Running/exponential conditional mean only.
- Prediction-error association update.
- Error-driven update with separate associability.
- Change detector or regime mixture.

### Outcomes

Use prequential log score (prediction before each sample), calibration, detection delay, false alarm rate, recovery after recurring regime, and memory/parameter count. Mean squared error after the whole sequence can conceal poor online adaptation. Random variability and actual drift must be parameter matched so a “surprise” rule cannot win merely because one stream has more extreme outcomes.

**Question answered:** does a second variable (associability/change probability) add out-of-sample explanatory value, or does a single error-driven estimate suffice?

## 3. When is an absence inhibitory?

### Training and tests

Use standard contingency phases with an outcome `US`:

- Excitor `A+`.
- Candidate inhibitor `AX−` (A predicts US except when X is present).
- Neutral control `Y` matched for exposure and sensory salience.

Then test **summation**: compound `BX` against independently trained excitor `B`, with `BY` and `B` controls. Test **retardation**: pair X with US and compare acquisition to Y under matched exposure. Add context shifts, extinction of A, and X-alone probes. Record both physiological/model prediction of `US` and behavior; behavior can be suppressed by motor competition/attention without a negative event prediction.

### Model contrasts

- Single signed associative strength.
- Separate excitation and inhibition accumulators.
- Explicit conditional outcome distribution `P(US|A,X,context)`.
- Latent context/cause memory.

### Identification caution

If X alone never predicts an observable absent outcome and is tested only as a motor cue, inhibition is not identified. A candidate passes a stronger criterion only if the learned relation transfers to B in summation and produces retardation beyond attentional/salience controls. The test is about functional inhibition; it still does not identify the neural mechanism.

**Question answered:** is conditional negative prediction required, or can the observed behavior be explained by weaker attention, retrieval, or action competition?

## 4. What kind of temporal mechanism is required?

### Worlds

Use a cue-action event at time zero and deliver an outcome after fixed lag `d`, variable lag from a known distribution, or no outcome. Cross these with (a) cue maintained vs removed; (b) an intervening irrelevant event; (c) two overlapping cues with different lags; (d) repeated vs one-off outcome; (e) same total outcome statistics but shifted hazard over time.

### Candidate mechanisms

- Exponentially decaying eligibility trace.
- A bank/basis of temporal features.
- An explicit event-time distribution/hazard prediction.
- Recurrent hidden state that learns sequence/time.

### Measurements

Measure (1) which past actions receive causal credit, (2) predicted event probability by elapsed time, (3) timing-dependent behavior, and (4) behavior after the event. These are separate outcomes. A delay histogram cannot count as timing prediction unless it improves next-event forecasts or choice. Vary lag out of sample; a trace tuned to one delay may fail at another. Compare likelihood to a base-rate event model and report false anticipation, not only correct event association.

**Question answered:** is temporal credit alone sufficient, or must the graph model event timing and event occurrence/consumption?

## 5. Does value propagation require planning?

### Worlds

Train two-step paths with reward at the terminal state. Then, without allowing reward experience from the start state:

- **Reward revaluation:** change terminal reward magnitude/value while transitions stay fixed.
- **Transition revaluation:** block an experienced transition or open a detour while terminal reward is unchanged.
- **Novel alternative:** test a route that was not traversed but uses known transitions.
- **Chain ablation:** remove or devalue an intermediate cue while holding terminal outcome fixed.

Match training exposure and graph resource budgets across candidates.

### Contrasts and predictions

- **TD(0)/local credit:** value propagates through experienced transitions over revisits; update speed should depend on experience/backup schedule.
- **Eligibility/return update:** faster credit along the experienced trajectory but no direct value for a never-experienced branch.
- **Successor representation:** quick reward revaluation with cached occupancy; transition change/detour can expose stale occupancy.
- **Learned model + search/replay:** can evaluate changed transitions and untried compositions if the model is learned accurately and computation permits.
- **Conditioned reinforcement chain:** intermediate predictive events can reinforce earlier actions, but outcome devaluation and route recombination may distinguish it from explicit model-based planning.

Track return, post-revaluation action latency, transition-model prediction accuracy, counterfactual choice accuracy, and compute. Do not call every successful chain “planning.”

**Question answered:** what is the minimum value-propagation mechanism needed for route learning, and where does its flexibility boundary lie?

## 6. Which exploration signal is useful?

### Factorial conditions

Create action choices crossing:

- familiar vs novel observation;
- epistemically uncertain vs aleatorically noisy outcome;
- safe vs harmful consequences;
- information can vs cannot alter a later decision;
- exploratory action cost low vs high.

Use matched environments so only one factor changes. Hidden deterministic mapping gives uncertainty that can be reduced; a known random generator gives irreducible noise; novelty alone does not distinguish them.

### Contrasts

- Uniform random actions.
- Novelty/familiarity count.
- Absolute prediction-error bonus.
- Epistemic uncertainty bonus.
- Expected decision value of information.

### Measurements

Measure task return and shock cost, samples to identify mapping, information actually gained, and whether the information changes later decisions. Directed information seeking should outperform noise-seeking on learnable unknown structure and should not keep revisiting a known random source. If an information bonus is independent of possible downstream utility, it can choose costly but irrelevant experiments.

**Question answered:** should the graph carry an intrinsic exploration signal, and what variable should drive it?

## 7. Is adaptive graph structure worth its capacity cost?

Run the same generated dependency family with increasing number `k` of active channels, interaction order, distractor features and distribution shifts. Compare: full powerset hyperedges; fixed-order pruning; sparse online structure selection; factorized low-rank interactions; retrieval of context-specific episodes. Fix total memory and compute budgets rather than allowing the largest representation unlimited resources.

Track held-out loss, adaptation and forgetting, edge count, update cost, and performance on recombined inputs. An architecture that wins only when unconstrained is not evidence that its structure mechanism is viable. A fixed order bound should be evaluated for failure as true dependency order exceeds it; unrestricted enumeration should be evaluated for scaling.

## 8. Does the learner itself obey channel relabeling symmetry?

Return comparisons alone are not a symmetry test: two unrelated policies may earn the same return. Construct a fixed interaction dataset `D` containing observations, physically executed actions, and outcomes. Train learner `L` on `D`; for each sampled bijection of anonymous sensory labels and action labels, transform the complete dataset and train an identically initialized `L_P` with the same defaults. At held-out histories `h`, query both policies without further updates and test

`π_{L_P}(P h) = Q π_L(h)`.

Also compare mapped next-observation predictions on every feasible action, not only greedy action. Report exact violation rate, by action-score margin. Separate clear-winner cases from ties; tie handling must itself be equivariant, e.g. random tie-breaking from a shared transformed source or a rule independent of numeric label. Repeat with a second phase of closed-loop transformed environments to test whether small decision mismatches compound into return loss.

This isolates algorithmic equivariance from interaction coverage: forced mapped actions ensure both learners see equivalent physical experience. Then the closed-loop experiment separately tests whether autonomous exploration preserves that equivalence. Neither average reward equivalence nor five arbitrary permutations is a proof of symmetry.

## 9. Decision rule for mechanism selection

A candidate mechanism earns inclusion only if all of the following are established:

1. A precise operational equation/state transition is specified.
2. A derivation explains what assumption/problem necessitates it.
3. It predicts a behavioral signature that a simpler alternative does not.
4. That signature is observed on held-out generated data under controlled comparison.
5. It improves the relevant objective without unacceptable regressions in memory, compute, safety, or old-regime retention.
6. Its role is not duplicated by another mechanism in the combined model (tested by ablation).
7. Parameters are set by units/assumptions or calibration, not per-benchmark tuning.

If these conditions fail, keep it as an open hypothesis, not a graph feature. The final implementation direction should be chosen from the smallest set of mechanisms that survives these tests.
