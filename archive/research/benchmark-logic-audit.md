# Benchmark logic audit: what the five experiments actually establish

**Method.** Read the checked-in environments, wrappers, and assertions; run `development/test_despecialized.py` in an isolated `.venv` with NumPy 2.4.6 and pytest 9.2.0. This file records facts about these particular experiments, not general biological conclusions. No external search was used. The full test command was `.venv/bin/python -m pytest -q development/test_despecialized.py`.

## Run result (this checkout)

`1 passed, 4 failed in 55.70 s`.

- Test 1 passed.
- Test 2 failed: across its five permutation/seed runs, tail values were `0.2883, 0.3117, 0.2900, 0.3017, 0.2933`; stay baseline was `0.5`; shock proportions were `0.524, 0.505, 0.511, 0.512, 0.504`.
- Test 3's variable-delay assertions passed, but its action cancellation assertion failed: safe action `0`; cancelled action `0`, expected `1`.
- Test 4 failed: mean reward on the final 300 steps was exactly `0.5`; the agent remained at the neutral baseline.
- Test 5 failed in the first phase before testing whether post-reversal adaptation succeeds: first-phase mean was `0.295`, below `0.53`.

An independent baseline check on the stock `BlindGateEnv` (10 seeds × 5,000 steps) yielded: oracle mean wellbeing `0.6109`, shock rate `0`, food rate `0.2218`; stay mean `0.5`, shock rate `0`; food-reactive mean `0.38913`, shock rate `0.22174`. The checked-in oracle uses current position plus precursor channels and reaches `0.6109`; it does not need a separate reversal detector or multi-step rollout for this stationary task. Its rule is: if already away from center, return to center; if centered with a left/right precursor, move to that side; otherwise stay centered. Under the environment's one-step transition, a center+precursor action determines the wellbeing on the next transition, while a wrong excursion without a cue yields zero. A direct transition probe of the source (set each position/precursor, suppress the next scheduled event, execute each action once) gives this next-wellbeing table:

| Current position | Precursor | left action | stay | right action |
|---|---:|---:|---:|---:|
| left | none | 0.0 | 0.0 | 0.5 |
| left | left | 1.0 | 1.0 | 0.5 |
| left | right | 0.0 | 0.0 | 0.5 |
| center | none | 0.0 | 0.5 | 0.0 |
| center | left | 1.0 | 0.5 | 0.0 |
| center | right | 0.0 | 0.5 | 1.0 |
| right | none | 0.5 | 0.0 | 0.0 |
| right | left | 0.5 | 0.0 | 0.0 |
| right | right | 0.5 | 1.0 | 1.0 |

This finite transition table shows that current position and precursor plus candidate action predict immediate wellbeing for the stationary one-step transition (noise and event timing excluded). Thus a generic one-step conditional predictor can, in principle, select beneficial actions here; this scenario does not by itself require multi-step planning. It is a derived candidate, not proof of globally optimal policy under the recurring event process. The oracle's high result is evidence that an adequate reactive policy exists and those observed variables suffice for its rule. It does **not** prove a learner can infer the rule from sparse experience, and the oracle itself is semantically hard-coded. The food-reactive policy sees food too late to control the already-completed reward transition, hence its `0.38913` score. These are finite-seed measurements, not formal estimates of all-seed expected return.

A separate 10-run permutation sweep (seeds 0–9; independent relabeling RNG seeds 2000–2009; 1,000 steps each) yielded mean final-300 wellbeing `0.28933` (sample SD `0.01658`, range `0.26333–0.31000`) and shock rate `0.51620` (sample SD `0.01116`, range `0.501–0.529`). A canonical-wiring check on seeds 0–4, 1,000 steps each, yielded `(tail, shock)` values `(0.28833,0.524), (0.27333,0.518), (0.29000,0.511), (0.30333,0.469), (0.29333,0.504)`, with mean tail `0.28967` and mean shock `0.5052`. The canonical and relabeled groups both perform much worse than stay, and their mean tail scores are close. This suggests a basic action-learning/control failure dominates the observed result; it does not show permutation invariance, which requires paired equivalent trajectories and mapped action agreement. A rough normal 95% interval for the across-run mean tail score is `[0.2791, 0.2996]`; with only ten runs this is descriptive and not a robust population guarantee. The result reproduces the qualitative failure and is far below the stay value `0.5`.

### Paired permutation experiment in progress

To separate label-permutation effects from baseline control failure, `research/permutation_equivariance_sweep.py` runs matched environment seeds with canonical wiring and one random bijective sensor/action relabeling, with a fresh default learner each time and no parameter fitting. It writes each pair immediately to `research/permutation_sweep_results.csv`. This is a robustness experiment on the current learner, not a mechanism comparison.

**Completed:** 300 matched pairs (environment seeds 0–299), 1,000 steps per run, canonical vs one random relabeling. Canonical final-300 wellbeing averaged `0.29707` (SD `.02490`); relabeled averaged `0.29522` (SD `.02009`). The paired mean difference (relabel minus canonical) was `−0.00185` (paired SD `.02788`, SE `.00161`, rough normal 95% interval `[-0.00500, 0.00130]`). Canonical shock rate averaged `.51061`; relabeled `.51594`; paired difference `+.00533` (SD `.04126`, SE `.00238`, rough normal interval `[0.00066, 0.01000]`). Both policies are far worse than stay (`0.5` wellbeing, zero shocks). There is no clear paired difference in tail wellbeing; relabeling has a small positive shock difference in this sample. The intervals are approximate and describe this seed/permutation protocol, not a universal distributional guarantee.

This result does **not** establish equivariance. Similar poor return can arise from different policies. A true symmetry test requires mapped action/prediction agreement on corresponding transformed histories (with tie behavior specified), in addition to return. The completed raw 300-pair record is `permutation_sweep_results.csv`; the deterministic sweep program and seed derivation are in `permutation_equivariance_sweep.py`.

## Scenario 1 — cue prediction, noise and conjunction

### Environment/test structure

This is an offline transition-learning test. The learner is explicitly given four repeated cases: two cue channels each predict distinct food channels, a noise channel is presented both off/on with the same wellbeing value, and a conjunction task where only cue+noise predicts wellbeing `1` while the single-feature/background cases predict `0`. Actions are fixed at index 1; there is no autonomous policy-selection problem.

### What a pass means

It establishes that, for these exact deterministic sequences and thresholds, the reported per-edge predictive strengths meet the assertions and stored scalars remain bounded. It is evidence for a restricted form of cue discrimination and a particular conjunctive association.

### What it does not mean

The test reads `predictive_strength` (a private representational statistic) instead of measuring held-out predictive accuracy or choice. Its noise criterion is an absolute strength bound; it does not establish calibrated conditional probability, causal discovery, or robustness to correlated/redundant cues. It uses one fixed cue configuration and no held-out recombination. In the conjunction half, target observations set wellbeing to `0` for the three controls and `1` for the compound; the reported learned compound strength alone does not show correct prediction under a novel test distribution.

**Deduction:** preserve the benchmark's behavioral intent, but any mechanism decision should add held-out log-loss/choice outcomes and matched factorial controls rather than interpret edge strength as truth.

## Scenario 2 — anonymous permutations + online control

### Environment structure

`BlindGateEnv` has a three-position world. Actions move one cell left/stay/right with boundary clipping. Each cue event presents one of two precursor channels; on the next step a food-side observation reflects the prior precursor. To get wellbeing `1`, the agent must be at the matching side when food arrives. Moving away from the center without getting food produces `0`; staying in center yields `0.5`. Noise is independently resampled in observation generation. Events recur after random gaps.

`RelabeledEnv` permutes the first eight sensor channels and maps three action indices to environment actions. It leaves wellbeing channel 11 fixed. The test averages late wellbeing and caps all-run shocks; it compares with stay.

### What the result says

The present controller is not learning an effective policy on these five runs: its late mean is far below the stable center action and shocks occur on about half of steps. Since the oracle uses only current observed position/precursor cues and obtains `0.6109` on a separate 10-seed check, the environment is not intrinsically impossible. It does not follow that the learner fails specifically because of missing change-point inference or planning; this task can be solved with the right one-step cue/action model, so investigate representation, transition timing, credit, exploration, and action symmetry first.

The learner has `argmax` tie-breaking by action list order. Direct source calculation: before any edge has confidence, each candidate's action-containing edges have confidence `0`, so local novelty is `1`; background wellbeing prediction is `0.5`; score is `0.5 + 1×(1−0.5)=1` for every action. `numpy.argmax` therefore selects candidate index 0, irrespective of physical semantics. Under a consistent action permutation, an initially tied score can select different physical actions. This is a concrete symmetry risk and a lack of deliberate initial coverage. It can be isolated by applying permutations to identical trained graph state and evaluating mapped choices before learning.

## Scenario 3 — cancellation and variable delay are different experiments

### Cancellation environment

A precursor event occurs periodically; at that step a blocker bit is sampled with probability 0.5. On the following step, food is present only if the prior precursor was present and blocker absent. The code updates wellbeing to 1 only if the agent is at the left side for an uncancelled precursor; otherwise moving to a side often yields 0 and staying center yields 0.5. The test learns autonomously for 900 steps, then queries action at two hand-constructed observations: center+precursor+blocker-off and center+precursor+blocker-on. A five-seed probe with the same protocol (environment seeds 0–4) produced `(safe, blocked)` actions `(2,0), (2,2), (0,1), (2,0), (2,2)`; only one of five matched expected `(0,1)`. Official seed 31 yielded `(0,0)`. The action contrast is not robust under this protocol; this still does not identify one uniquely missing inhibitory mechanism.

### Logical limitation

These two observations test action discrimination, not conditioned inhibition in the standard sense. No independent excitor is trained and combined with a separately trained candidate inhibitor at a summation test; there is no retardation comparison. The blocker occurs with the precursor at the same time, so it also changes which event follows. A failure to choose different actions might reflect weak predictive learning, insufficient behavioral evidence, feature/target timing, score ties, or a wrong action consequence model. A success alone would not isolate inhibitory learning.

### Variable-delay environment

A cue is followed by food after random 2, 3, or 4 steps; agent action is ignored and wellbeing stays fixed at 0.5. The test checks bins in `observed_delay_distribution(3,5)`, but this dictionary does not contribute to action scoring in the learner. Hence passing these assertions establishes only that a set of recorded delay values exceeds thresholds. It does **not** establish that the learner predicts a future event, uses timing to choose action, or discharges an expectation on event occurrence.

**Deduction:** cancellation and temporal prediction should remain separate claims; current combined test cannot support a single “NOT+TIME works” conclusion.

## Scenario 4 — delayed distant goal

`DistantGateEnv` starts at position 2. A cue is observed when the agent is at 2; a food opportunity is scheduled two time steps later and is visible/rewarded only if the agent is then at position 0. Position 1 yields `0.45`, position 2 yields `0.5`, and position 0 yields `0` unless food is available. Thus the useful route accepts an immediate loss at the intermediate location in anticipation of a future reward.

Observed final-300-step score is `0.5`—no improvement over remaining at position 2—on the official seed. A five-seed probe (0–4), each 1,200 steps, yielded final-300 means exactly `0.5` for all five; the all-step mean was `0.496167` and shock rate `0.006667` for each. This makes the neutral-stay behavior reproducible on these seeds and horizon, but does not estimate all possible environment sequences. It directly confirms no successful value propagation on these runs. It does not prove that the only solution is generic planning: a learned cue-conditioned chain or temporal predictive controller might suffice. Separate tests must establish whether cue-to-intermediate, intermediate-to-food and action effects are learned and whether outcome value propagates across them.

**Diagnostic derivation:** on the cue step at position 2, moving left gives immediate `.45`; staying gives `.5`. If the agent moves left again, it reaches position 0 on the food step two transitions after the cue and receives `1.0`. Over these two transitions, the route return is `.45 + γ·1`, while staying yields `.5 + γ·.5` if no new event intervenes. The route is preferred by this two-step comparison exactly when `γ > .1`. A one-step wellbeing maximizer instead rejects the first move because `.45 < .5`. The cue disappears at the next observation, so the agent must retain its significance through a state/trace/sequence representation. This establishes the benchmark's requirement for non-myopic consequence or learned temporal credit; rollout, value propagation and conditioned cue chains remain competing implementations.

## Scenario 5 — compound reversal and prerequisite failure

The test trains for 600 steps, reverses both action mapping (swapping left/right motor interpretation) and sensor mapping of two cue channels, then measures performance in first-phase steps 400–599 and post-reversal steps 900–1199. It asserts both phases exceed `0.53`.

Measured first phase is `0.295`, so the test provides no evidence about post-reversal adaptation. A 10-seed/permutation pre-phase replication (seeds 0–9, permutation RNG seeds 7700–7709, same 600-step training, measuring steps 400–599) gave means `[0.2675, 0.2925, 0.2625, 0.2550, 0.2825, 0.3000, 0.2975, 0.2750, 0.2550, 0.2950]`; average `0.27825`, sample SD `0.01764`, range `0.255–0.300`, and zero of ten runs exceeded `0.53`. Thus the missing precondition is not just an unlucky seed in the original assertion. It cannot validly be summarized as “reversal fails” until competent pre-change behavior is obtained. Simultaneously changing action and cue wiring also makes it impossible to attribute any later failure to cue remapping versus motor remapping.

**Deduction:** treat this as a two-stage acceptance test. First establish pre-change competence; then report immediate post-change regret and eventual recovery against the pre-change competent level. A separate sensor-only, action-only, and combined-change decomposition is needed for diagnosis, without encoding those cases as learner rules.

## Cross-test conclusions justified by current evidence

1. The current code passes one deterministic association/conjunction exercise but fails four out of five test functions in this run.
2. The current data do **not** support the claim that the graph is broadly adaptive. It also does not identify one missing mechanism as the unique cause.
3. Some benchmarks overstate what their assertions establish: test 3 delay bins are metadata-only; cancellation lacks classical inhibitor controls; test 5 fails its pre-reversal competence precondition.
4. Test 4 specifically requires evaluating consequences beyond one step or an equivalent learned temporal chain. This is a task-structure fact, not proof of a biological mechanism.
5. The 300-pair matched sweep shows canonical and relabeled control both fail similarly; in this protocol, relabeling is not the main cause of low return. This does not prove symmetry because mapped decisions were not compared.
6. The most informative next action is not tune thresholds. It is make minimal, separable experiments: (i) one-step transition prediction/action learning; (ii) delay credit vs event-time prediction; (iii) route value propagation vs planning; (iv) a true pre-trained reversal with isolated change dimensions; (v) direct permutation equivariance under ties and after equivalent training.

## Provisional mechanism priority from the benchmark logic

This is a dependency ordering for investigation, not a final architecture recommendation.

| Priority | Capability to establish | Why it comes first | Evidence that would justify moving on |
|---|---|---|---|
| P0 | Correct action-conditioned one-step prediction and safe action coverage | Tests 2, 3, and 5 fail before deeper explanations can be identified; current canonical agent also fails, so permutation is not the leading diagnosis. | Held-out transition predictions and policy beat stay with low shock on canonical and relabeled mappings. |
| P1 | Interaction/conditional prediction and feature selection | Scenario 1 indicates this representational capability, but only with edge-strength measures. | Held-out factorial/recombination tests and proper prediction metrics. |
| P2 | Temporal credit and event-time prediction as separate capacities | Existing delay-bin check measures metadata only; cannot support a claim of behaviorally useful timing. | Lag generalization improves next-event calibration or decision behavior, including omission/repeated-event controls. |
| P3 | Multi-step value propagation / planning boundary | Scenario 4 requires accepting immediate loss; candidate mechanisms are not yet distinguished. | Route success plus reward revaluation and transition revaluation to separate TD, SR, chaining and model-based planning. |
| P4 | Regime retention/change adaptation | Scenario 5 currently fails the pre-reversal competence prerequisite. | Establish competent baseline first, then compare isolated reversal/drift/recurrence with regret and retention measures. |
| Cross-cutting | Symmetry, uncertainty and graph resource control | These affect every learned mechanism; current score ties have label order and all-subset enumeration is exponential. | Paired permutation equivariance and matched-capacity/model-selection tests. |

The evidence suggests a sensible research order: make a basic controller learn action-conditioned consequences and obtain adequate support; then characterize whether the specific temporal and multi-step tests require additional mechanisms; only then measure reversal adaptation. This order does not preclude a unified eventual mechanism. It prevents attributing poor behavior to sophisticated missing machinery when a prerequisite action model has not yet been demonstrated.

## Reproducibility note

The isolated `.venv` was created locally for test execution and then removed. No source code or test assertions were changed for these measurements. Exact run output and seeds are in the command/test definitions; rerun from the repository root using:

```bash
.venv/bin/python -m pytest -q development/test_despecialized.py
```
