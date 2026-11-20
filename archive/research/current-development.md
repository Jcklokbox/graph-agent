# Current Development: Research Map

This note describes the checked-in implementation and its known limits. It is a guide for research, not a claim that the model fulfills every foundational principle. Recheck source when details matter.

## System and topology

`development/blind_gate.py` defines a synthetic three-position delayed-food task with precursor/food cues, noise, and wellbeing. `development/benchmark_envs.py` defines cancellation, variable-delay, and distant-goal variants.

`BlindGateGraphBrain` stores graph-node activity in a vector. The default topology reserves nodes 8–10 for candidate motor events and node 11 for wellbeing, matching `BlindGateEnv`. Observation width, action-node IDs, and wellbeing-node ID are configurable. All non-action input nodes, wellbeing included, enter the same antecedent and learning machinery. The learner must still be told which nodes are outputs and which input represents wellbeing; that is an explicit controller/environment interface, not a sensor-role rule.

## Learning and action selection

`forward(observation, action)` creates a source from current inputs and one internally activated motor node. It enumerates nonempty subsets of positive source activity, with no fixed antecedent-order cap. `learn(next_observation)` updates bounded expectations, confidence, separate positive/inhibitory strengths, and local eligibility using the bounded operations in `ops.py`. Each compound's reference is the maximum of background expectation and available lower-order expectations. The empty-antecedent background and learned relations occupy the same hyperedge store. Edge-local traces update only their own target; target transitions record recency-weighted delay probabilities before discharge.

Action scoring combines active wellbeing excitation and inhibition against the background. It also uses local novelty: low confidence in action-conditioned edges leaves more optimism toward wellbeing's upper bound. This is deterministic and local, not random exploration or a step-count schedule. Its calibration and biological interpretation are open questions.

The learner does not contain a task-specific policy, virtual chunk vertices, Q-value tables, mirrored pool stores, fixed two-step rollout, legacy compatibility layer, or arbitrary fallback action bonus. `brain_p2.py` and unused historical JSON snapshots were removed with those structures.

## Current tests and evidence

`development/test_despecialized.py` contains five behavioral experiments aligned with the research hypotheses:

1. informative cue prediction, uninformative noise, and conjunctions when single cues are insufficient;
2. autonomous learning under random consistent sensor/action permutations, compared with a neutral baseline;
3. cancellation by a blocker and event prediction across variable delays;
4. pursuit of a delayed goal requiring a distant route;
5. reversal after the sensor/action wiring changes without resetting the learner.

The tests target phenomena, not internal layout or private statistics. On the current prototype, test 1 and the variable-delay subcheck in test 3 pass; the autonomous-permutation, cancellation-action, distant-goal, and reversal criteria fail. The five permuted runs averaged wellbeing 0.297 over their final 300 steps versus a stay baseline of 0.5, with shock rates from 0.504 to 0.524. In the distant-goal task the agent stayed at the neutral 0.5 level; before reversal its mean was 0.295. These failures are direct evidence that the current choice rule is not yet adaptive enough, not justification for task-specific patches.

## Known research risks

- **Combinatorics:** enumerating every active subset scales exponentially with the number of positive inputs. Avoid restoring a fixed order cap as a shortcut; test principled sparsity or complexity control.
- **Compound parsimony:** max lower-order reference is a simple blocking heuristic, not a full Rescorla–Wagner or latent-cause theory. The compound benchmark tests only one small case.
- **Inhibition:** excitation and inhibition have separate bounded strengths, but context-sensitive Pavlovian inhibition, memory preservation, renewal, and reversal are not yet robustly demonstrated.
- **Time:** traces use an explicit exponential decay parameter. Variable delay is measured behaviorally through event timing; multiple outcomes, overlapping traces, and discharge need stronger experiments.
- **Novelty:** local evidence drives an optimistic action score, but animal-like orienting, rapid danger avoidance, and success reinforcement are not established.
- **Value chains:** current scoring is a direct wellbeing prediction over active hyperedges. Deep secondary-reinforcement chains or general planning are not implemented; the distant-goal test intentionally probes this gap.
- **Evidence quality:** bounded expectations/confidence are not automatically calibrated probabilities or causal effects. Report seeds, controls, and failures.

For project-level scientific commitments and how agents should reason from them, see [Foundational Research Principles](foundational-principles.md).
