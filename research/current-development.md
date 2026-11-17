# Current Development: Research Map

This note describes the checked-in prototype as inspected on 2026-11-17. It is a map of current behavior, not a specification or endorsement of every design choice. Recheck the linked source when using this document after code changes.

## System overview

The prototype comprises a synthetic environment, a graph-oriented online learner/controller, supporting bounded operations, and tests. Its task has three positions (left, center, right), delayed precursor/food cues, occasional binary noise, and a wellbeing observation. The environment has an oracle policy and simpler baselines for comparison. It is a compact testbed for prediction across time and action-dependent consequences, not evidence of general animal-like learning by itself.

## Implemented components

### Environment — `development/blind_gate.py`

`BlindGateEnv` emits a 12-element observation. Indices 0–2 encode position, 3–4 precursor cues, 5–6 food cues, 7 noise, 8–10 are unused by the environment (reserved for motor nodes in the learner's shared index space), and 11 is wellbeing `W`. The environment internally schedules events and produces reward/wellbeing values 0, 0.5, or 1. A transition returns observation, wellbeing, done, and a `StepInfo` record. The environment's semantics are explicit; the learner's code is intended not to rely on the semantic meaning of individual sensor indices.

### Bounded operations — `development/ops.py`

The module implements `N(x)=1-x`, saturating `A(x,y)=min(1,x+y)`, `M(x,y)=xy`, `D(x,y)=N(A(N(x),y))`, repeated self-combination `B`, and a confirm/falsify helper `update_Q`. These functions accept NumPy values/arrays. They establish a useful bounded vocabulary but are not all used as a single formal calculus by the learner; in particular, the graph learner also contains explicit arithmetic updates and thresholds.

### Graph controller — `development/blind_gate_graph.py`

`BlindGateGraphBrain` uses 12 node slots: 0–7 for environment inputs, 8–10 for candidate actions, and 11 for wellbeing. `forward`/`select_action` select among actions; `learn(next_observation)` learns the transition after a preceding `forward` call. The agent validates observations in `[0,1]` and rejects environment feedback on action-node slots.

The learner forms antecedents up to order three, subject to activity thresholds and additional restrictions on sensor-only third-order combinations. A source combines current observed sensor activity with projected trace activity and the selected action. Per-edge records keep counts, totals, EMA and blended outcome estimates, positive predictive strength (`q`), inhibitory strength (`q_minus`), lag-bin statistics, and traceability/index flags. Base-rate estimates provide a reference for determining whether an edge predicts beyond background. Higher-order updates consult lower-order expectations.

Temporal processing maintains edge-local three-bin traces, decays active traces, indexes them by target, records approximate lag categories, and discharges a trace when its predicted target occurs. A discharged edge is prevented from firing again until its antecedent deactivates. These are specific prototype mechanics; they should be evaluated against the temporal-trace and expectation-discharge hypotheses rather than presumed biologically correct.

Action evaluation predicts target inputs from active graph relations and scores actions with wellbeing-related estimates. The code also maintains auxiliary data: base-rate vectors, trace projections, indexes, compatibility pools/snapshots, `q_val_stats`, `vessel_q`, and value-related structures including `solo_w`/`pair_w`. This is an important architectural fact: the ideal of a single graph as all learned memory is not yet realized. Some structures are projections or indexes; others may contribute independently to decisions and merit explicit ablation or unification research.

Compatibility APIs include prediction/inhibition/value queries, expected-delay queries, trace prediction, and snapshots. Legacy virtual handles support older nested chunk queries; they are compatibility mechanisms, not literal causal vertices.

### Supporting pool — `development/brain_p2.py`

`DynamicQPool` is a lightweight keyed store and statistics container. In the current graph learner it serves compatibility/indexed views; do not infer that it is itself the complete learning architecture.

## Existing evidence and test map

- `test_blind_gate_graph.py` covers controlled transitions, prediction/inhibition/value query behavior, and chunk/legacy compatibility assumptions.
- `test_autonomous_and_generalization.py` defines task variants including cancellation, variable delay, and distant gates, and tests autonomous behavior/generalization.
- `test_despecialized.py` combines code inspection and behavioral checks: index-role avoidance, controlled associations/noise, permuted autonomous play, cancellation and delay, distant-gate behavior, and reversal.
- JSON files capture task/benchmark-related records; inspect their fields and generation history before treating them as authoritative results.

The existence of a test does not mean the corresponding principle is satisfied. Tests may be stochastic and should be run with their reported parameters and seeds. Results in an older README are historical, not a current validation guarantee. The full suite may be expensive, and NumPy/pytest are runtime requirements for relevant commands.

## Research gaps suggested by the code

1. **Informative association and habituation:** compare edge updates with true baseline-relative prediction error; test whether repeated independent noise decays in influence without relying on a hand-selected noise index.
2. **Blocking and conjunction formation:** test whether existing reliable predictors block redundant compounds, and whether compounds form when individual cues fail. Track both false-positive and missed compound links.
3. **Inhibition and reversal:** determine whether inhibition is context-specific and preserves positive memory; measure reversal speed, retention, and renewal under changed contexts.
4. **Temporal credit assignment:** quantify delay resolution and trace decay across variable intervals, cue persistence, multiple outcomes, and repeated outcomes. Validate discharge behavior independently.
5. **Novelty and exploration:** current action selection and exploration should be inspected directly; design tests that distinguish local novelty-driven orienting from randomness, fixed priors, and elapsed-step schedules.
6. **Action permutation/parity:** current action slots have fixed internal ordering. Test equivariance under action relabeling, with output mapping, and document any residual ordering effects.
7. **Value propagation and architectural unity:** inspect the exact action-scoring path; ablate or unify value side structures; test delayed reward chains and immediate-cost/future-value tradeoffs.
8. **Calibration and locality:** distinguish edge probability-like `mu`, confidence-like `q`, inhibition, lag proportions, and action scores; test calibration, boundedness, update locality, and sensitivity to thresholds/rates.

These gaps are research opportunities, not a mandatory implementation roadmap. Future work may revise the theory or the benchmark if evidence shows the current framing is inadequate.
