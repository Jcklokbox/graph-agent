# Algorithmic core

This note summarizes the present computational approach and frames it as an evolving hypothesis. For implementation details and known discrepancies from the research ideal, see [Current Development: Research Map](current-development.md).

## Present loop

At a high level, the agent receives an observation, evaluates possible motor events, acts, observes the next input, and updates learned predictive relations. In the Python API, `BlindGateGraphBrain.forward(observation, action=None)` both selects/records the action and retains a pending transition; `learn(next_observation)` consumes the next observation and updates the learner. Call `forward` before `learn` for each transition.

The shared node space reserves environmental inputs, candidate action nodes, and a wellbeing node `W`. For each candidate action the source representation combines active observations, eligible traces, and that action's motor node. Active antecedent sets are enumerated up to order three, subject to thresholds and constraints. Learned edges predict target activity. Base rates and lower-order predictions provide references for updating higher-order associations. Candidate actions are scored using predicted consequences and wellbeing-related estimates.

Temporal association is edge-specific: traces are created for learned/traceable antecedents, decayed over steps, accumulated in coarse age bands when a target appears, and discharged on target occurrence. A discharged relation waits for antecedent deactivation before reactivation. This is one concrete temporal mechanism; its sufficiency, biological meaning, and robustness remain open.

## Intended principles for algorithm design

- Learn predictive structure from interaction rather than encode semantic rules for named sensors or task scenarios.
- Let compound antecedents earn support beyond what lower-order predictors explain; track complexity and avoid redundant rules.
- Preserve positive expectations while learning conditional inhibitory relations.
- Use local novelty/uncertainty to guide orienting and exploration, and outcome evidence to update route selection.
- Evaluate actions through predicted consequences toward wellbeing, and investigate whether intermediate predictions can carry value across delay.
- Keep causal learned memory conceptually unified. Indexes and caches can be implementation details, but independent tables that drive behavior should be documented and tested against a graph-based alternative.

These points are research goals. Current code includes explicit auxiliary state and some hard-coded structural conventions (e.g. node ranges, a designated `value_node`, fixed max antecedent order, thresholds, and decay constants). Index-agnostic sensor handling is not equivalent to full architectural universality.

## Experiments and evaluation

A good experiment isolates one mechanism before combining mechanisms. Use controlled transition sequences for causal attribution; then evaluate autonomous behavior and generalization in held-out seeds, random sensor permutations, and action permutations. Include negative controls (uninformative cue, redundant predictor, absent blocker, irrelevant noise), delays and reversals, and report both performance and mechanism-level measures. Distinguish source inspection from behavioral evidence. Keep stochastic seeds and environment settings alongside results.

The existing tests cover controlled graph links, cancellation/inhibition, temporal delays, autonomous behavior, distant gates, and reversal. Some criteria may fail; treat failures as evidence for diagnosis, not a prompt for test-specific code. The [foundational principles](foundational-principles.md) provide the project's scientific guardrails.

## Open algorithmic questions

How should novelty be computed without semantic indices? How should sparse higher-order structure be selected from noisy observations? How should confidence, expectation, and inhibition combine during planning? What planning depth is required for delayed value chains, and can it emerge from local prediction rather than explicit lookahead tables? What notion of invariance should hold under channel permutation, including random tie-breaking and exploration? These remain research questions, not settled features.
