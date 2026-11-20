# Mathematical core

This note maps the current model's mathematical quantities and identifies what remains unproved. The implementation is not a proof that the desired mathematical or biological properties hold.

## Bounded operation vocabulary

`development/ops.py` defines scalar- and array-valued operations intended for `[0,1]`:

- `N(x) = 1 - x`: complement;
- `A(x,y) = min(1, x+y)`: saturating addition;
- `M(x,y) = xy`: multiplication;
- `D(x,y) = N(A(N(x),y)) = max(0,x-y)`: bounded difference;
- `update_Q`: a confirm/falsify update helper built from bounded operations.

`update_Q` is a helper; these algebraic primitives are not the same level of description as the five functional capabilities (AND, NOT, OR, TIME, VALUE CHAIN) in [Foundational Research Principles](foundational-principles.md). A derivation relating the levels remains open.

## Quantities stored on graph edges

The learner stores learned state in one hyperedge map. A key is `(antecedent_nodes, target_node)`; the empty antecedent `()` holds the target's background expectation in the same graph. Edge records contain:

- `expectation`: an exponentially updated target activity estimate;
- `confidence`: bounded accumulated evidence for the relation;
- `positive` and `inhibition`: separate bounded strengths, allowing excitation and suppression to coexist;
- `trace` and `age`: eligibility intensity in `[0,1]` and elapsed integer steps;
- `delays`: recency-weighted event-delay probabilities whose values remain in `[0,1]`.

The wellbeing estimate combines the background expectation with active excitation and inhibition using bounded `A`, `M`, `N`, and `D`. Novelty is derived from low local confidence and gives candidate actions optimistic wellbeing only within the unresolved range. All learned activation-like quantities are bounded; discrete node identifiers and elapsed step age are indices, not activation variables.

The current expectation, confidence, association, and delay distribution are model estimates, not automatically calibrated causal probabilities. Initialization and default update rates remain hypotheses. Constructor rates are explicit and bounded (`expectation_rate`, `association_rate`, `extinction_rate`, and `trace_decay`) rather than hidden inside per-scenario branches.

## Locality and remaining mathematical questions

Updates are made per active antecedent/target edge or its local trace. The empty-antecedent background is also represented as a graph edge. Action selection aggregates only active relations. The graph does enumerate every active subset; this preserves the possibility of higher-order conjunctions but creates exponential computational cost. A principled complexity mechanism is an open problem, not grounds for an arbitrary fixed order cap.

For each quantity or equation, specify:

1. domain and meaning (expectation, confidence, excitation, inhibition, trace, delay probability, or action score);
2. observations used in its update and the locality boundary;
3. initialization, decay, saturation, and parameter interpretation;
4. calibration and uncertainty, if probability-like language is used;
5. behavior under correlated predictors, absent events, repeated outcomes, and nonstationarity.

Open questions include whether max lower-order reference produces informative association and blocking; how absent events should be represented under sparse observation; how independent excitatory and inhibitory strengths behave under reversal; what temporal kernel follows from trace decay; whether saturating combination overcounts correlated predictors; and whether local novelty is a defensible uncertainty estimate. Deep value propagation is not implemented.

Derivations and experiments should make assumptions explicit and may propose different bounded operations if they better explain evidence.
