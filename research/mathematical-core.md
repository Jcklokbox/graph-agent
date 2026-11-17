# Mathematical core

## Current formal ingredients

The prototype's `development/ops.py` represents scalar and array-valued activations in the unit interval. Its primitive operations are:

- `N(x) = 1 - x`, a complement;
- `A(x, y) = min(1, x + y)`, a saturating additive combination;
- `M(x, y) = x y`, a multiplicative combination;
- `D(x, y) = N(A(N(x), y))`, a saturating form of subtraction;
- `B(x, k)`, repeated self-combination through `A`;
- `update_Q`, combining confirmatory and falsifying evidence with separate rates.

This bounded algebra provides a compact language for graded evidence. In the current graph learner, edge records maintain quantities such as predictive strength, inhibition, value, and temporal evidence; see `development/blind_gate_graph.py` and its `q_pred`, `q_inhib`, `q_val`, and delay-query methods. The graph learner also uses exponentially weighted moving averages for state/action-conditioned value estimates.

## Questions for continuing work

The operations are implementation choices, not axioms about biology. Research can examine their algebraic properties, calibration, stability, and relationship to probabilistic or fuzzy interpretations. Useful topics include evidence accumulation under correlated inputs, normalization and saturation, learning-rate schedules, uncertainty representation, and how temporal decay or delay-specific prediction should be defined. Analyses should state their assumptions and need not be constrained to the current code's data structures.
