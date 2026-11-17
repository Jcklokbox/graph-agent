# Mathematical core

This note maps the mathematical ideas currently visible in `development/` and identifies where the formal theory remains incomplete. The code is not a proof that the desired mathematical properties hold.

## Bounded operation vocabulary

`development/ops.py` defines scalar- and array-valued operations intended for values in `[0,1]`:

- `N(x) = 1 - x`: complement;
- `A(x,y) = min(1, x+y)`: saturating addition;
- `M(x,y) = xy`: multiplication;
- `D(x,y) = N(A(N(x),y)) = max(0,x-y)`: bounded difference;
- `B(x,k)`: repeated self-combination using `A`;
- `update_Q`: a confirm/falsify update helper built from bounded operations.

The first four are a small closed operation set for bounded quantities. `B` and `update_Q` are helpers built around it. These algebraic primitives are not the same thing as the five functional capabilities (AND, NOT, OR, TIME, VALUE CHAIN) described in [the foundational principles](foundational-principles.md); a derivation connecting them is still needed.

## Quantities in the current learner

The graph implementation (`development/blind_gate_graph.py`) stores per-antecedent/target edge records. Among other fields, these include:

- `n`, `total`, and `ema`: sample count, accumulated target activity, and an exponentially weighted activity estimate;
- `mu`: a blend of empirical frequency and EMA, used as an outcome estimate/reference;
- `q`: positive predictive association strength, updated with separate positive and decay rates according to improvement over a reference;
- `q_minus`: inhibitory strength, updated from selected unexpected absences and reduced on target activity;
- `lag`: normalized proportions over three approximate delay categories, with accompanying counts;
- edge-local trace intensity: a short/mid/slow activity record with decay and target-triggered discharge.

The exact meanings and calibration of these quantities are implementation-specific. In particular, `q` is not automatically a calibrated conditional probability, and `mu`, `q`, and action scores should not be treated as interchangeable. The learner uses baseline-rate estimates and lower-order edge estimates as references for compound edges. It also updates values and action scores through additional structures; see the [development map](current-development.md).

## Intended mathematical constraints

The research direction favors quantities bounded in `[0,1]`, transparent update equations, and local learning based on an edge's active antecedents and target, with explicit reference statistics. This does not mean all current code updates literally use only `N`, `A`, `M`, and `D`: source contains direct arithmetic, clipping, thresholds, and parameterized EMA formulas. Record these accurately when deriving or auditing the model.

For each quantity or equation, specify:

1. domain and intended interpretation (probability, expectation, evidence strength, inhibition, trace, or score);
2. observations used in its update and the locality boundary;
3. initialization, saturation, decay, and parameter units;
4. whether it is calibrated and how calibration is evaluated;
5. interactions with correlated predictors, absent events, repeated outcomes, and nonstationarity.

## Open formal questions

- Under what assumptions does baseline-relative strengthening implement an informative-association or blocking effect?
- How should evidence for a target's absence be represented when observations are sparse or partially observed?
- Can positive and inhibitory associations coexist without erasure, and what prevents order-dependent interference?
- What temporal kernel is implied by the current trace cascade, and how does it behave across delays outside the three bins?
- Does saturating accumulation preserve useful evidence under correlated or redundant inputs?
- Which numerical parameters have a physical/statistical interpretation, and which are provisional engineering choices?
- How can local edge updates produce stable long-horizon value propagation without an unacknowledged global optimizer or task-specific table?

Derivations and experiments should make assumptions explicit and may propose a different bounded algebra if it explains data better.
