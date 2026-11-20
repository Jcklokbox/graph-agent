# Architecture synthesis: a sparse posterior predictive graph

## Decision

The most defensible next architecture to prototype is a **sparse posterior-predictive transition graph**. It is a unifying candidate, not a proved solution and not a biological model. It replaces the current heuristic association-strength score with one graph model of conditional transitions and outcomes; action values and information value are computed from that model, not trained in external Q/state tables.

The candidate is restricted to environments with a **sparse, compositional dependency structure** and identifiable observations. Without such a restriction, exact learning of arbitrary channel interactions and polynomial computation cannot both be guaranteed (see `first-principles-derivation.md`, R10). State this assumption rather than promise universal learning.

## 1. Representation and posterior semantics

The graph contains:

- anonymous primitive observation channels and action alternatives as interface vertices;
- learned sparse context/assembly vertices only when evidence supports an interaction;
- directed conditional factors for transitions and objective outcomes;
- local posterior/evidence state on factors and local pending-event state for delayed predictions.

For a binary target `Y` conditional on parent/context `C`, use the Beta-Bernoulli posterior:

`θ_C ~ Beta(a,b);  θ_C|D ~ Beta(a+n_+,b+n_-);  p(Y=1|C,D)=(a+n_+)/(a+b+n_++n_-)`.

This likelihood is defined for observations in `{0,1}`. Fractional `y=.5` is not a half-success/half-failure. For a continuous target, use an explicitly separate likelihood, e.g. a Normal-Inverse-Gamma model with `y|μ,σ² ~ Normal(μ,σ²)`, `μ|σ² ~ Normal(μ0,σ²/κ0)`, `σ² ~ InvGamma(α0,β0)`. Its posterior updates with `κn=κ0+n`, `μn=(κ0μ0+nȳ)/κn`, `αn=α0+n/2`, `βn=β0+M2/2+κ0n(ȳ−μ0)^2/(2κn)`. The integrated likelihood then responds to residual variance: repeated deterministic `.5` and Bernoulli `{0,1}` at rate .5 have different likelihoods. A bounded-support model (e.g. Beta with concentration) is preferable if calibrated tails near 0/1 matter; Gaussian is only a tractable reference and can assign mass outside `[0,1]`. The outcome family/type must be identified or inferred with model comparison, never by feeding fractional values into Bernoulli counts. For categorical targets use Dirichlet-multinomial. These assumptions do not automatically cover missing events or regime shifts.

Learning the graph means updating a posterior over both parameters and structures:

`p(G,θ|D) ∝ p(D|G,θ)p(θ|G)p(G)`.

The graph prior `p(G)` supplies the complexity preference. For a proposed assembly `g` versus its simpler parent graph `G0`, admit it only when its posterior odds exceed the declared threshold:

`log O(g:G0 | D) = log p(D|g,G0) - log p(D|G0) + log p(g|G0)/p(not g|G0)`.

For a binary target with independent Beta(1,1) context parameters, the integrated log evidence of one context with counts `(n1,n0)` is `log B(1+n1,1+n0)-log B(1,1)`, summed over contexts. This is a useful small-sample reference, not a universal likelihood: correlated contexts, repeated adaptive proposals and nonstationarity require an explicit model/prior. A local residual-triggered proposal still needs multiplicity control: testing many pairs raises the chance of spurious birth. Use a prior whose total birth mass is bounded per node (or an explicit proposal correction), and report that choice rather than treating per-pair Bayes factors as cost-free. Never enumerate all `2^k−1` subsets. A possible proposal is a co-active pair/context assembly after simpler factors repeatedly leave structured residuals, with recursive composition for higher-order dependencies. This only searches a restricted sparse/compositional model class; misses outside that class are a known limitation, not grounds for secretly broadening search.

## 2. How the five behaviors could arise from this one model

The unifying mechanism is the **conditional transition posterior**. The bullets below describe computations over the same graph, not claims that one scalar can have five meanings.

1. **Conjunction and parsimony.** A compound parent is admitted only if its conditional predictive evidence improves over a simpler parent set enough to outweigh its structural prior cost. AND is therefore a residual dependency, not automatic storage of every co-occurring set. Test AND and XOR truth tables plus held-out recombinations; measure predictive score and graph growth.
2. **Conditional NOT with retention.** A table containing `p(Y|A,X)` and `p(Y|A)` alone does *not* identify an inhibitor or imply transfer to a new excitor B: AX− can be stored as an indivisible exception. A transfer-capable hypothesis needs a shared signed factor, for example `logit p(Y|C)=b_Y+Σ_{e∈C} w_{eY}+Σ_{(i,j)∈C}u_{ijY}`, with shrinkage favoring reusable singleton effects over isolated exceptions. Inhibition is the negative conditional effect `w_XY<0`, established by comparing `p(Y|X,other parents)` with matched no-X contexts; summation predicts the same learned `w_XY` reduces prediction under B+X. This transfer is an explicit generalization assumption, not identified from A+/AX− alone. Require A+/AX−/neutral Y training, then held-out B+X summation and X+ versus Y+ retardation; separately report AX fit and B transfer. Context-specific interactions may be necessary if inhibition is not shared, at a structural-complexity cost.
3. **OR / route convergence.** Predictions from competing causes/routes are combined by posterior marginalization, e.g. `p(Y|D)=Σ_c p(Y|c,D)p(c|D)`, or by a calibrated factorization justified for the dependency structure. Do not use a raw maximum or assume independent causes for noisy-OR.
4. **Time, pauses and consumption.** A relation carrying a delayed event maintains a local pending-event/survival state. Its hazard is `h(τ)=P(event at age τ | no event before τ,C)`; an observed event consumes that pending prediction, while an observed omission updates survival. This is not a global step schedule. Eligibility traces may assign delayed evidence to active graph factors, but trace credit and event-time prediction are measured separately. Absence updates require complete observation.
5. **Value propagation and directed curiosity.** From the learned transition posterior, evaluate downstream objective by Bellman recursion:

   `Q(b,a)=E[r_{t+1}+γV(b_{t+1}) | b_t=b,a_t=a,D]`.

   `V` is a readout/computation on graph transitions, not a separate Q table. An initially costly step is selected only when its posterior expected future utility compensates for that cost. Exploration is value of information: expected improvement in the future decision after the observation, minus action cost—not novelty or prediction error by themselves. This computation has a horizon/model-computation cost and is not yet implemented in the prototype.

Thus one learned posterior model can support all five readouts. Distinct edge statistics (outcome counts, structure uncertainty, event survival, objective) are still mathematically different. If the experiments show that these cannot be unified in a common graph update without independent memory stores or incompatible assumptions, reject the unity hypothesis rather than rename the stores.

## 3. Equivariance and computational constraints

For sensor permutation `P` and action permutation `Q`, the target property is

`p_{P(G)}(P x'|P x,Q a,P(D)) = p_G(x'|x,a,D)`

and `π_{P(G)}(P x)=Qπ_G(x)` under mirrored tie-breaking randomness. The implementation must use set-valued parent identity, symmetric proposal scoring, and no numeric-index priority. The objective/action interface roles are explicit, but sensor identities are not.

With `k` active channels, base prediction should visit active edges, not all subsets; structure proposals should be local and only triggered by residual evidence. Pair proposals are `O(k²)` in the worst case when triggered. Higher-order growth composes selected graph nodes; it is not a guarantee of exact arbitrary Boolean-function learning. Memory is proportional to supported graph structure and observed temporal contexts. Growth, pruning and worst-case bounds must be measured.

## 4. Fast falsification sequence

Before a full environment run, test this model on deterministic/short data:

1. **Calibration vs noise:** known Bernoulli edge, balanced independent nuisance feature, prequential Brier/log loss and posterior coverage. Exposure with no predictive gain must not create a confident causal edge.
2. **Conjunction/structure:** balanced AND and XOR, a redundant cue, and held-out recombination. Compare sparse residual proposals with no interaction and full powerset oracle. The sparse model must represent supported low-order interactions and not grow on a conditionally irrelevant cue.
3. **Inhibition/memory:** train A+, AX−, neutral Y; run both summation (BX vs BY) and retardation (X+ vs Y+); switch context and test old A memory.
4. **Timing:** variable delay, omission, overlap, pause and recurrent outcomes. Test hazard calibration and post-event discharge, not merely whether a lag bin was recorded.
5. **Value path:** two-step controlled graph with immediate cost and later reward; compare selected action to exact Bellman solution; then separately alter reward and transition.
6. **Equivariance:** exhaust all sensor/action permutations in a tiny task, first with teacher-forced identical physical transitions then closed-loop. Compare predictions and mapped decisions, including ties.

Each should run in seconds, be deterministic, and include a simpler baseline. Only a candidate passing these controls should be implemented against the five scenario tests. No threshold or prior may be fitted on those five tests.

## 5. Why this is a candidate, not an established winner

This synthesis follows from the gaps recorded in the first-principles derivation: calibrated prediction requires a likelihood; sparse structure needs a complexity preference; negative relations should be conditional predictions; timing requires event/survival semantics beyond an unused lag histogram; delayed reward requires credit or downstream value; and action selection requires action support. It does **not** yet show that a local graph posterior and residual-triggered structure search can solve them together, remain computationally bounded, or satisfy the stringent permutation condition. The next step is to implement only the minimum graph kernel and run the six fast falsifiers before changing the old controller or claiming a new architecture works.
