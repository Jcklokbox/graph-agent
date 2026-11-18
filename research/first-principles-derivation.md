# First-principles derivation: requirements on an adaptive graph

## Scope

This note derives constraints from the task definition. It does **not** claim that the derived requirements uniquely determine an algorithm, that a biological system implements them, or that they are already satisfied by the prototype. No literature analogy is used as a premise here. The question is: what must be true of any graph-based learner that is expected to adapt across unknown environments without scenario-specific rules?

## 1. Formal problem

At step `t`, the learner has interaction history

`h_t = (o_0,a_0,r_1,o_1,...,a_{t-1},r_t,o_t)`

and chooses an action `a_t` from available actions `A`. The environment is described by an unknown parameter/regime `θ_t` and a possibly hidden state `s_t`:

`(s_{t+1}, o_{t+1}, r_{t+1}) ~ P_{θ_t}(· | s_t,a_t)`.

The learner is not given semantic meanings for observation channels. Its graph, internal state, and action rule are functions of history and declared interface roles. A policy is a mapping from the learner's internal state and current observation to an action. The objective may be finite-horizon cumulative wellbeing, discounted return, or another specified criterion; these objectives are not interchangeable, so the environment must say which one is measured.

**Identifiability boundary.** If two environment histories are identical from the learner's available observations and actions, but require different actions for optimal performance, no learner can always choose correctly on both. It must receive further information, use a prior that favors one environment, or accept error. This follows because any algorithm sees the same input and therefore has the same action distribution in both cases. “Self-adapting” cannot mean inference beyond available evidence.

## 2. Derived requirements

### R1 — Internal representation must preserve decision-relevant distinctions

Suppose the learner maps histories to internal state `z_t = φ(h_t)`. If `φ(h)=φ(h')`, then a policy using only `z` and the current observation cannot distinguish those histories. Therefore, if the optimal action sets differ, `A*(h) ∩ A*(h') = ∅`, any such policy must be suboptimal for at least one history.

**Consequence.** A graph representation must either (a) retain sufficient predictive distinctions, or (b) retain uncertainty over distinct possible latent situations. Compressing histories is valid only when merged histories are equivalent for future prediction and action value. This is a task-relative criterion, not a demand for a complete world model.

### R2 — Learning needs an explicit target and error criterion

A graph state changes only according to an update rule. For that rule to be called learning, it must be tied to a measurable discrepancy between prediction and observation, or to an explicitly stated objective (e.g. return). Otherwise arbitrary internal change cannot be distinguished from improvement.

For a predicted conditional distribution `q_ψ(y|x)` and observed target `y`, a generic update can minimize a proper scoring loss `ℓ(q,y)`. In the Bernoulli case with predicted probability `p`, log loss is

`ℓ(p,y) = −y log p − (1−y) log(1−p)`.

For `p=σ(u)`, `∂ℓ/∂u = p−y`; thus a gradient step changes the prediction in the direction of the signed residual. This is a derivation for this likelihood/link, not a claim that all graph edges should have this update or that wellbeing is Bernoulli. Continuous, categorical, censored, and missing outcomes require different observation models.

**Consequence.** Edge weights, confidence, activity, and value cannot be treated as synonyms. A quantity needs a definition from an update and a test of its intended meaning. In particular, repeated exposure alone is not evidence of predictive reliability if outcomes conflict.

### R3 — General relational learning requires a representation of interactions

If the target depends on two inputs jointly but neither input alone predicts it, a purely additive predictor `f(x_1,x_2)=g_1(x_1)+g_2(x_2)` cannot represent all such functions. For the binary XOR relation, changing either input while holding the other fixed reverses the outcome; no additive independent effects reproduce XOR exactly. The learner must add an interaction term, transform its representation, or infer a latent state that makes the mapping simple.

**Consequence.** AND-like capability is not a decorative feature; it is one possible means of representing non-additive dependence. But storing every conjunction is not implied. A compound is warranted only if the chosen objective rewards its predictive/decision benefit enough to justify representation cost. Which complexity criterion to use remains open.

### R4 — Reliable action choice requires comparing consequences, not just cue strength

For a fully specified finite horizon, the value of an action is the expectation of future objective under that action and subsequent policy:

`Q_t(h,a) = E[Σ_{k=0}^{H−1} r_{t+k+1} | h_t=h, a_t=a, π, environment]`.

Choosing `argmax_a Q_t(h,a)` is optimal for that horizon if the conditional expectations and feasible actions are correct. A prediction of a cue is not itself a reason to select the action that predicts it: the cue matters only insofar as its downstream consequences affect the objective or future information. Thus a graph intended to choose actions must connect predictions to objective-relevant consequences, directly or through a computationally equivalent value calculation.

**Consequence.** A learned cue relation, a value estimate, and an action policy are distinct mathematical objects even if implemented in one graph. A direct edge-to-wellbeing score may solve shallow cases; it cannot generally substitute for multi-step consequences.

### R5 — Delayed credit and planning solve different problems

Consider two actions `a` and `b` taken in otherwise identical states; only `a` leads, after delay `d`, to a reward. To learn the action's predictive/value relation from that reward, the learner needs some retained information linking the earlier choice to the later observation. This information can be an eligibility trace, explicit episode memory, a state-transition model, or another statistic; if all traces of the choice are erased before feedback, the later reward provides no evidence that distinguishes `a` from `b`.

But assigning credit to the experienced `a` does not determine the value of an untried alternative `c`. To choose `c` without trying it, the learner also needs a compositional transition/consequence model or stored counterfactual information. This follows because two worlds can agree on the experienced trajectory but differ on the consequences of `c`.

**Consequence.** A trace mechanism can be necessary for delayed learning under some observation schedules, but it is not by itself planning. The distant-goal capability must be tested for both experienced-route credit and unexperienced-route evaluation.

### R6 — Reversal and retention impose a memory trade-off

Let one relation's target mean be `μ_A` in regime A and `μ_B` in regime B. A constant-rate estimate `m_{t+1}=(1−α)m_t+αy_t` adapts after B begins as

`E[m_n−μ_B] = (1−α)^n(m_0−μ_B)`

under independent observations with mean `μ_B`. Larger `α` shortens the adaptation time; it also gives more influence to each noisy sample. Under recurring A/B regimes, a single estimate converges toward a history-weighted mixture rather than preserving both conditional predictions. No choice of one fixed `α` simultaneously provides instantaneous switching, noise immunity, and perfect retention for all regime schedules.

**Consequence.** Fast plasticity and preserved old knowledge require either accepting this trade-off, inferring context/regime and conditioning retrieval, storing multiple hypotheses/episodes, or another memory organization. These are alternatives with costs, not an automatic mandate for a Bayesian change-point module. A single reversal score cannot establish which solution is needed.

### R7 — Surprise, uncertainty, and information value are distinct

Prediction error `|y−ŷ|` measures a realized mismatch. Predictive uncertainty measures spread over possible `y` under the current model. Epistemic uncertainty is uncertainty about model parameters/structure; aleatoric uncertainty is irreducible outcome variation. These can diverge: an unpredictable but known coin has high outcome entropy and low epistemic uncertainty; a newly encountered deterministic rule can have a surprising first outcome and high epistemic uncertainty.

To decide whether an action is worth taking for information, the relevant quantity is not surprise but expected improvement in future decisions. If `D` is current data, the value of information for experiment/action `a` can be expressed as

`VOI(a)=E_{y|D,a}[ max_{π} E[U|D,y,a,π] ] − max_{π} E[U|D,π] − cost(a)`.

If observations do not change a future choice, the information may have zero decision value even if it is novel. This expression follows from comparing best expected utility with and without observing the result; computing it exactly may be costly.

**Consequence.** Novelty bonuses need a theory of what uncertainty they represent and why reducing it can improve future objective. Neither a rising error signal nor low exposure alone supplies that theory.

### R8 — Channel-permutation invariance is an equivariance requirement

Let `P` be a bijection that relabels anonymous sensory channels and `Q` a bijection on action labels. Let `T_{P,Q}` transform histories consistently. A semantically blind learner should satisfy, in distribution,

`π(T_{P,Q}h, P o) = Q π(h,o)`

when the task's objective and action effects are relabeled consistently. This is **equivariance**, not invariance of the raw action index. Deterministic tie-breaking by numeric channel order violates this property when candidates are equally supported, unless tie-breaking is itself transformed symmetrically or choices are evaluated after evidence breaks ties.

**Consequence.** Permutation tests must permute all relevant interfaces and map outputs back. Passing a fixed handful of permutations is evidence over those trials, not a proof of symmetry.

### R9 — Unbounded relational capacity needs selection or resource limits

With `k` active vertices, the set of all non-empty antecedent subsets has size `2^k−1`. If every subset is materialized/updated, compute and storage grow exponentially in `k`. Any scalable implementation must impose a resource bound, exploit sparsity, factorize interactions, or selectively create structure. A fixed maximum conjunction order is one bound, but it excludes higher-order functions regardless of evidence; unrestricted growth risks explosion.

**Consequence.** Structural selection is unavoidable at scale; the *selection rule* is the research question. A general rule should trade predictive/decision value against storage, compute, and uncertainty, not recognize benchmark identities.

## 3. What these deductions do and do not select

The requirements above imply broad capabilities, not a unique list of mechanisms:

- a representation that can express interaction and relevant history;
- an explicit predictive or objective-based update criterion;
- a way to assign delayed consequences when feedback is separated in time;
- an action evaluation that accounts for downstream objective;
- a memory strategy for nonstationarity and recurring conditions;
- a principled capacity/structure-selection rule;
- exploration only where information has expected decision value;
- symmetry under anonymous relabeling, to the extent the interface itself is symmetric.

Many mechanisms can implement each capability. For example, delayed credit may use traces or episode replay; regime memory may use recency, latent context, or a mixture model; value propagation may use TD, a predictive occupancy representation, or explicit search. The mathematics here does not prove which is best for this project. That requires comparing their different predictions under controlled conditions and matching the assumptions to the actual environments.

## 4. Unresolved claims to derive or test next

1. **Is the graph's node/edge representation sufficient for the five tasks?** Construct minimal histories that alias under the present representation but require different actions; otherwise do not add state machinery preemptively.
2. **What is the actual objective signal?** Determine whether benchmark wellbeing level, change, episodic return, or event labels are the target. A learner cannot infer which objective matters from naming alone.
3. **What representation is minimally sufficient for conjunction and generalization?** Compare explicit hyperedges, factorized interactions, and latent-context representations on truth-table and recombination tasks.
4. **How does the graph know an event is absent?** Establish observation completeness and event semantics before deriving inhibition or extinction rules.
5. **What information must be retained for delayed value?** Use matched delayed tasks to identify whether traces suffice, whether timing prediction matters, and whether counterfactual structure is needed.
6. **What shift schedule is intended?** Abrupt reversal, gradual drift, recurring regimes, and sensor remapping imply different memory demands. The combined current reversal task confounds them.
7. **Which mechanism provides the best cost/benefit under resource constraints?** Compare candidate implementations at matched graph size, interaction data, compute, and parameter tuning budget.

The next useful artifact should be a set of minimal counterexample environments and predicted outcomes for competing hypotheses. Only after those predictions are explicit should parameters or code be changed.
