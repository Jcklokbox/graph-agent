# Mechanism landscape: what could make a graph adaptive?

## Purpose and target

The engineering question is not “which biological analogy can be attached to the graph?” It is: **which small set of domain-general mechanisms lets a learned relational graph acquire useful structure, revise it under change, retain what remains valid, and choose actions whose consequences are beneficial?** Mechanisms are candidates until they survive discriminating tests. The eventual output of this research should be a ranked set of explicit claims, their assumptions, evidence, counterevidence, and implications for graph design—not a premature architecture.

“Self-adapting” does not mean solving arbitrary environments without information. Any proposal must state what the agent observes, what it can act on, what is stationary, what can change, and what is identifiable. If two environments yield the same interaction history but demand different actions, no learner can know which one it is in without additional information or prior assumptions.

## A. Functional decomposition (provisional)

A graph's flexibility may require distinct functions that can share representation but should not be conflated:

1. **Prediction:** estimate consequences of active relations and actions, including uncertainty and absent events.
2. **Structure formation:** decide which relations, conjunctions, abstractions, and contexts merit graph capacity.
3. **Credit assignment:** link delayed outcomes to preceding events/actions and decide which predictions have been fulfilled.
4. **Change inference:** infer whether a mismatch is noise, gradual drift, a new regime, or a changed observation/action mapping.
5. **Value propagation and choice:** estimate delayed consequences, transfer value through intermediate events, and select actions.
6. **Memory control:** retain reusable regularities and specific episodes without either overwriting everything or preserving every stale edge indefinitely.
7. **Information acquisition:** select behavior that is informative when that information can improve future decisions, while avoiding merely surprising or dangerous novelty.

A single graph may encode many of these relations. It does not follow that one scalar edge weight can represent all of them. Shared substrate is an architectural hypothesis; shared semantics requires derivation and tests.

## B. Candidate mechanisms and their trade-offs

### B1. Error-driven local prediction

**Candidate.** Every candidate edge predicts an observable target; update the prediction in proportion to outcome error, weighted by local activity/eligibility. In a linear model, `ŷ = wᵀx` and `w ← w + α (y−ŷ)x` is a stochastic-gradient step for squared error (with fixed features and suitable conditions). With a Bernoulli likelihood/logistic link, the same residual form is a likelihood-gradient update.

**Why useful.** A generic prediction-error rule can learn relations without labels such as “food” or “noise.” Repeatedly accurate predictors receive small corrections; systematic mismatch changes them. A target-specific distribution can distinguish a low mean from high irreducible noise.

**What it cannot establish by itself.** It does not decide which features should exist, provide an adequate representation, infer a new latent regime, solve delayed credit, or plan far into the future. RW is a useful special case for cue competition, not a general proof that all graph relations should be updated with one signed error. The current learner's independently accumulated expectation, confidence and positive/negative strengths are not this simple gradient rule (see `implementation-equation-audit.md`).

**Discriminator.** Compare predictive log loss/calibration and out-of-distribution response under stochastic outcomes, blocking, overshadowing, and uninformative correlated cues. Compare signed residual learning against the current max-subset baseline and adaptive associability. A learner that predicts average outcome but misrepresents uncertainty should fail uncertainty-sensitive action tests.

### B2. Structure learning: conjunction, abstraction, and latent causes

**Candidate family.** Add a graph relation, compound node, or context-indexed relation when doing so explains reproducible residual structure better than simpler alternatives. Candidate approaches include (i) sparse conditional-predictive gain, (ii) attentional/associability rules, (iii) configural representations, and (iv) latent-cause inference that clusters experiences into contexts/regimes and retrieves cause-specific memories.

**Why useful.** AND-like conjunctions address interactions not explained by parts; OR-like abstraction pools experiences with common predictive structure; latent causes protect old associations when new context-specific evidence conflicts. Gershman, Blei & Niv (2010) show a Bayesian latent-cause account that explains context-dependent learning/extinction phenomena, but such models have their own assumptions and do not automatically solve cue competition/blocking.

**Mathematical decision principle (not a biological claim).** Retain structure only when expected predictive or decision benefit exceeds complexity/storage cost. This can be written as a penalized criterion, e.g. `J(G)=L_data(G)+λ C(G)` for a graph `G`, or a Bayesian marginal likelihood. The cost must reflect a stated resource prior (description length, memory, computation), not a threshold selected because a benchmark fails. This principle requires held-out evidence: training fit always favors added structure.

**Discriminator.** Factorial cue designs separate additive prediction from genuine interaction; cue competition separates residual conjunction from indiscriminate compound storage; context switches, extinction/renewal, and return to earlier regimes separate latent causes from a single recency-weighted weight. Hold out recombinations to test abstraction rather than memorization.

### B3. Temporal credit: traces, temporal bases, and event models

**Candidate family.** Use a local eligibility state for active graph relations and an outcome-driven modulator (three-factor form); represent delay with a temporal basis or learned distribution when event timing itself predicts action. Distinguish (a) credit to past causes, (b) predicting when an event occurs, and (c) consuming a one-off predicted event. They are not the same operation.

**Biological status.** Synaptic eligibility is a plausible bridge between local activity and delayed neuromodulatory feedback; behavioral-timescale plasticity and synaptic tagging/capture provide related evidence but operate at different timescales and have different interpretations. A review documents synaptic tags that are transient and synapse-local, and distinguishes hour-scale consolidation tags from second-scale reinforcement eligibility; numerical biological timescales must be tied to the specific preparation, not imported as a universal parameter. See Frey & Morris (1997), Redondo & Morris (2011), and the review on behavioral tagging/eligibility below.

**Discriminator.** Vary cue-outcome lag while holding outcome statistics fixed; test transfer to untrained lags, overlapping causes, omission, repeated/sustained outcomes, and trace interruption. Fit a trace-only kernel against explicit event-time predictions and a latent temporal-state model. Report both learned causal credit and timing accuracy. A table of observed lags that never affects predictions is not a temporal mechanism for control.

### B4. Regime change: recency adaptation versus latent-state inference

**Candidate family.** A fixed learning rate tracks drift but mixes memories after abrupt reversal. A change-point/run-length posterior or latent-cause memory can switch quickly and return to an old mapping without relearning; a continuous state-space learner may handle gradual drift better. Hybrid mechanisms are plausible but increase complexity.

**Normative basis.** If regime `m` is hidden, Bayesian filtering updates `P(m_t | history)` from prior transition probabilities and predictive likelihoods. This is optimal only under the assumed model/prior. A surprise signal is evidence against current predictions, not proof of a regime change. Noise, outliers and sensor faults can produce the same immediate mismatch.

**Discriminator.** Cross abrupt reversals, gradual drift, isolated outliers, recurring regimes, changes to reward mapping, transition mapping, and sensor/action wiring. Evaluate adaptation latency, false-change rate, accumulated regret, and savings on regime recurrence. Compare a fixed-rate learner, a change-point model and a context-conditioned graph with matched compute/memory.

### B5. Planning/value: TD, successor representations, model-based inference, and chaining

**Candidate family.** (i) TD values propagate slowly from later outcomes; (ii) eligibility speeds credit over experienced trajectories; (iii) successor representations cache expected future occupancy and allow rapid reward revaluation under stable transitions; (iv) a transition model plus search/replay supports detours and transition revaluation; (v) conditioned reinforcement/chaining allows intermediate predicted events to support action sequences.

**Trade-off.** TD/eligibility can be local and efficient but cannot instantly evaluate unexperienced alternatives. SR interpolates between cached values and full planning but can become stale when transition structure/policy changes. Model-based planning handles such changes but requires a transition model and computation. Chaining is behaviorally grounded, but a simple scalar “intermediate cue inherits all future value” assumption is stronger than evidence warrants; test outcome-specific transfer, devaluation, and conditioned reinforcer controls.

**Discriminator.** After training, separately manipulate reward values (outcome revaluation) and transitions (detour/transition revaluation) without allowing new reward experience. Fast reward update + poor detour supports SR-like prediction; flexible detour supports model-based recomputation. Train versus test routes disambiguates chaining from explicit planning. Measure cost and horizon scaling, not only success on one two-step path.

### B6. Exploration: novelty, uncertainty, information value

**Candidate.** Represent epistemic uncertainty about predictions separately from aleatoric outcome variability and from reward. For action `a`, expected information gain can be written `E_{o'|a}[KL(p(θ|D,o',a) || p(θ|D))]`; decision-theoretic value of information instead asks how much the new information improves future choice. Neither is equivalent to unfamiliarity or prediction error.

**Potential failure.** A novel route may be dangerous, uninformative, or irrelevant. A high prediction-error bonus can repeatedly attract inherently noisy states. Pure information gain may be wasteful if information cannot affect later decisions. An optimistic prior can create novelty seeking that is not a separate curiosity bonus.

**Discriminator.** Cross novelty with epistemic uncertainty (unknown vs stochastic), controllability (can action change what is learned), and consequence (safe/harmful). Compare random, prediction-error, uncertainty, expected information gain and value-of-information policies under matched trial budgets. Test whether experience reduces directed exploration in learnable regions while persistent noise does not attract it indefinitely.

### B7. Memory stability, replay, and structural plasticity

**Candidate family.** Fast episodic storage + gradual integration; selective replay; metaplastic consolidation; or structural addition/pruning of graph relations. These may address different problems rather than being one “memory mechanism.” Complementary learning systems theory posits hippocampal rapid sparse storage and slower cortical integration as a solution to interference in neural network learning (McClelland, McNaughton & O’Reilly, 1995; O’Reilly et al., 2014). Experience-dependent synapse formation/elimination is supported, especially at local microcircuit scales, but the causal contribution of rewiring to adult learning is difficult to isolate; long-range rewiring is not an established general account of rapid adaptation (Barnes & Finnerty, 2010 review).

**Discriminator.** Sequentially learn recurring contexts with overlapping features; measure retention, recombination transfer, memory/computation cost and recovery after reappearance. Compare one fast graph with context-indexed episodic traces, replay/consolidation, and add/prune rules. If replay is added, determine which information it contributes beyond the graph's current predictive model.

## C. Cross-mechanism deductions

These are conditional logical implications, not empirical claims:

1. **An adaptive graph needs a prediction target and a revision rule.** Without an observable loss/error or another explicit update criterion, “adaptation” cannot be measured except as arbitrary state change.
2. **Error alone is insufficient for flexible behavior.** It changes estimates inside the current representational model; interaction, temporal dependency, hidden regimes or alternative routes require representation/structure or memory mechanisms that can express them.
3. **Fast adaptation and retention are in tension under a single mutable relation.** If observations from old and new regimes share one estimate, high step size accelerates reversal but overwrites old knowledge; low step size preserves but adapts slowly. This follows directly from an exponential average under a two-regime sequence. Separating regimes, storing episodes, or adapting uncertainty can change the trade-off, each with a cost.
4. **Credit assignment and planning are distinct.** An eligibility trace assigns a later teaching signal to past experienced events; planning evaluates possible consequences not necessarily experienced on the current trajectory. One does not imply the other.
5. **Representational growth requires a capacity/complexity rule.** Unrestricted hyperedge enumeration can express conjunctions but has `2^k−1` active subsets for `k` active channels; a sparsity/selection mechanism is required for scalability. A fixed order cap is a possible engineering approximation, not a general explanation of when structure is warranted.
6. **An uncertainty estimate is only useful if calibrated to a decision.** Exposure count, prediction error and posterior uncertainty measure different things. Exploration should be selected by expected future decision benefit if the objective is adaptive return.
7. **No-regime-label adaptation is possible only through inferred structure.** The graph can remain semantically blind by representing anonymous predictive regularities, but task/environment interfaces may still declare action and objective roles. Claims of complete channel parity must specify these interface asymmetries.

## D. Provisional claim ledger for later decision

| ID | Candidate claim | Status now | Minimum evidence before choosing it |
|---|---|---|---|
| C1 | Prediction-error updates are a useful generic plasticity substrate for edge-level learning. | Well-founded computational family; not sufficient alone. | Matched cue competition, stochastic calibration and prediction tests. |
| C2 | Adaptive associability is needed beyond static error updates for robust graph structure learning. | Open; competing accounts. | Redundant cue, latent inhibition, surprise transfer and held-out feature learning. |
| C3 | Compound relations should be created only for residual predictive/decision value. | Normative sparsity hypothesis; biological mapping open. | Factorial interaction and recombination holdouts with complexity cost sensitivity. |
| C4 | Context/regime-conditioned memory can improve abrupt reversal and recovery. | Computationally plausible; task dependent. | Abrupt/gradual/noise/recurring regime comparison with false alarms and regret. |
| C5 | Local eligibility is required for delayed credit, but not sufficient for time prediction or planning. | Supported by algorithmic distinction; biological implementation remains open. | Lag generalization, omission and counterfactual detour tests. |
| C6 | A predictive occupancy representation may be a useful middle ground for value propagation. | Established candidate algorithm, not an obligatory graph design. | Reward revaluation plus transition revaluation, and horizon/cost comparison. |
| C7 | Information seeking should be valued by impact on future decisions, not novelty alone. | Decision-theoretic consequence under utility assumptions. | Safe/unsafe, learnable/noisy and controllability factorial exploration. |
| C8 | A fast-memory + slow-integration or replay mechanism may balance rapid learning and retention. | Biological/computational hypothesis with supporting CLS literature. | Recurring task sequence, retention, transfer and matched-capacity baselines. |
| C9 | Structural graph growth/pruning may aid capacity but is not automatically needed for adaptation. | Open; evidence for biological rewiring does not prove computational necessity. | Compare fixed topology + weights, dynamic edges, and context memory at equal resource budget. |

Do not select all nine mechanisms by default. The goal is to determine the smallest subset with unique explanatory value, through ablations and task splits that distinguish substitutes. A mechanism is worth adding if it improves held-out prediction or decision behavior in the regime where its theory predicts benefit, without causing unacceptable regressions elsewhere.

## E. Research workflow and standards

For each candidate mechanism, maintain one record containing:

- exact computational definition and variables;
- assumed observations, stationarity, and capacity;
- mathematical result (proof/derivation or explicit lack thereof);
- animal evidence, species/preparation, and caveats;
- what competing mechanism predicts differently;
- the minimal experiment and a negative control;
- predicted result before running it;
- code/equation mapping only after the theory is specified;
- metrics, seeds, uncertainty and failed outcomes;
- decision: retain, revise, reject, or unresolved.

The five project scenarios are end-to-end constraints, not a license to tune parameters against them. Use independent synthetic tasks to discriminate theory, then evaluate the frozen candidate mechanism on all five. If a principled mechanism fails a scenario, first determine whether the environment matches its assumptions; do not add a special-case branch to force a pass.

## Selected sources for this landscape

- Barnes, S. J. & Finnerty, G. T. (2010). Sensory experience and cortical rewiring. *Neuroscientist* (review). https://www.sciencedirect.com/science/article/pii/S0149763417304864
- Frey, U. & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385, 533–536. https://doi.org/10.1038/385533a0
- Gershman, S. J., Blei, D. M. & Niv, Y. (2010). Context, learning, and extinction. *Psychological Review*, 117, 197–209. https://doi.org/10.1037/a0017808
- McClelland, J. L., McNaughton, B. L. & O’Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102, 419–457. https://doi.org/10.1037/0033-295X.102.3.419
- Monosov, I. E. (2024). Curiosity: primate neural circuits for novelty and information seeking. *Nature Reviews Neuroscience*, 25, 195–208. https://doi.org/10.1038/s41583-023-00784-9
- O’Reilly, R. C. et al. (2014). Complementary learning systems. *Trends in Cognitive Sciences*, 18, 310–320. https://pubmed.ncbi.nlm.nih.gov/22141588/
- Redondo, R. L. & Morris, R. G. M. (2011). Making memories last: the synaptic tagging and capture hypothesis. *Nature Reviews Neuroscience*, 12, 17–30. https://doi.org/10.1038/nrn2963
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. http://incompleteideas.net/book/the-book-2nd.html
- Russek, E. M. et al. (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. https://gershmanlab.com/pubs/Russek17.pdf
