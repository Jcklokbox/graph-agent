# Theoretical synthesis: adaptive prediction, credit assignment, and change

**Status.** This is a research proposal, not a description of the current implementation and not a claim of biological equivalence. It separates (i) established behavioral/computational results, (ii) mathematical consequences under stated assumptions, and (iii) hypotheses for this project. The aim is a compact set of mechanisms that can explain all five behavioral tests without test-specific branches. A theory cannot guarantee those tests pass: implementation and measurement remain empirical.

## 1. Problem formulation

At time `t`, the agent receives anonymous sensory activity `o_t`, selects an action `a_t`, and observes `o_{t+1}` and a scalar homeostatic outcome `r_{t+1}` (the benchmark's wellbeing change/level). The task may be partially observed and its generating regime may change. The learner should estimate what follows from actions in context, discover when its predictive model is stale, allocate credit across delay, and compare candidate actions by their expected future consequences.

A useful sufficient internal state is not necessarily the raw observation. Let `b_t(z)` be a belief over latent context/state `z`, and let `M_t(s,s')` predict future occupancy under the current transition structure. The proposal has three interacting quantities, not one overloaded association strength:

- **predictive model** `p(o', r | z, o, a)`;
- **belief/change estimate** `b_t(z, m)` over latent state and regime `m`;
- **value** `V_t` derived from predicted outcome streams, not equated with confidence or novelty.

Anonymous channel permutations should act as a relabeling symmetry: if observations and action labels are consistently permuted, predictions and choices should map correspondingly, except for explicitly documented tie-breaking.

## 2. Established findings versus modeling choices

### Empirical / computational foundations (not universal laws)

- In Rescorla–Wagner (RW), cue strengths change in proportion to outcome prediction error; competition among simultaneously present cues explains blocking in the model. Blocking is a behavioral phenomenon, but RW is not a complete theory of conditioning. Attention, configural representations, retrieval, and context can matter. Pearce–Hall instead makes associability depend on prior outcome surprise; Mackintosh-style accounts emphasize predictive relevance. These mechanisms make different predictions about learning to redundant, uncertain, or pre-exposed cues [Rescorla & Wagner, 1972; Pearce & Hall, 1980; Mackintosh, 1975; review: Pearce, Kaye & Hall, 2012]. Inhibition is conventionally established with both **summation** (a candidate inhibitor reduces responding to an independently trained excitor) and **retardation** (it acquires excitation more slowly than a control cue); either test alone admits alternatives [Rescorla, 1969; overview in Bouton, 2e].
- Temporal-difference learning defines `δ_t = r_{t+1} + γV(s_{t+1}) − V(s_t)`. Eligibility traces distribute this error over recently active features; they address temporal credit assignment but do not, by themselves, provide counterfactual planning [Sutton & Barto; Sutton & Barto, 1987].
- Successor representations separate predictive state occupancy from outcome values: `M^π(s,s') = E_π[Σ_{k≥0} γ^k 1(S_{t+k}=s') | S_t=s]`, and `V^π(s)=Σ_{s'}M^π(s,s')r(s')`. This supports rapid reward revaluation while transitions remain stable, but cached successor structure alone is not generally flexible to transition changes [Russek et al., 2017; Stachenfeld et al., 2017].
- Online Bayesian change-point models maintain posterior mass over run length; their conclusions depend on the hazard and observation model [Adams & MacKay, 2007]. Reversal behavior consistent with latent-state inference has been observed in specific primate tasks, but is not evidence that all animals use one exact Bayesian algorithm [Bartolo & Averbeck, 2020].
- Information seeking, novelty, and reward-driven exploration are dissociable. Novelty can orient behavior, but “uncertainty” is not automatically a reward and random exploration is not equivalent to curiosity [Monosov, 2024 review; Hsee & Ruan, 2016; see also project bibliography note].
- Conditioned cues can influence action and intermediate cues can support conditioned reinforcement, but transfer depends on outcome-specific and general motivational representations and experimental conditions; a value chain is a computational hypothesis, not a universal scalar propagation law [Pavlovian-to-instrumental transfer reviews; Zhang et al., 2009].

These sources motivate candidate mechanisms, not direct claims about the benchmark or neural implementation. Search-derived summaries should be checked against original methods/results before quantitative or biological claims are made.

## 3. A coherent candidate model

### 3.1 Prediction error with feature competition

For a feature vector `x_t` built from active sensory, action, context, and eligible-trace features, predict a vector of next-event/outcome statistics `ŷ_t = W_t x_t` (or a sparse nonlinear function when conjunctions are warranted). For a scalar bounded outcome, use a calibrated link such as `p_t = σ(w_tᵀx_t)` and Bernoulli error `δ_t = y_t − p_t`; for continuous wellbeing use an explicit likelihood/noise model rather than calling arbitrary bounded activations probabilities.

A minimal local update is

`e_t = ρ e_{t−1} + ∇_w f_w(x_t)`

`w_{t+1} = w_t + α_t δ_t e_t`.

Here `ρ` is a decay in **steps or physical time**, specified by the environment; `e_t` is eligibility, not confidence. For delayed events, maintain event-specific eligibility and let observed timing shape a temporal basis/kernel. An event prediction should be consumed or downweighted after occurrence when the predicted event is one-off. This discharge rule is a hypothesis to test against repeated-event environments; blanket discharge could be wrong for sustained or recurrent outcomes.

Competition can be expressed by the summed prediction/error across active causes, as in RW, but does not imply that every compound must be stored. Use a sparse representation selected by out-of-sample predictive improvement: retain a conjunction only when its conditional predictive gain exceeds its complexity cost. That is a model-selection principle, not a benchmark-specific threshold. Compare RW-like competition with adaptive associability and configural alternatives; do not equate a max-of-subsets reference with a derived RW rule.

For inhibitory learning, represent signed predictive effects or conditional outcome distributions, rather than allowing an independent positive and negative accumulator to cancel without calibration. Evaluate inhibition by summation and retardation tests and by recovery/renewal after context change. The inhibitory relation should be conditional on the context in which absence was predicted; absence is data only where the event was observable.

### 3.2 Latent context and adaptation

Maintain a posterior over regimes `m` (e.g. sensor/action mapping or transition contingency), updated by predictive likelihood:

`b_t(m) ∝ p(o_t,r_t | h_{t−1},m) Σ_{m'} p(m | m') b_{t−1}(m')`.

The transition prior may be represented as a hazard, but the hazard must be learned, justified by the task family, or included in sensitivity analysis. A mismatch/surprise score alone is insufficient: noise can also produce surprise. Rapid reversal should emerge when evidence raises posterior probability of an alternate regime; retaining regime-conditioned memories avoids destructive overwriting. If the change is gradual, a change-point-only model can fail; compare drift, switching, and mixed models.

This permits principled relearning without knowing which channel changed: inference is over predictive structure and candidate regimes, not semantic labels. It does not grant magical zero-shot adaptation where the new regime is observationally indistinguishable from the old one.

### 3.3 Action value and information seeking

For action `a`, evaluate expected discounted wellbeing under a learned transition/outcome model:

`Q(b,a) = E[Σ_{k≥0} γ^k r_{t+k+1} | b_t, a_t=a, current model]`.

A computationally efficient approximation may use a learned successor representation `M`; a transition surprise should trigger updating/recomputation rather than assuming the cached map remains valid. Long routes require either multi-step prediction, a learned predictive representation, replay, or actual experienced value propagation. The current direct active-edge score does not establish any of these.

Separate **instrumental value** from **epistemic value**. A principled information-seeking candidate is expected information gain

`IG(a) = E_{o'|a}[KL(b_{t+1} || b_t)]`,

but this is not automatically intrinsic reward. Prefer a decision-theoretic score such as `Q(a) + β_t IG(a)` only when information can improve future choices; otherwise pure curiosity may waste resources. `β_t` should follow a stated resource/uncertainty objective, not elapsed-time schedules. Test novel-but-dangerous versus novel-and-safe actions, and compare against uncertainty-neutral and random baselines. A simple optimism bonus based on low confidence may not be calibrated information gain.

## 4. What follows mathematically, and what does not

Given a correctly specified stationary Markov model and a fixed policy, the successor identity `V=M r` follows from the discounted occupancy definition. If only `r` changes, keeping `M` fixed and recomputing `V` is valid. If transition dynamics, observation mapping, or policy-conditioned occupancy changes, old `M` may be stale and needs updating. This is a mathematical implication of definitions, not evidence the brain explicitly stores a matrix.

Given a differentiable prediction model and a stated likelihood, the gradient update above is stochastic gradient descent on expected prediction loss (subject to standard step-size/stationarity assumptions). With nonstationarity it need not converge; adaptation speed and stability trade off. Eligibility traces implement a temporal weighting of recent gradients; they do not solve hidden-state inference or causal counterfactual credit by themselves.

A Bayesian regime posterior is optimal only relative to its prior, likelihood family, and candidate model class. A misspecified posterior can become confidently wrong. Bounded arithmetic alone gives neither calibrated probabilities nor stability guarantees. `A/M/N/D`, confidence accumulation, independent excitation/inhibition, and max-reference blocking need derivations before their values can be interpreted probabilistically.

## 5. Predictions and discriminating tests for the five benchmarks

| Benchmark | Required behavior | Candidate mechanism | Diagnostic contrast / failure meaning |
|---|---|---|---|
| 1. Informative cues, noise, conjunctions | Learn distinct cue-outcome links; ignore uninformative noise; admit conjunction only for residual information | Prediction-error learning plus sparse complexity control / associability | Compare cue-only vs cue+noise, blocked vs novel compound, and held-out compound contexts. Persistent noise learning suggests base-rate/attention defect; no conjunction suggests over-regularization or representation failure. |
| 2. Autonomous learning under channel/action permutations | Learn useful behavior without semantic labels and preserve mapped behavior under bijective relabeling | Shared feature/model update and symmetric action evaluation; exploration separated from value | Paired seeds, multiple permutations, mapped action agreement, return vs stay and random baselines. Failure across labels indicates symmetry/interface defect; failure everywhere indicates action-value or exploration problem. |
| 3. Cancellation and variable delay | A blocker suppresses an excitor conditionally; event timing is learned over delay variation | Signed/contextual prediction plus eligibility/time basis and event-specific discharge | Require summation and retardation, blocker-alone and excitor-alone controls, timing generalization and repeated outcomes. Correct timing but absent suppression isolates inhibition; suppression without transfer/retardation may be response competition, not learned inhibition. |
| 4. Delayed distant goal | Choose an initially costly route whose delayed consequences improve wellbeing | Multi-step predictive occupancy/value propagation, or empirically learned conditioned value chain | Short vs long route, shuffled transitions, reward devaluation, transition revaluation/detour. Reward revaluation with stale route is SR-like; detour failure diagnoses cached transition structure. No benefit over neutral baseline means value has not propagated. |
| 5. Wiring reversal without reset | Detect change, switch behavior, retain useful old knowledge without catastrophic interference | Regime belief/change detection plus context-indexed memory and revaluation | Abrupt vs gradual reversals, noisy false alarms, change in action mapping vs reward mapping, recovery on return. Gradual relearning alone can be adequate; abrupt switching is not assumed unless benchmark timing demands it. |

These tests establish behavioral adequacy only for their distributions. Add controlled mechanism tests where an existing scenario does not distinguish alternatives; avoid expanding the suite merely to create implementation targets. Report per-seed metrics, confidence intervals, baselines, adaptation latency, post-switch regret, and pre/post-switch retention.

## 6. Immediate empirical program

1. **Equation audit:** record the exact present update equations, initialization and activation semantics from source; compare them to this proposal. Do not retrofit the current code into the theory by renaming fields.
2. **Model discrimination:** generate matched datasets for RW/attention, signed vs separate excitation-inhibition, explicit regime inference vs constant-rate adaptation, trace-only vs predictive-state value propagation, and random vs information-gain exploration. Score held-out likelihood and behavior, not just training reward.
3. **Ablations as causal tests:** remove one mechanism at a time and predict which observable fails; preserve all other settings. Include null and adversarial controls.
4. **Robustness:** random seeds and channel permutations; parameter sensitivity over interpretable ranges; abrupt/gradual reversals and stochastic outcome noise. A result that exists at one fitted point is weak evidence.
5. **Biological calibration (optional future work):** only compare with animal data after matching task, deprivation, sensory/motor constraints, and outcome measures. Do not infer neural homology from shared equations.

## 7. Limitations and falsifiers

This account is falsified or materially weakened if: a simple stationary predictor explains all benchmark behavior while regime inference adds no predictive value; information gain systematically selects harmful novel actions despite a well-defined decision objective; the proposed eligibility kernel cannot fit variable-delay generalization without arbitrary per-task timing; inhibitory effects fail standard transfer/retardation controls; or value-chain performance requires scenario-specific rollout depth/thresholds. A computationally successful alternative should be preferred over preserving this synthesis. No one mechanism is asserted to be biologically necessary.

## References (starting points)

- Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742. https://arxiv.org/abs/0710.3742
- Bartolo, R. & Averbeck, B. B. (2020). *Prefrontal cortex predicts state switches during reversal learning*. Neuron. https://pmc.ncbi.nlm.nih.gov/articles/PMC7422923/
- Bouton, M. E. *Learning and Behavior*, 2nd ed., chapter 4 summary. https://bouton2e.sinauer.com/summary04.html
- Monosov, I. E. (2024 review/article context). Novelty and information seeking in primates. https://www.nature.com/articles/s41583-023-00784-9
- Mackintosh, N. J. (1975). A theory of attention: Variations in the associability of stimuli with reinforcement. *Psychological Review*, 82, 276–298. https://doi.org/10.1037/h0076778
- Pearce, J. M. & Hall, G. (1980). A model for Pavlovian learning: Variations in the effectiveness of conditioned but not of unconditioned stimuli. *Psychological Review*, 87, 532–552. https://doi.org/10.1037/0033-295X.87.6.532
- Pearce, J. M., Kaye, H. & Hall, G. (2012). A configural theory of attention and associative learning. *Learning & Behavior*. https://doi.org/10.3758/s13420-012-0078-2
- Rescorla, R. A. (1969). Pavlovian conditioned inhibition. *Psychological Bulletin*, 72, 77–94. https://doi.org/10.1037/h0027760
- Russek, E. M. et al. (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. *PLoS Computational Biology*. https://gershmanlab.com/pubs/Russek17.pdf
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275, 1593–1599. https://doi.org/10.1126/science.275.5306.1593
- Sutton, R. S. & Barto, A. G. (1987). A temporal-difference model of classical conditioning. *Proceedings of the Cognitive Science Society*. https://escholarship.org/content/qt9ps125p9/qt9ps125p9.pdf
- Stachenfeld, K. L., Botvinick, M. M. & Gershman, S. J. (2017). The hippocampus as a predictive map. *Nature Neuroscience*, 20, 1643–1653. https://doi.org/10.1038/nn.4650
- Zhang, J. et al. (2009). Neural network models of reinforcement learning and conditioning. https://sites.lsa.umich.edu/m3lab/wp-content/uploads/sites/1376/2024/11/2009-Zhang-Neural-Network-B.pdf
