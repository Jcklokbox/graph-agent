# The posterior-graph model (v11)

This note is the mathematical model behind `development/agents/v11.py` (formerly `development/posterior_graph.py`). It describes the
model as one system: one kind of statistic, one test for structure, one way to combine levels
of context, and a planner on top. Results and failures are in [results.md](results.md). This
is a description of what the code does and why, not a claim of biological equivalence or
general optimality.

## 0. Setting

At step `t` the agent sees `x_t ∈ [0,1]^n` (anonymous channels; one of them, `v`, carries the
value `w`), chooses one of `A` actions and then sees `x_{t+1}`. Nothing else is given: no task
labels, no reward channel separate from the observation, no designated cues.

* **Value frame.** `w` is re-coded as `z = 1/2 + (w − ref)/S`, where `ref` is the median
  experienced value (status quo) and `S = max(2·max|w − ref|, value_scale)`. When the frame
  changes, every stored value statistic is re-expressed exactly (`z' = a z + b`). Scores are
  therefore invariant to affine re-scaling of the reward (tested).
* **Active set.** `S_t = {i : x_{t,i} > 1/2}`. In the value frame "active" means better than
  the status quo. Every prediction depends on `x` only through `S_t`, the action and the open
  expectation windows.

## 1. One statistic: the discounted conjugate cell

Every learned quantity is a *cell*: Beta(1 + k, 1 + n − k) for 0/1 data or Welford moments
for fractional data (predictive variance with one Uniform(0,1) pseudo-observation).
Observations may carry a weight (timing exposures are weighted by salience, §4).

**Change points.** Before absorbing `y`, a cell computes

    S ← min(50, S + log p0(y) − log p(y | cell)),     c = σ(logit h + S),

(`p0` = 1/2 for 0/1 data, the uniform density otherwise), discounts its evidence by `1 − c`
and keeps `S·(1 − c)` (CUSUM: a run of surprises adds up, predictable outcomes drain it).
`h = change_hazard = .01`. Memory therefore saturates near `2/h` *visits* of a stable cell and
collapses after two or three surprising ones; a cell that is not visited does not forget.
Event-count cells (timing, event baseline, cancellation contrast) use the same rule without
the accumulator (`S` = the current step's log ratio only): their events are rare by design,
and accumulating their surprise erased timing memory in the delayed-food tasks (Round 13).

## 2. One test for structure: the split Bayes factor

All structural decisions compare "the rate depends on a partition of the data" with "one shared
rate", for counts `(k_g, n_g)` of the groups `g`:

    log BF = Σ_g log B(1 + k_g, 1 + n_g − k_g) − log B(1 + Σk, 1 + Σn − Σk).

| decision | groups | use |
|---|---|---|
| **attention**: does channel `i` predict channel `j`? | `i` on / off at lag `k`, within (action, lag, previous state of `j`) | `i, j` attended when the summed log BF > log(#tests/α) |
| **controllability**: does the action matter for `j`? | actions, within (attended context, open windows) | `P(uncontrollable) = σ(−Σ log BF)` |
| **timing relevance**: does context channel `c` modulate a timed effect? | `c` on / off among the context children of a timing cell | an uncontrollable channel tested on both sides and without effect is marginalised |

Conditioning each comparison on what could confound it (the target's own past for attention,
the context and the open windows for controllability) is what makes the same test usable for
three different questions. *Irrelevance must be shown*: a channel never tested on one side
keeps its split.

Two decisions use large-sample versions of the same comparison rather than the exact Beta
Bayes factor: timing relevance uses the score (χ²) statistic of the same 2-group split with
the threshold `max(3.84, log n)` (BIC-like), and a timing cell is *credible* when its rate
differs from the event baseline by more than two posterior sd. Replacing both by the exact
split BF > 0 was tested in Round 13 and lost .06 on DEV (the uniform-prior Occam penalty
makes rare-event timing cells too slow to count as informative).

## 3. One way to combine context: a chain of shrinkage

The prediction of channel `j` after `(x, a)` walks down a chain of increasingly specific
contexts; each level is a cell whose prior is the prediction of the level above:

    p_L = (k_L + m_L · p_{L−1}) / (n_L + m_L)

| level | key | prior strength `m` |
|---|---|---|
| 0. factor prior | single cues (see below) | — |
| 1. attended context | attended part of `S`, `a` | `m = 2·n₀/(1+n₀)`, `n₀` = support of level 0 |
| 2. exact context | `S`, `a` | learned: `m = within-variance / τ²`, `τ²` = spread of exact contexts around their attended parent (empirical Bayes over all parents) |
| 3. belief state | `S`, `a`, informative open windows `(cue, age)` | 2 |

**Level 0 — lifts over a baseline.** For a cue `i` the cue-present and cue-absent cells give
the *lift* `s_i = lift(p_i, b_i)`, the fraction of headroom gained (`p ≥ b`) or of the baseline
removed. Enabling cues combine as a noisy-OR over the mean background, suppressive cues scale
the background down, and learned inhibitors (a 2×2 cancellation contrast that transfers across
excitors) remove a fraction of the result. Kamin blocking keeps cues that co-occur with an
already-sufficient explainer from being credited. Level 0 matters only for situations not yet
seen; any visited context overrides it as its evidence grows.

**Value channel.** For `j = v`, level 0 also includes the *model-based* value: the predicted
next observation is fed to the reward model `R(y) → w` (exact next state, else its attended
part pooled over siblings, else the uninformed prior) and combined with the factor prior by
precision. A predicted state never seen is worth checking.

**Controllability.** The action-pooled cells (same keys with `a = *`) carry what was observed
under any action; the prediction is the mixture `P_unc · pooled + (1 − P_unc) · action-specific`.

**Timing lift.** Open windows of a cue add the strongest credible lift of their timing cell
`(cue, j, lag, a)` over the event baseline; that cell itself is a chain
generic → reduced context (relevant channels only, §2) → exact context, each level shrunk
toward the one above with `prior_strength` pseudo-observations (the same rule as §3; an
empirical-Bayes strength was removed in Round 13 without loss). The lift is a prior: it is shrunk by `2/(2 + n)` with `n` the evidence of the
belief-state cell, which has observed the timed effect directly.

## 4. Time: expectation windows

A rising edge of a *transient* channel (mean run ≤ 3 steps) opens windows `(cue, target, age)`
for up to `max_age = 12` steps; a recurrence renews them; an event consumes the window; an
unobserved target right-censors it. Every window step adds exposure (and the event, if any) to
its timing cell, weighted by the **salience** of the onset, `1 − P(onset)` before it happened:
a predicted (e.g. self-caused) onset carries no timing information beyond its predictor. A
window enters the belief-state key only at lags with a credibly non-baseline rate.

## 5. Change of the world: global change points and latent regimes

Joint evidence that the whole world changed is the log likelihood ratio fresh-prior vs
exact-context cells over all channels of a transition, made outlier-robust (one step adds at
most `log 1/ε`, ε = .05) and accumulated over a run of surprise; `c = σ(logit H + S)` with
`H = .001`. Once `c > .25` the current model is archived as a latent regime (at most 4), then
every cell of every table (including the attended-context and reward cells) is discounted to
`1 − c` once per run. On surprising transitions stored regimes
compete with the current model by exponentially weighted predictive evidence; one that leads by
3 nats is reinstated (savings on reversals instead of relearning).

## 6. Control

    Q(x, a) = E[z'] + sqrt((explore · sd)² + B(x, a)²) + γ · E_{x'} max_{a'} (E[z''] + B'(x', a'))

* `E[z'], sd`: the predicted value and its posterior sd (for a novel predicted state this is the
  uninformed reward prior). Optimism only at the root; imagined successors use plain means and
  never activate the value channel.
* `B = event_explore · salience/(1 + n) · (1 − E[z'])`: the value of testing an open, salient,
  not-yet-located expectation; `n` counts tests of (cue, lag, action, reduced context). Posterior
  sd and `B` are two uncertainties about the same reward: their variances add (one optimism
  budget per transition).
* `E_{x'}`: expectation over the predicted next observation, enumerating at most 4 uncertain
  attended channels.

## 7. Parameters

`gamma .9, explore 2, event_explore .5, max_age 12, change_hazard .01, global_hazard .001,
alpha .01 (attention), value_scale .5, prior_strength 2, rollout_branch 4`; inhibition and
blocking thresholds (excitor `.65`, credit `.5`, blocking lift `.6` and error `.25`); regimes (`ε .05, store .25, decay .8, 3 nats,
4 regimes`). Nothing is tuned per task.

## 8. What the consolidation removed and why

Details and numbers in [results.md](results.md), "Round 13". Removed because the general
mechanism already covers them (DEV equal or better without them):

* **Pair (conjunction) births** with their own 2×2 Bayes factors: exact and attended contexts
  already represent every conjunction that has been seen, and the pair prior was harmful in
  unseen contexts (DEV +.034 without it).
* **Exact-context controllability**: a second controllability sum over exact contexts; the
  attended-context version (§2) is the same test without fragmentation by noise channels.
* **Protection of strong cue cells** from cancellation credit: the 2×2 contrast already
  excludes excitors (DEV +.009 without it).
* **Empirical-Bayes strength of the reduced timing level** (method-of-moments concentration):
  a fixed `prior_strength` is the same rule as in the rest of the chain.
* **Exceptions in the global discount**: the attended-context and reward cells now forget
  with everything else.

Tested and kept, because removing them hurt: Kamin blocking (−.06), learned inhibition (−.03,
cancellation with delayed food collapses), the irrelevance test of timing context (replacing
it by attention: −.12, dist breaks), the coarse reward level (−.005 overall but −.10 on the
noise variants it exists for), the memoryless change point of event counts (−.014, delayed
food), the 2-sd / χ² large-sample tests (exact BF: −.06), and the plain (non-shrunk) mixing
of the action-pooled cells (−.04).

## 9. Open problems

* **Deep exploration.** With a two-step planner, a test that needs two costly steps after a cue
  (dist with iid noise) is only attempted when the immediate bonus is large enough; more
  optimism helps there and hurts the delayed-food tasks.
* **Gain/loss asymmetry of the value frame** (dist-sqrt: food +.21, shock −.5 relative to the
  reference): no exploration at all. The frame is symmetric around the reference.
* **Delayed food with noise** remains near the stay baseline on fresh variations.
* Level 0 is a heuristic generaliser (noisy-OR, inhibition, blocking thresholds); it is the
  least principled part of the model and the part that unseen contexts depend on.
