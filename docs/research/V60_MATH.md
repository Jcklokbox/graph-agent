# v60: from observation to action and back

This is a specification of **v60's default implementation** as read at
`88318f8d95bd7868d5927fb0d5e8869b2827bc92`, not a proposed replacement or a
proof that its rules are optimal. The equations describe the computations,
not an instruction to reorder reductions for bitwise reproduction.
Read the sections in order to follow one transition; use the final section to
locate places where the rules interact or where a premise should be tested.
[Architecture](ARCHITECTURE.md) is the shorter map; [evidence](../EVIDENCE.md)
identifies the historical freeze and the scope of the accepted result.

**Labels:** **Exact** means an identity or finite calculation for the stated
model, not exact inference about the environment. **Approximation** means a
limited hypothesis family, projection, sample, or approximate update.
**Heuristic** means a decision rule or design constant not implied by the
probabilistic model. One calculation can have more than one label.

1. [One real transition and its state](#1-the-loop-and-the-three-different-notions-of-state)
2. [Sensory prediction](#2-sensory-prediction-from-history-to-a-joint-law)
3. [Controlled motion and reward](#3-controlled-motion-and-reward-what-the-real-transitions-fit)
4. [Exploration and information](#4-what-gets-treated-as-a-promising-experiment)
5. [Action values and choice](#5-the-action-value-one-two-step-calculation)
6. [Real learning updates](#6-close-the-loop-what-learny-actually-changes)
7. [Hand checks](#7-two-small-hand-checks)
8. [Audit questions](#8-where-i-would-audit-first)

## 1. The loop and the three different notions of state

Let `x` be the current observation, `a` the chosen action, `y` the next
observation, and `r = y[v]` the bounded real utility at the designated value
index `v`. With `J` other channels and `A` actions, the sensory bits are
`b[j] = 1{x[j] > 1/2}` after removing `v`. The value channel is **not** a
sensor bit, and the learner does not receive the task label, oracle, or
separately scored benchmark reward. Write `b'` for the sensory part of `y`.

| State | Meaning | Where it affects the computation |
| --- | --- | --- |
| `(i,z)` | Sensory particle `i` and local latent regime `z` | Mixture over the next sensory bits; not an actuator state |
| HMM state | Hidden state inside one discovered group and one HMM size | Supplies a competing per-channel sensory forecast |
| `s = c(b)` | Bit-coded subset of up to four *observed*, action-sensitive channels | Transition matrix for planning, reward laws indexed by arrival `c(b')` |

A separate, sequential reward-mode detector decides which control and reward
statistics are current. A reward mode is not the sensory regime `z`, and a
slow sensory channel is not by definition a controlled reward mark.

The [v60 constructor](../../development/agents/v60.py) installs `JointReward`
in the canonical `factor-L4-P-C` model. Defaults: 200 particles, at most 10
sensory regimes per particle, lag grid `0..4`, own lags `0..2`, five discovered
parent slots per target, and five rollout draw rows. Action roles in the
sensory slow-channel test are **off**. Reward posterior sampling in the
controller is **off**. The controller and the factor have separate RNGs.

The [controller](../../development/agents/factor_controller.py) draws one
`5 × (J+4)` array of uniforms, calls `two_step_values`, and chooses uniformly
among the exact floating-point maximizers. All actions share those uniforms
when they need samples. It then calls **real** `predict(x,a)` to populate the
learning cache and keeps `(x,a)` pending. `learn(y)` consumes it once. The
active planning path is `two_step_values → event_values → provisional_values`;
other rollout functions and the generic alternative reward heads are not
part of this path. The last uniform column is not consumed by this provisional
rollout; its draw still advances the controller RNG.

## 2. Sensory prediction: from history to a joint law

Code: [factor.py](../../development/models/factor.py),
[kernels.py](../../development/models/kernels.py),
[discovery.py](../../development/models/discovery.py).
Write `normalize` for division by total mass, `sigmoid(u) = 1/(1+exp(-u))`,
and `Bern(z;p) = p^z (1-p)^(1-z)` for a bit `z`.

### 2.1 Channel roles and place

Only continuations after a channel was 1 contribute to the role test. For
channel `j`, accumulate log likelihoods `T[j,h]` for a tonic continuation with
`P(stay)=1-h`, over `h ∈ {0.0003, 0.001, 0.003, 0.01, 0.03}`. In parallel,
accumulate log likelihood `F[j]` under a phasic continuation with online
Beta(1,1) counts: the likelihood for the observed next bit is
`(its count + 1)/(both counts + 2)` before updating those counts.

Set `u[j,h] = T[j,h] − F[j] − log J` and
`R[j,h] = Σ_{k≠j} log(1+exp(u[k,h]))`. The implemented log odds are

```text
L[j] = logsumexp_h(R[j,h] + u[j,h]) − logsumexp_h R[j,h].
```

This marginalizes the common tonic rate on a finite grid and uses prior tonic
odds `1:J`. **Exact** for that comparison; the hard label `slow[j] = L[j] > 0`
is a MAP decision, not posterior averaging. `place` is the set of active
slow bits. A step is `quiet` if no fast bit is active. Slow bits themselves
are predicted with separate Beta(1/2,1/2)-smoothed two-row Markov counts,
conditioned on their current bit, not on action.

### 2.2 Sensory regime weights

For each particle, the new-regime proposal is the normalized vector of
regime entry counts plus one unit on the next unused regime when a slot is
available. Before the first update the mass is all on regime 0. For a known
place, form `D = normalize(place_occupancy + G)`, where `G` is that proposal;
mix `D` with the last stored posterior there, weighting the posterior by
`(1 − h_i)^(gap+1)` (particle `i` has its own sampled grid hazard). `gap` is
`max(0, t − last_t − 1)`. This is a bounded recurrence rule, not an unlimited
nonparametric posterior. The place table retains at most 10 LRU entries.

With both slow channels and active fast cues, route the prior by the
probability of seeing *one token from the active fast set*. For a regime take
`(sum of its cue counts on that set + sum of pooled cue probabilities on that
set)/(total regime cue counts + 1)`; the pooled distribution normalizes summed
regime cue counts plus `1/2` each. Multiply the prior by that number and
renormalize within each particle. It is input-dependent routing, not another
independent likelihood on the next observation. After routing, component
mass is `w[i,z] ∝ exp(log_weight[i]) × prior[i,z]`.

### 2.3 Local emission members

For each target and action, form a binary feature vector `U` from the three
own lags, one-hot action, and up to five installed delayed source bits.
Missing history and inactive slots read as zero. The table bank contains the
empty parent set, all singletons and all pairs of feature positions. It does
not enumerate arbitrary higher-order tables. Let `g1,g0` be pooled counts in
one table cell. Its pooled predictive `p_g` is `(g1 + p_backoff)/(g1+g0+1)`:
back off to `1/2` for the empty set, the empty set for a singleton, and the two
singleton predictions weighted by their exponentiated pooled log scores for
a pair. In regime `(i,z)`, a cell predicts `(q1 + p_g)/(q1+q0+1)`.

The extra additive member is `p_add = clip((μ_shared + μ[i,z])·[U,1], .01,.99)`.
Take the weighted mean of the table and additive members using exponentiated
prequential log scores. If there are slow channels, compute a support deficit
`ρ` for pairs of current cues: for each such pair let `lo` be the largest
selected-on singleton count and `hi` the pair-cell count; take the maximum of
`(lo−hi)/(lo(1+hi))` and zero across eligible pairs (zero if the denominator
vanishes). Mix the member mean with `p_add` in proportions `1−ρ,ρ`. This is
**heuristic support-dependent shrinkage**, not a derived posterior probability
that the additive member is correct.

The background path changes what that member forecast means:

- **No slow channels:** use the member forecast; on an unused regime blend
  40% of the regime base rate into it.
- **Slow channels, quiet input:** use only place background `B`. The regime
  rate has one pseudocount at the pooled rate, and the place rate has one at
  that regime rate. With no place history, use the regime rate directly.
- **Slow channels, nonquiet input:** `p_local = B + (1−B) p_cue`; shrink
  `p_cue` by 40% on unused regimes. This is a noisy-OR-like parameterization:
  the cue term can add probability but cannot suppress `B`.

### 2.4 The competing process expert

Code: [decompose.py](../../development/models/decompose.py),
[online_hmm.py](../../development/models/online_hmm.py), and `hmm_*` in
[kernels.py](../../development/models/kernels.py). The actual path is
`Decomposer(..., windows=True, fast=False, conj=False)`. It starts with one
group per sensor. A singleton mixes `K=2,3` HMMs and one window member; a
merged group uses `K=2,3,4,6` and a window member. For HMM state `h`:

```text
pool[h,:] = normalize(Σ_a transition_counts[a,h,:] + .2 + 2 I[h,:])
T[a,h,:]  = (transition_counts[a,h,:] + 2 pool[h,:]) / (row_count + 2)
E[h,j]    = (emission_ones[h,j] + .5) / (emission_mass[h] + 1)
prior     = belief · T[a]
lik[h]    = Π_j Bern(next_bit[j]; E[h,j])
forecast  = prior · E
score     = log(prior · lik)
belief'   = normalize(prior × lik)
```

Initial belief normalizes the first observation's likelihood from a uniform
state prior; emission counts get random symmetry-breaking pseudocounts. For
learning, run backward messages over at most five buffered transitions with
the current parameters. On the oldest transition increment the transition
counts by `ξ(h,h') ∝ saved_belief[h] T[a,h,h'] lik[h'] β[h']`; increment
emission mass by `Σ_h ξ(h,h')` and emission ones by that mass times the
observed channel value. This fixed-lag, plug-in-parameter learning is an
**approximation** to integrating over the HMM parameters. Likelihoods use
thresholded bits, while the packed HMM's emission update accumulates raw
sensor values; they coincide for binary observations.

The window member predicts conditionally independent Bernoulli bits from
`(action, suffix of length ≤ 4)` using `(ones+1)/(n+2)`. It uses the shortest
available key until a longer key has at least eight observations, and updates
all available suffixes. A group's component weights are the softmax of
cumulative one-step joint log scores, with no best-size hard choice.

**Group discovery.** Lagged binary count tables on a rotating bounded
neighbor bank propose merges using the maximum Jeffreys-prior log score for
an edge versus no edge, over lags `0..4`, with a search cost
`log(max(1, bank_width × 5))`. That score schedules tests, not acceptance.
Every five steps consider proposals, at most six disjoint live candidates, up
to six channels per group. Compare a freshly learned merged model with an
equal mixture of two separate-model baselines, fresh and already warm:

```text
D = summed_joint_log_score − logsumexp(summed_fresh_sep, summed_warm_sep) + log 2.
bar = log(1 + bank_width × 5) + log(k) + log(1+k) + log(4).
```

`k` is the proposal count. Decide when `|D| > bar`; merge on the positive
side, otherwise cache the proposal score to raise the bar for future
screening of that group pair. No split follows an accepted merge in this
path. Both the sparse proposal screen and plug-in evidence comparison are
approximations, not a guarantee of discovering all interactions.

### 2.5 Join the experts; define the quiet clock

Per channel, local weight `λ_j = sigmoid(L_j)` starts at `1/2`; `L_j`
accumulates the local versus process expert's Bernoulli predictive log-score
difference on real fast bits (see §6). For component `(i,z)` use

```text
p[i,z,j] = λ_j p_local[i,z,j] + (1−λ_j) p_process[j].
```

**Approximation:** `p_process` contributes marginal probabilities, not its
joint HMM emission distribution. Fast bits form a product within each
component and a mixture across `(i,z)`. Slow bits use their own independent
Markov marginals. The policy does not separately draw a process state or a
family label for these bits.

For quiet input, write `I_m = Π_fast(1−p[m,j])`, `I = Σ_m w_m I_m` using
normalized weights, and `P_m(z)` for component `m`'s product law on a complete
fast pattern `z`. The actual joint clock is

```text
P(z = all zero) = I
P(z ≠ all zero) = (1−I) Σ_m w_m P_m(z)/(1−I_m).
```

Thus quiet timing is shared and the composition of a nonempty event is
regime-specific. This is **not** plain mixture conditioning: components
inside the nonempty event keep their original weights `w_m` rather than
weights proportional to `w_m(1−I_m)`. For an all-zero **retained subset**,
the exact marginal includes both the full idle mass and nonempty events on
omitted fast channels. [Projection](../../development/models/factor_projection.py)
adds these pieces explicitly. The [sampler](../../development/models/factor_reward.py)
uses a shared idle draw; in the nonidle case it chooses the first success
from the truncated product law, then draws the remaining bits. In the real
prediction, the fast marginals on quiet input are scaled per component by
`(1−I)/(1−I_m)` before mixing. Numerical denominators are guarded in code.

## 3. Controlled motion and reward: what the real transitions fit

Code: [factor_reward.py](../../development/models/factor_reward.py),
[factor_event_reward.py](../../development/models/factor_event_reward.py),
[factor_context_reward.py](../../development/models/factor_context_reward.py),
[factor_joint_reward.py](../../development/models/factor_joint_reward.py), and
[factor_regime_reward.py](../../development/models/factor_regime_reward.py).
The methods follow the effective `JointReward` inheritance order. In
particular, `DwellEventReward` inherits the transition prior and onset bound
of `ProvisionalEventReward`, **not** the similarly named `ElapsedEventReward`
alternatives. The successor bridge, `one_window`, and action-permutation
recall are inactive on this path.

### 3.1 Which marks are controllable?

For each bit track the next-bit counts conditioned on its present bit, both
with and without the action. Add the pre-update log likelihood ratio of the
action-specific to pooled Bernoulli predictors, both Jeffreys-smoothed. A bit
can occupy one of up to four slots only if that cumulative ratio exceeds
`log(1+J)`, the maximum action probability spread over the two current-bit
rows exceeds `0.5`, **and** the action spread in the current-on row exceeds
`0.5`. The latter is the explicit sustained-control test. Already recognized
roles survive a reward-mode switch (`known_mode_roles`). Otherwise retain
eligible installed marks and fill vacancies by score. This is a **predictive
screen**, not a formal causal identification result: the actions were chosen
by the collecting policy, and score and support vary with that policy.

`c(b) = Σ_d 2^d b[slot[d]]` with absent slots zero. For each real transition,
count `N[a,c(b),c(b')]` and update the arrival mass `M[c(b')]`. The deployed
transition forecast is

```text
base[c] = normalize((M[c] + .5) × valid[c])
T[a,s,c] = (N[a,s,c] + base[c]) / (Σ_d N[a,s,d] + 1).
```

A code is valid if it has no set bit in an absent slot. This shrinks sparse
rows toward a pooled arrival distribution, not toward the sensory forecast.

### 3.2 Reward-mode changes and memory

Before recording the current transition, compare its predictive probability
under the current transition row, a recent row, and the single archived row
(if any), using the current base above for each. For each alternative add
`log(P_alt/P_current)` to its running score and clip the running score below
at zero. The check requires a witnessed current row and a live control slot;
it is disabled during replay. Switch if the best score exceeds
`log(1 + steps_in_mode) + log(1 + number_of_valid_codes)`. Recent counts decay
by `1/2` on **the visited row only**, before adding the real arrival. A
switch recalls archived banks or zeros them for a new mode, archives the
outgoing banks, and clears the replay deque before retaining the current
transition. Arrival, reward, credited-feature, and onset banks switch
together. This is a sequential **heuristic**, driven by motion surprise;
reward changes alone are not the trigger.

The deque holds the last `2 × C × A` real transitions, with `C=2^min(J,4)`;
it stores old/new sensory bits, action, real utility, and real features and
onsets. When the control code changes, rebuild transition/reward statistics
from that window; on ordinary transitions those statistics accumulate beyond
the deque. Context tables normally accumulate as well; a change in anchor,
control code, or reward mode triggers their bounded reprojection. The joint
selector, by contrast, recomputes its current choice **only from the deque**
on every update. The histories are therefore not all of the same length.

### 3.3 Reward bins, representative utilities, and unary conditioning

Reward has `B=16` bins `k=min(15,floor(16r))` for `r ∈ [0,1]` (invalid rewards
are rejected). For a fixed arrival or subgroup record bin count `n_k` and
sum of **real amounts** `s_k`. Define `m_k=(k+.5)/B`, `α=1/B`, and

```text
p_k = (n_k + α)/(Σ n + 1)
u_k = (s_k + α m_k)/(n_k + α)
mean = Σ_k p_k u_k = (Σ s + .5)/(Σ n + 1).
```

This is a symmetric Dirichlet histogram with a smoothed within-bin amount.
The learned amount is neither a Bernoulli target nor just the bin midpoint.
The mean identity is **exact** for the represented law. Counts, aggregate
utility, and continuous representative values have different units.

Set `L=5`. Feature `(l,j)` is bit `j` at lag `l` relative to the **origin**
of the real transition, not the arrival. Its onset is that bit AND NOT the
same bit one step earlier (missing history is zero). Bin and feature counts
are scored **at the arrival code**. For each feature the pre-update log score
compares its Bernoulli likelihood given the realized reward bin with its
pooled likelihood, smoothing both with `(.5, .5)`. At most one feature per
arrival survives (`P=1`), excluding controlled coordinates, and only above
`log(1 + J L)`. Given credited bits, multiply the prior bin mass by their
Bernoulli likelihoods and normalize. Without a credited feature, the result
is the pooled law. This reverse conditioning `P(feature | bin)` is not
identical to fitting a separate forward reward histogram per bit value.

### 3.4 Contextual moderators and the unary-then-context fallback

Keep at most four noncontrolled anchors. Rank a bit by
`onset_intensity × (1 − smoothed prevalence) × sqrt(onset_count)`, where
`onset_intensity = onset_count/(past_on+.5)` and the prevalence here uses
`(past_on+.5)/(total+1)`. When an installed anchor is currently on, collect
potential moderator co-occurrences and compare their pre-update Bernoulli
likelihoods within the anchor with their pooled likelihoods. The relevant
moderator must be noncontrolled, different from the anchor, have conditional
on-rate above its pooled rate, and clear log score `log(1+J)`. Some users of
this bank (pair optimism and information proposals) additionally require
observed moderator-off support. Its eligibility is not identical everywhere.

For onset of an anchor, store `(count, sum, bin counts, bin sums)` for both
moderator values at that lag and arrival. The ordinary signed fallback
requires the anchor/lag to be the already credited unary feature. For two
nonempty groups with means `μ0,μ1` and total `n=n0+n1`, rank by excess of

```text
gain = n0 n1 / n × (μ0 − μ1)^2
cost = .25 log(1+n) + .25 log(number_of_eligible_moderators).
```

Accept a positive excess. If several fallback rules apply to a queried
feature vector, `context_moment` uses the largest raw gain among the
supported onsets. It replaces the bin law only **on** the selected onset;
otherwise it leaves the unary/default law. The gain is an explained
squared-error identity; the admission tariff is a **heuristic**.

### 3.5 v60's joint selector: one support, three competing partitions

This runs after real reward learning, even before a unary feature is
credited. `joint_basis` tests all eligible noncontrolled bit-lag candidates
as unary splits and eligible anchor/moderator pairs at the same lag as
compound splits. For each arrival, classify the **same** retained rewards:

| Hypothesis | How the window is divided |
| --- | --- |
| Pooled | All rows in one group |
| Unary | Feature bit off/on; require both groups |
| Compound | Anchor onset and moderator 0; anchor onset and moderator 1; no anchor onset (either moderator); require all three groups |

For a group's bin counts, let

```text
ell(n) = Σ_k [log Γ(n_k + 1/B) − log Γ(1/B)] − log Γ(1 + Σ_k n_k).
score(partition) = Σ_groups ell(n_group) − ell(n_pooled) − log(number of candidates in family).
```

This is the integrated probability of **the specified sequence** under a
categorical likelihood and symmetric Dirichlet prior, with no multinomial
coefficient. The evidence is exact under those assumptions on this retained
window; the window, screen, and family-size penalty are design choices, not
an exact posterior over all reward structures. Select a compound split only
if its score is strictly above both zero and the best penalized unary score.
`joint_edge` stores the excess above that threshold. Fewer than three rows
cannot support a compound. If the split stops qualifying, it disappears.

For a joint-selected arrival, `rule` marks the anchor as credited (affecting
exploration) and masks the fallback signed rules there. `context_moment`
uses the joint window's smoothed bin law in **all three** cells, including
the no-onset cell. Contrast this with §3.4. It also returns the cell ID and
its posterior mass `n+1` for the lookahead correction. No joint-selected
arrival means the inherited calculation remains active.

## 4. What gets treated as a promising experiment

Code: `provisional`, `pair_context`, `refine_root`, `posterior_option`, and
`discovery_value` in the [reward](../../development/models/factor_reward.py),
[context](../../development/models/factor_context_reward.py), and
[information](../../development/models/factor_information.py) modules.
These mechanisms have different input support and different mathematical
status. The action score is not an expected reward under one posterior.

### 4.1 Upper forecasts and matched local odds

For an eligible onset without a credited parent, the onset forecast is the
smoothed local mean `m=(sum+.5)/(count+1)` plus
`(1−m) × onset_intensity × (1−feature_prevalence)/sqrt(count+1)`.
Controls and arrivals with a credited feature are masked. The provision
also computes a variance `(sum_of_squares+1/3)/(count+1) − m²` floored at
zero; this rollout uses the **upper forecast**, not the variance.

The paired offer `pair_context` can refine a known law, but in v60 requires
the same credited anchor/lag. For the best association-supported moderator,
shrink the subgroup mean toward the combined onset mean with one
pseudocount. Add an upper term `(1−mean)/sqrt(count+1)` times anchor rarity
and excess moderator frequency normalized within its off/on branch. More
explicitly, with smoothed anchor-conditional moderator rate `p` and pooled
rate `p0`, that frequency factor is `(p−p0)/p` for moderator-off and
`(p−p0)/(1−p0)` for moderator-on, with guarded denominators. There is no
new lag-0 paired opportunity in the continuation forecast (`shift=True`).

Separately, at an arrival without a credited rule, `refine_root` can change
the *mean law* before hard admission. Candidates come from current anchor
onsets with `anchor_n ≥ 2`, using the best supported lag-0 moderator; v60
requires a pair for this refinement. Evaluate a proposed subset versus its
complement against the pooled histogram:

```text
odds = −log(number_of_anchor_slots) + ell(subset) + ell(complement) − ell(whole).
```

The implementation clamps negative bins in `whole−subset` to zero. Keep
the candidate of greatest odds at each arrival. Smoothed subset and current
pooled laws mix with weights `sigmoid(clip(odds,−30,30))` and its complement;
atom amounts mix by mass *within* each bin, not by unweighted amount. This
changes both the immediate mean and the first-reward law; it is **not** an
information bonus. At a refined arrival the root forecast takes that mean
rather than the competing upper offer.

### 4.2 A separate repeated-decision option

At a current anchor onset, an information candidate is a lag-0 moderated
subset or the pooled-on anchor subset. Its frequency proxy is the smoothed
onset frequency, multiplied by the conditional probability of the selected
moderator branch when paired. Skip unseen arrivals. If an arrival already
has a credited rule, v60 allows only a paired candidate matching its
credited current anchor; skip arrivals equally reachable under all actions
(to numerical tolerance).

Take input fixed forecast `q[a]` and `reach[a]=T[a,s,c]`. Let `λ` be the
same pooled/split odds weight, `p0,p1` the pooled/subset bin laws, `u_k` the
subset representatives, `μ0` the pooled real mean, and
`μ1_k=((1+subset_count) μ1 + u_k)/(2+subset_count)` the subgroup mean after
adding one hypothetical observation of representative amount `u_k`.
For each bin `k`, the mass-weighted next decision values are

```text
P_k = (1−λ)p0_k + λ p1_k
G[k,a] = P_k q[a] + λ p1_k (μ1_k−μ0) reach[a]
option = max(0, Σ_k max_a G[k,a] − max_a Σ_k G[k,a]).
```

The difference is **exact for this finite table**, and compares the same
future decision with versus without observing the bin. `λ p1_k` is the
joint mass of the split hypothesis and the bin, not a second copy of `P_k`.
This is not full Bayes-adaptive control: the fixed forecast is offset by a
local reward-mean change, not by refitting the sensory or transition model.

For repeated opportunities the implemented horizon is
`H=Σ_{k=1}^{2 replay_capacity} (1+steps_in_mode)/(1+steps_in_mode+k)`;
it is a truncated survival proxy from a Beta(1,1) hazard. Scale the option
by `H × frequency × (reach[a]−min_a reach[a])`. Use the **maximum** across
candidate subsets and arrivals, not a sum. This horizon, frequency transfer,
and choice of maximum are **heuristics**. The code does not promise to
observe those future opportunities.

## 5. The action value: one two-step calculation

Code: [provisional_values](../../development/models/factor_rollout.py) and
[factor projection](../../development/models/factor_projection.py). Everything
below is a query on frozen current parameters. There is no hypothetical
learning of particle weights, control transitions, actuator scope, or
reward structure, and no exploration RNG inside the query.

### 5.1 Root laws and branches

1. For every possible arrival `c`, evaluate the root's reward law using its
   present and lagged features: unary likelihood → any contextual or joint
   override → any soft root refinement. Denote its mean `m0[c]`, bin mass
   `p0[c,k]`, and representative amount `u0[c,k]`. Denote the group ID by
   `g0[c]` if a contextual law is in play. The optimistic immediate forecast
   `I[c]` is the maximum of the modeled mean and eligible upper offers,
   with the exclusions in §4.1; context offers are masked where an explicit
   group already applies. The first reward's **distribution** for the inner
   correction remains `p0,u0`, not the optimistic forecast.
2. Enumerate all controlled arrivals that are reachable under at least one
   action, with root probabilities `T[a,s,c]`. Independently form the
   residual sensory projection for that root action. Determine the
   lag-0 relevant columns from credited features and moderators of signed
   rules. If none are relevant, use a single placeholder branch of mass 1.
   If one to three are relevant, enumerate their exact marginal from §2.5.
   Otherwise draw five equally weighted full sensory patterns from §2.5.
   This last path is Monte Carlo; the same uniforms are shared across root
   actions. Even with exact marginalization, the use of `T` followed by a
   separate sensory projection is a **factorization approximation**.
3. Cross each residual branch with each `c`, and overwrite the controlled
   columns by `c`'s bit pattern. Shift the history by one to build the
   next features. Compute the new onset as `hypothetical_next_bit & ~b`
   and shift the existing onset history for older lags. Compute arrival
   reward means `m2[d]` for possible *second* arrivals `d` from these
   features, applying the same credited and signed rules. For exploration,
   an unobserved lag-0 onset is not credited with a speculative upper
   offer at the continuation; ordinary predicted reward at that onset
   can still matter. Older observed opportunities may raise the
   continuation mean vector to `cont[d]` by a pointwise maximum.

### 5.2 One-atom update, then choose the second action

Let `β` identify a residual sensory branch of mass `ωβ`, and fix its first
controlled arrival `c`. From the *unconditional-to-group* likelihood
calculation for the continuation, `L2[c,k]` is the likelihood of its active
features at reward bin `k` and `Z2[c]` the sum of that likelihood times
unnormalized bin counts (`(M[c]+1) × pooled_bin_mass`). Without a group
override, the effect on the continuation's mean for **the same** arrival
`c` of one first reward atom is

```text
Δ[c,k] = L2[c,k] / (Z2[c] + L2[c,k]) × (u0[c,k] − m2[c]).
```

This is the rank-one change in a normalized weighted mean when adding one
atom with its fixed likelihood weight. With a selected continuation group,
use `(u0[c,k]−m2[c])/(1+group_posterior_mass)` instead. If signed rules are in
play and the first and continuation group IDs for `c` differ, set the
correction to zero. `m2`, not the optimistic `cont`, is the mean being
updated. This update applies to exactly one reward type distribution; it is
not a virtual count update to all the candidate banks.

For continuation action `a'`, multiply means by its controlled transition
row `T[a',c,:]`. The optimistic branch also pushes this expected continuation
toward the best `cont` at a witnessed destination (or the maximum of all if
no destination is witnessed), with fraction `1/(2+N[a',c,:].sum())`. Witnessed here means positive
arrival mass anywhere in the active mode, not just in that transition row. Write
`Vβ,c(a')` for the result. For the first bin `k`, add the mean correction
weighted by `T[a',c,c]`. Then **clip to `[0,1]` and maximize over the second
action inside the expectation over the observed first bin and branch**:

```text
G[c] = Σ_β ωβ Σ_k p0[c,k] max_a' clip(Vβ,c(a') + T[a',c,c] Δ[c,k], 0, 1).
Q_optimistic[a] = Σ_c T[a,s,c] (I[c] + G[c]).
```

`G` depends on the root action through the sensory branches. The discount
is 1 for v60. Before returning `Q_optimistic`, the code applies the same
motion optimism at the root: pull toward the best `I[c]+G[c]` among
witnessed root arrivals (all enumerated arrivals if none is witnessed), with
fraction `1/(2+N[a,s,:].sum())`. The motion pulls and clipping are
**heuristics**; the inner order of averaging and maximizing is the exact
finite calculation for this decision table.

### 5.3 The final maximum and the real action

If no actuator slot is live, the separate information option is disabled.
Otherwise compute `Q_fixed` using the same branch probabilities and
reward-atom correction, but no upper offers and no motion optimism.
"Fixed" does **not** mean that the second action ignores the first reward:
the one-atom correction and max-inside-expectation still apply. The v60
path shares the identical sensory projection between these forecasts.

For each root action use `Q[a] = max(Q_optimistic[a], Q_fixed[a]+option[a])`,
where `option[a]` is §4.2's repeated-decision value. This is a pointwise
choice between two forecasts, not an addition of the bonus on top of the
optimistic forecast. The controller then chooses a maximizing `a`, runs the
real `predict(x,a)` to cache the sensory calculation, and waits for `y`.

## 6. Close the loop: what `learn(y)` actually changes

Code: `FactorModel.learn` and the numerical kernels in
[factor.py](../../development/models/factor.py) and
[kernels.py](../../development/models/kernels.py). The order matters.

1. **Reward first.** Record `(b, a, r, b')` on the old retained history.
   Re-evaluate the control slots, mode, anchors and joint split as detailed
   in §3. The planner's hypothetical branches did none of these things.
2. **Sensory roles and competing expert.** Update the tonic/phasic test
   with `b` following the previous real input. Clear parent positions
   pointing to newly slow bits and reset the corresponding statistics.
   Use the real cached prediction if its bit/action key and roles still
   match; otherwise prepare again. Update the group expert and the
   two-row slow counts (tracked for all sensors). Append `b` to bounded
   history and clear the real cache. If the slow-bit place signatures of
   `b` and `b'` differ, stop the contextual update here. Reward and
   process expert have already learned this transition.
3. **Expert weights.** On fast bits with no slow-context silence, form the
   local expert's marginal `q_local` by averaging the local forecast over
   particle/regime weights, and add
   `log Bern(b';q_local) − log Bern(b';p_process)` to the per-channel log
   odds with probability floor `1e-12`. This is a comparison of forecast
   scores before assimilation, not another count in either expert.
4. **Regime filtering.** In each particle use the **local** emission law,
   not the process-mixed law, to update the prior on `z`: multiply by the
   fast Bernoulli product and by the nonempty clock correction if
   applicable. For a quiet-to-silent event the common clock cancels, so
   keep the prior and leave particle log weights unchanged. Otherwise add
   the log normalizer of the per-particle filter to its log weight.
   Sample one `z` per particle from its normalized posterior; record
   regime entries and births. A posterior used in place memory has unused
   regimes masked out after the sample and is stored only at input/output
   marks or on the first visit. Between silent steps the last mark anchors
   the next recurrence calculation rather than creating repeated occupancy
   counts. This combines finite filtering with **sampled** assignments.
5. **Contextual sufficient statistics.** With slow channels and quiet
   input, count place background instead of updating cue members. Update
   regime base rates on quiet steps or in the absence of slow channels;
   cache active cue tokens on nonquiet scoped steps. To update cue models
   at other steps, let `obs` be the sampled regime's local probability and
   `B` its background. For a positive bit give the cue fraction
   `max(0,obs−B)/max(1e-12,obs)`; for a negative bit give weight 1. This is
   **fractional credit**, not exact integration over all assignments.
   Score members with pre-update log probabilities including background.
   With deficit `ρ>0`, table counts get the additional responsibility
   `(1−ρ) L_table / ((1−ρ)L_table+ρ L_add)` for the observed bit; score
   that same combined likelihood. Pooled counts receive particle-weighted
   cue credit and their own pooled pre-update log score.
6. **Additive member and resampling.** With feature vector `f=[U,1]`,
   covariance `S`, and credit `γ>0`, the Gaussian rank-one update uses
   `K=S f/(fᵀ S f + .25/γ)`, `μ ← μ + K × residual`, and
   `S ← S − K (S f)ᵀ`. For a particle the residual removes both the old
   shared and particle forecasts; for the shared bank it removes only the
   old shared forecast, and its credit is the unweighted particle mean.
   This Gaussian update is a surrogate for binary outcomes; the forecast
   is clipped and scored as Bernoulli, not as Gaussian. Resample
   systematically when the normalized particle weights obey
   `ESS=1/Σ_i w_i² < N/2`; draw a single offset and copy ancestors and
   their place memory. Reset log weights after resampling.
7. **Discover the next local feature.** On visited source/lag candidates,
   accumulate `n, Σx, Σe, Σxe` using residual `e = b'−predicted_marginal`.
   The score is `(Σxe−ΣxΣe/n)^2/(1 + Σx(1−Σx/n))`. Install at most one
   change per target per step: the new score must exceed `.25` and the
   weakest installed score by `.25`. Exclude slow sources, own lags
   already included, and other lags of already installed sources.
   A replaced parent's cells and additive coordinate are reset, but its
   table evidence backs off to the subconfiguration without that parent.
   The bounded source bank rotates unpinned entries every `J` steps and
   clears their exposure statistics. This is residual screening, not a
   conditional independence test with a calibrated significance level.

`FactorController.learn` then clears its pending transition. The updated
statistics, not the hypothetical branches, determine the next action.

## 7. Two small hand checks

### 7.1 Same rows, different partitions

At one arrival imagine four identical observed rewards in each of the
three groups, in three distinct bins. Suppose two unary candidates both
isolate one of the pure groups, and there is one compound candidate that
separates all three. All fits have the same 12 rows. Omitting zero bin
entries in `ell`, §3.5 gives

```text
pooled ell(4,4,4)                       = −22.593656
best unary ell(4,4) + ell(4)            = −16.389098
compound 3 ell(4)                      = −12.140603
unary score (including log 2)           =   5.511411
compound score (including log 1)       =  10.453053
compound excess over best unary        =   4.941642
```

The compound wins; it would be ineligible if any of its three groups were
empty. If one pure group contains four exact rewards `0.9` (bin 14), its
bin-14 mass is `4.0625/5 = .8125`, its bin-14 representative is
`(3.6 + .0625 × 14.5/16)/4.0625 ≈ .900096`, and its total fitted mean
across all bins is `(3.6+.5)/5 = .82`. The selection score uses counts; the
utility calculation also uses amounts. The
[joint reward tests](../../development/tests/test_factor_joint_reward.py)
check this distinction against independently computed scores and atoms on
an asymmetric fixture, including the no-onset group.

### 7.2 Maximize after seeing the first outcome

Suppress optimistic terms and compare just the continuation: the immediate
term is the same under both second-step policies. Take one first arrival,
one residual branch and two equally likely first reward bins. Suppose
continuation values over two actions are `(.9,.1)` for one bin and
`(.1,.9)` for the other. Then the continuation is `.5×.9+.5×.9=.9`, **not**
`max(.5,.5)=.5`. The latter describes a different policy that commits to the
second action before it observes the outcome. The difference `.4` is
already in the `.9`; adding it again would double-count this two-step
information. The deployed separate option has a different horizon proxy
and is combined by a maximum, as in §5.3. The
[exact two-step fixture](../../development/research/ar5_voi_fixture.py) and
[control tests](../../development/tests/test_factor_control.py) exercise
this order of operations and a separate one-atom correction example.

## 8. Where I would audit first

These are properties and questions exposed by the calculation, not
conclusions that dropping a component would improve v60. Start with
counterexamples and frozen inputs, not another benchmark cohort.

| Question | Why the code makes it worth checking | Small discriminating check |
| --- | --- | --- |
| Is the same latent dependence preserved from prediction to planning? | The process model fits joint group likelihoods, but the planner receives only its marginals; it then overwrites controlled bits after projecting residual bits. Exact marginalization of that projection cannot restore discarded dependence. | Two bits with identical marginals but opposite coupling; compare the branch law and action ordering against an explicit joint table. |
| Is silence treated consistently wherever its likelihood is used? | Quiet timing in projection uses the mixed forecast; regime learning uses the contextual forecast, and the expert comparison uses the local marginal without the quiet correction (and skips quiet steps with slow context). These are distinct probability calculations, not one shared posterior. | A two-regime quiet fixture with different idle probabilities; tabulate forecast, regime likelihood and expert score separately. |
| Is the first reward being credited to the same conditional law on both steps? | Hard contextual changes are gated by group IDs, but a soft root refinement changes `p0,u0` without creating a group ID or applying the same refinement to `m2`. The rank-one forecast is not automatically the result of refitting that soft split. | Compare a soft-refined root and unchanged continuation against a one-row refit of both pooled and split laws. Also verify that the fallback group identifier, which omits moderator identity, remains unambiguous within a fixed lookup; the selector chooses one moderator per arrival, anchor and lag. |
| Are comparisons always made on comparable histories? | The joint score is common-support, but the old unary law, signed context, active/archived mode banks, and soft split odds do not all have the same reset policy. `max(0, whole−subset)` can mask a noncontained subset in the odds calculation. | After a control-scope change or archive recall, record the provenance and binwise counts of `whole` and `subset`; require actual containment before interpreting the odds as a Bayes factor. |
| Which exploration term actually determines an action? | Upper onset offers, motion optimism, the one-atom update and the repeated option enter at different points. The final pointwise maximum can pick a different forecast for each action. The resulting score is not a two-step reward bounded by 2 once the repeated option is included. | On one frozen transition report `Q_fixed`, the upper/optimistic increments, the separate option, and the winning branch per action; then remove terms one at a time in a scratch calculation. |
| Is a winning structure stable to a single new row? | Four anchors, one credited unary, hard tonic labels, strict joint admission, and one archived mode all introduce hard boundaries. The joint winner can vanish as rows age out even with unchanged long-history unary evidence. | Hold the real input fixed and vary one supporting row or one tie; compare `rule`, projected values and action margin. |
| Are there redundant rules, or just similarly named rules? | The model has both onset optimism and posterior options, both sampled sensory regimes and HMM groups, and both recurrent sensory place memory and reward-mode recall. They are not mathematically interchangeable, but overlapping effects are possible. | Trace a failure from output back to the first changed quantity; test removal or replacement at that source, not by adding a compensating layer. |

### What evidence does and does not establish

The [AR-7 report](AR7.md) and [registered checks](../../development/results/ar7/assurance-v60/registered-checks.json)
provide historical parity, planning isolation, and task evidence. The
[read-only verdict](../../development/results/ar7/dev20-v60/DEV.json)
checks normalized returns `(agent−stay)/(oracle−stay)` on 15 variants,
calculates the mean paired difference per seed block, then takes a 99%
t interval across 20 blocks. None of that evaluation mathematics feeds
`step` or `learn`. It does not validate every assumption in the audit table,
prove component necessity, or establish transfer to unopened T4.
