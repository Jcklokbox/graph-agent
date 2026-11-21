# v60: what survives difficulty, and what to simplify first

This is a read-only analysis of the accepted v60, using the
[computation map](V60_MATH.md), [AR-7 report](AR7.md), and retained data.
There are three different levels of evidence here: a controlled intervention
on an exposed tape, a descriptive comparison of closed-loop results, and
an implication of the equations. None is a new qualification.

## 1. Difficulty has different causes

### A stable rule can be hard to see without being hard to learn anew

Recomputed from the [paired episode journal](../../development/results/ar7/dev20-v60/episodes.jsonl).
Scores are the means of per-episode `(brain−stay)/(oracle−stay)`; raw delta
is the mean paired difference in `brain` before normalization. Wins count
seeds on which v60 beat v12 within that cell, not independent tests. Values
within `1e-12` are treated as tied for these counts.

| Task | v12 score | v60 score | Paired score delta | Raw delta | v60 wins / 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aut-fooddelay2` | .8679 | .9589 | +.0911 | +.00838 | 15 |
| `aut-fooddelay3-noise1` | .4120 | .8975 | +.4854 | +.03700 | 19 |
| `dist` | .9419 | .8756 | −.0663 | −.00489 | 1 |
| `dist-delay3` | .9805 | .8742 | −.1063 | −.00766 | 1 |
| `dist-delay3-noise2` | .7082 | .9654 | +.2572 | +.01884 | 20 |
| `aut-serial300` | .8998 | .8231 | −.0767 | −.00852 | 0 |
| `canc` | .9803 | .9429 | −.0374 | −.00162 | 3 |
| `canc-noise3` | .8794 | .8757 | −.0037 | .00000 | 9 |

Two delayed/noisy cells contribute **99.6% of the net normalized advantage**
(the other 13 average +.00024). The largest gains survive removing the
normalization. But delay alone does not explain the result. On the two
`dist-delay3` variants, the oracle and stay returns agree seed by seed.
Adding the two distractor channels changes the *raw* mean by **−.01995** for
v12 and **+.00655** for v60. v60 improves on 9 seeds, worsens on 5, and ties
on 6 across those two variants. This is a comparison of realized policies,
not evidence that irrelevant channels are intrinsically useful. In
particular, it does not isolate sensory inference, exploration, or joint
reward selection. The model can handle a delayed opportunity while still
underperforming on a clean version of it. A two-step planner can exploit a
cue older than two steps once its lag is represented in the reward state;
the planning horizon and the evidence-history horizon are different.
Before extending the rollout for a missed late reward, check whether its
lag enters the credited law, whether the relevant arrival remains reachable,
and whether the agent ever sees the payoff under that combination.

### The largest distal-delay loss happens before cue interpretation

The archived `dist-delay3` block `1905953054` contributes `.122` of the total
`.15325` raw difference summed over 20 pairs (79.6%). Its archived v60 score
was `.450`, versus `.572` for v12; the remaining pairs average a `.00164`
raw disadvantage for v60. This is a reason to diagnose that block, not to
claim its mechanism explains all 19 losses.

On a 160-step prefix with the current v60 and the registered seed offsets,
its only observation of the right side was the initial state. It saw no
cue and got no food. At decision 40, the center row had recorded action
counts `[2,29,0]`, the right bit was not in the controlled code, and no
reward feature was credited. The untried right action had lower values than
staying both in the fixed forecast (`.8153` vs `.8976`) and in policy Q
(`.8589` vs `.8978`). By decision 159 the right action still had not been
tried at that origin. There was no lag-zero reward cue to reweight, so the
arrival/cue coupling diagnostic was not applicable. On the noisy sibling
with the same seed, v60 saw 23 cues and got 15 food outcomes in 160 steps;
its scored lagged feature did not require a new lag-zero projection at the
sampled decisions either.

A single forced right move at decision 40, with all other decisions free,
produced 19 cue sightings and 10 food outcomes. Hiding only the cue on that
first arrival, while delivering the same position and payoff, still produced
18 later cue sightings and 11 food outcomes. Thus one useful contact was
sufficient on this prefix; neither treatment forced a food outcome or needed
to deliver an early reward cue at the contact. This does not isolate the
relative contributions of motion, occupancy and payoff updates during or
after the visit.

| 160-step prefix | Rest of actions | Mean reward over last 80 |
| --- | --- | ---: |
| Clean, no intervention | Free | .4500 |
| Clean, one forced right move | Free after step 40 | .5306 |
| Clean, same move and first cue masked | Free after step 40 | .5331 |

As a separate, task-blind control, prioritize a never-tried action when an
active control origin already has some recorded moves; use the existing Q
to rank unseen actions. This rule uses neither a cue nor a task-specific
step number. It is a coverage ablation, not a proposed default:

| Case | Free v60, last 80 | Coverage control, last 80 | Changed actions (160 steps) |
| --- | ---: | ---: | ---: |
| Clean distal, `1905953054` | .4500 | .5506 | 2 |
| Noisy distal, `1905953054` | .5025 | .5569 | 2 |
| Clean distal, `1774893116` | .5569 | .5556 | 1 |
| Noisy cancellation, `15` | .5125 | .5375 | 3 |

The short ablation supports testing coverage separately from reward and
sensory prediction. An empty row means a transition prior, not counterevidence
against an action. It does not justify a permanent exploration policy, in
particular around control-code changes and regime resets that clear counts.
The nominal RNG choice is always drawn before any override; the chosen
intervention is used by the real prediction cache and learning update.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.distal_contact_probe
```

### A mode reset does not necessarily make repeated trials wasteful

The [mode coverage probe](../../development/eval/scripts/mode_coverage_probe.py)
forks the same agent and environment just before a reversal, then changes
only the exploration choice. Compare no override, coverage from the active
transition counts, and coverage from real witnesses in `recent_trans`.
The last bank survives a mode reset, but not a controlled-code revision.
Both coverage arms require at least one active visit at the origin and rank
untried moves by the existing Q; the witness arm additionally excludes
moves already tried in the current code even if the active bank was cleared.

| 80-step evaluation window | No override | Active coverage | Witness coverage | Repeated old trials in active arm |
| --- | ---: | ---: | ---: | ---: |
| First serial reversal, seed `6` | .38125 | .43125 | .38125 | 6 |
| First serial reversal, seed `1822923777` | .38750 | .39375 | .38750 | 5 |
| Return to the old serial regime, seed `6` | .58750 | .58750 | .58750 | 0 |
| Stationary `aut`, deliberate empty-bank reset, seed `6` | .44375 | .46875 | .44375 | 6 |
| Same stationary checkpoint, no reset | .61250 | .61250 | .61250 | 0 |

The two first serial reversals produced new mode banks on steps 303 and 305.
Rechecks in the active arm were not wasted on these prefixes: they improved
recovery, while retaining the witness mask left the policy unchanged. At the
return reversal (step 600), the archive was recalled at 602, already carrying
transition support, and neither coverage arm took over. After the deliberate
reset in a stationary world, active coverage brought archive recall forward
from 365 to 342 but did not undo the whole cost of discarding a useful bank.

At the problematic distal seed, both policies still make the same two early
overrides and recover the `.5506` last-80 mean. There is a real front-loaded
cost there: the first 30 steps average `.2833` rather than `.3300`.
The evidence therefore favors testing **support in the active model**, not
prohibiting rechecks of every historically tried action. It does not justify
forcing coverage by default: the prefixes were short and exposed, and the
initial trial cost and costs under code revisions remain relevant. This is
an intervention audit, not a qualification or an agent update.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.mode_coverage_probe
```

### Waiting for a revisit after recoding does not remove the cost of probing

The [recode probe](../../development/eval/scripts/recode_coverage_probe.py)
compares seven 160-step dev prefixes from identical starting states. All
have real control-code revisions and none has a mode switch. The new arm
uses active-bank coverage but waits for one live departure from the same
origin after each recode. The preceding transition and rows reprojected
from bounded replay do not satisfy the wait. A mode switch clears this
restriction: rebuilding the code is not the same as learning new dynamics.

| Prefix and seed | First 30: off / immediate / wait | Last 80: off / immediate / wait |
| --- | ---: | ---: |
| `dist-delay3`, `1905953054` | .3300 / .2833 / .2717 | .4500 / .5506 / .5556 |
| `dist-delay3-noise2`, `1905953054` | .3700 / .3550 / .3667 | .5025 / .5569 / .5563 |
| `dist-delay3`, `1774893116` | .3000 / .2850 / .3000 | .5569 / .5556 / .5569 |
| `canc-noise3`, `15` | .2833 / .1667 / .1833 | .5125 / .5375 / .5375 |
| `aut`, `6` | .2500 / .2500 / .2500 | .6125 / .6125 / .6125 |
| `aut-noise4-sticky`, `6` | .2833 / .2500 / .2000 | .6125 / .6125 / .6125 |
| `dist-noise2-sticky`, `6` | .3883 / .3733 / .3883 | .5619 / .5619 / .5619 |

Nine of the immediate arm's 12 action overrides occurred on the first
live departure from that origin in the new code. Waiting prevented all such overrides
and eliminated the one on the already-working clean distal seed; that run
matched the unmodified action and reward path (no overrides). However,
the wait arm still changed 12 actions overall. On the diagnosed failure
it moved the two probes from steps 12/14 to 13/15 and **increased** the
early cost while preserving the late recovery. It also worsened the early
return on sticky `aut` without gaining anything in the last 80 steps.
The changed trajectories can cause different later recodes, so counts are
closed-loop outcomes, not a matched per-action causal estimate.

Do not add a revisit gate as a general cost control: it delays probes but
does not bound their number or opportunity cost. Nor is an immediate
obligatory probe justified by these exposed prefixes. A more useful next
comparison would bound actual overrides and measure cumulative reward,
while retaining the distinction between recoding and a detected mode change.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.recode_coverage_probe
```

### A hard probe budget caps interventions, not their reward cost

The [budget probe](../../development/eval/scripts/budget_coverage_probe.py)
tests one or two credits on the active transition bank. Charge a credit
only when the proposed action differs from the nominal v60 choice. A code
revision or archive recall never refills credits; a newly detected mode
does. The bound is per refill interval, not over the lifetime of the agent.
The 80-step serial forks start with zero credits, so they measure rechecks
triggered by mode detection, not new startup credits at the fork.

| Window | No probe | One credit | Two credits | Unlimited |
| --- | ---: | ---: | ---: | ---: |
| Clean distal failure, 160 steps | 68.40 | 79.05 | 78.10 | 78.10 |
| Noisy distal sibling, 160 steps | 76.25 | 80.65 | 80.65 | 80.65 |
| Working clean distal, 160 steps | 79.05 | 78.45 | 78.45 | 78.45 |
| First serial reversal, seed `6`, 80 steps | 30.50 | 30.00 | 30.00 | 34.50 |
| First serial reversal, seed `1822923777`, 80 steps | 31.00 | 33.50 | 33.50 | 31.50 |
| Archive return, seed `6`, 80 steps | 47.00 | 47.00 | 47.00 | 47.00 |

The table reports cumulative raw reward. One intervention was sufficient
for the original distal failure, but allowing only the first one or two
rechecks after the first serial reversal eliminated the recovery benefit
seen with six unrestricted rechecks. The end result of forcing only the early
rechecks is not an estimate of the marginal value of later ones, because
the trajectories after the cap are different. On the other serial seed,
the cap instead improved on unrestricted probing. Archive recall took no
extra actions in any arm.

The separate dev check fixed the two-credit arm before the comparison:
seeds `26` and `91`, four tasks, 160 steps each. The cumulative differences
versus unmodified v60 were:

| Task | Seed `26` | Seed `91` |
| --- | ---: | ---: |
| `dist-delay3` | +4.60 | -.15 |
| `dist-delay3-noise2` | -.05 | +3.55 |
| `aut-noise4-sticky` | +4.50 | +2.00 |
| `canc-noise3` | +1.50 | -3.50 |

That is five improvements and three regressions, with `+12.45` total over
1,280 paired steps (about `+.0097` mean raw reward per step). The early
cost still matters: on sticky `aut` seed `26`, two trials lower the first
30-step mean from `.3500` to `.2333` even though the full window gains.
All 28 budgeted paths obeyed their caps, with no warm-fork override before
a fresh mode switch. This is evidence for targeted startup exploration,
not for a shared default budget at startup and mode changes. Keep this
policy offline: spending fewer credits did not reliably avoid an early
reward loss, and the serial results do not support a uniform refill rule.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.budget_coverage_probe screen
python -m eval.scripts.budget_coverage_probe check
python -m eval.scripts.budget_coverage_probe modes
```

### Exact reuse in the live forecast

The default planner now separates the conditional value of a branch from
its probability under the first action. For zero to three current sensory
parents, every first action has the same ordered pattern support and the
same union of reachable controlled arrivals. For a given pattern and arrival,
reward likelihood, posterior correction and the best second action do not
read the first action. Build those conditional tables once, not `A` times.
Keep each action's sensory masses, transition row and optimism adjustment
separate. Falls back to per-action branch evaluation for sampled draws.
Reward bins, sensory patterns, and controlled states are contracted in the
same order as before, rather than replacing the per-action reductions with
one large matrix product.

The particle read likewise evaluates the never-used row once per action,
not once per eligible particle. Counts and local coefficients in that row
are zero, member weights are one, and its shared inputs are identical.
Slot reset and ancestor copy preserve the invariant. Particle-specific
prior weights, rates and place effects still apply after this table read.
Neither reuse changes learning, chooses approximations, or removes a
model component.

Checks relative to the local v60 just before reuse:

- Rollout only: on nine exposed dev prefixes, 1,976 policy vectors and
  124 separately requested fixed vectors matched byte for byte. Median
  planning times on those prefixes fell by about 7–23%.
- Both changes together: paired cold starts, 700 full brain steps over
  three prefixes, same Q bytes, actions, observed vectors and final sensory
  bank bytes checked. This includes a detected serial reversal. Alternating
  execution order, excluding the first 30 steps, gave the following local
  medians (not a cross-machine throughput claim):

| Case | Previous `step` + `learn`, ms | Shared forecast, ms |
| --- | ---: | ---: |
| `dist-delay3`, seed `1905953054` | 17.50 | 13.20 |
| `canc-noise3`, seed `15` | 24.80 | 21.21 |
| `aut-serial300`, seed `6` | 18.73 | 15.96 |

The §9.1 hashes and 13 fixed joint tape checkpoints stayed exact. Unit tests
cover zero and one exact sensory parent, the sampled fallback, and mixed
occupied/unoccupied particle rows. No qualification split was opened.

### Cutting the process expert or the particle inventory

This [component screen](../../development/eval/scripts/v60_simplification_screen.py)
uses the current v60 with no action overrides. It compares eliminating the
entire parallel grouped HMM/window expert, or retaining the architecture
but reducing the sensory population from 200 to 32. These are separate arms;
neither is enabled in default v60. The second arm is a resource reduction,
not a removal of a model mechanism.

Across nine short exposed dev cases (160 to 380 steps), 32 particles matched
all nine cumulative returns. The separate four-case check also matched the
action hashes, but longer runs did not. In 680 steps with two reversals it
scored `373.5` versus `372.5`, with two detected switches in either arm. On
400 sticky `aut` steps both scored `226.0` despite different action hashes.
The named particle banks used `13,237,760` instead of `82,736,000` bytes at
11 channels and three actions (84% reduction for those arrays, not a process
RSS measurement). A fixed 96-transition tape at each of seeds `7` and `29`
found Q differences at 74 steps, with a maximum about `1.25e-4`.

Removing the process expert was not equivalent. The same forced tape had
Q differences as large as `.1634`. On sticky `aut`, seed `26`, the cut gained
`1.0` cumulatively over 160 steps but fell from `.6000` to `.55625` over the
last 80. On the separate serial seed `1822923777`, the 380-step cumulative
return fell from `198.5` to `197.5`. Do not infer safety of deletion from
matching rewards on short paths where the action score may barely depend
on the sensory forecast. Keep the expert and default particle inventory;
the smaller population merits further behavioral testing as a separate arm.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.v60_simplification_screen
python -m eval.scripts.v60_simplification_screen --check
python -m eval.scripts.v60_simplification_screen --stress
python -m eval.scripts.v60_simplification_screen --numeric
```

### Seeing a reward split and getting to see its evidence are different jobs

The [exposed intervention fixture](../../development/results/ar7/joint-fixture.json)
compares the fixed action values after actual contacts against the independent
batch oracle. It admits a compound on **10 of 54 probes**, all in same-side
intervention arms (10/18 versus 0/36 across opposite-side and neutral arms).
For seed 15 in each of the clean and noisy variants, each of the four
same-side probes admits the split; the sign of `Q(0)−Q(1)` follows
cancellation versus noncancellation. For seed 6 the sign can also follow
cancellation at a probe **without** compound admission. Admission alone is
therefore neither a general prerequisite for the sign nor a guarantee of
better closed-loop returns. The arms are constructed probes, not an estimate
of how often the selector fires under free policy.

This is the most direct mechanism evidence available: once positive and
adverse outcomes are observed in the right scope, a common-support split
can distinguish them without changing the sensory model. A binary choice
between reward representations can matter even though it activates at few
fixture checkpoints. Correspondingly, the exposed v60/v59 comparison records
two improvements and 21 ties, not a broad shift. The larger fresh v60/v12
gains above **cannot** be attributed to this selector without a matched
comparison. Keep two different failure labels: **missing real evidence**
(the policy never reaches the informative outcome) versus **poor use of
available evidence** (the same logged outcomes imply a better reward law).

### A changed rule is a different failure from obscured evidence

At serial reversal the action/precursor mapping actually changes. v60 loses
to v12 on **all 20** registered seed blocks in that cell. This is a stronger
directional signal than a single low score, but the episode journal lacks
per-reversal traces. The
[mode detector](../../development/models/factor_regime_reward.py) looks at
surprise in controlled transitions, decays the visited row only, and has
one archived mode. It does not directly test for a change in reward or a
change in the sensory-to-payoff relation that leaves those rows consistent.
That is a specific visibility limit, **not** an established explanation of
the serial result. Trace the first few transitions across an already exposed
boundary: were slots valid, did the visited row register surprise, when
was the switch, which mode was recalled, and when did the action margin
recover? A late detector, a wrong recalled mode, and a correct transition
forecast paired with stale reward predictions require different fixes.

**Short check, already exposed seed 6.** In the first 320 steps of
`aut-serial300` (original DEV environment and agent seeds for block 6),
the control slots were unchanged across the first reversal. The boundary is
transition 300 (zero-based); the reward head switched on transition 303,
and still received zero utility on transition 310. That rules out "no
switch" as the explanation for this particular early prefix. It does not
establish whether the policy, re-estimation after the switch, or later
recall drives the aggregate deficit. The next trace should include the
post-switch action margin and the first return to a previously visited
mode, not just count switches.

## 2. A law worth testing: preserve the information that changes a choice

v60 already uses this principle in two working places: the joint selector
keeps a moderator only if it helps on the same real reward support, and the
planner enumerates an exact sensory marginal when only a few columns are
relevant. The boundary between those components is less selective: it
passes per-channel marginals from the process expert and combines residual
sensory branches independently with the separately predicted controlled
arrival. Arbitrary extra correlation is not the goal. Preserve a dependence
only when it can change the ranking of actions at fixed support.

### What the coupling check found

In a constructed state, the two actions have the same marginal probabilities
of a controlled bit `c` and a cue `z`, but the bits agree after one action and
disagree after the other. The fixed v60 rollout gives `Q=[1, 1]`. Reweighting
its existing branches by the observed dependence (with smoothed odds) gives
`[1.343, .657]`. This checks an error the interface can make, not its incidence
on DEV.

The [diagnostic](../../development/research/cue_coupling.py) independently
enumerates the deployed fixed values on exact sensory branches. For each
root action, freeze the utility surface `V[pattern,c]`, arrival probabilities
`T[c]`, and residual sensory pattern probabilities. To vary one cue `z`, keep
the other bits' distribution conditional on `z`. Let `v[z,c]` average `V`
over that conditional distribution. The joint mass `w[c]=P(z=1,c)` must
satisfy `0<=w[c]<=T[c]` and `sum(w)=P(z=1)`. Then the new score is

```text
T @ v[0] + w @ (v[1] - v[0]).
```

Filling arrival capacities in increasing or decreasing order of the
contrast `v[1]-v[0]` gives exact bounds. For all patterns simultaneously,
the diagnostic uses conservative bounds instead: subtract a reference
pattern's payoff (whose contribution is fixed by `T`), and bound each
pattern's residual by its minimum and maximum across reachable arrivals.
Intersect the bounds over reference patterns. A positive gap between the
winner's lower bound and every competitor's upper bound rules out a rank
change under *any* joint law on the same support and with the same marginals.

On four exposed, forced cancellation tapes (17 queries), the independent
branch enumeration agreed with the current fixed v60 readout within `2e-14`.
Across 26 single-cue checks, the largest possible change in any one Q was
`.000337`. Even allowing all relevant cues to couple jointly, the smallest
certified gap to the runner-up was `.3115`: no query could reverse rank.
Additionally, no action/cue query met the matched replay support criterion
(two witnessed arrivals with at least two visits each, and both cue values
observed, at the same origin bits and action). An unchanged empirical score
in this case is not evidence that the cue and arrival are independent.

These checks cover fixed values, not the optimistic policy or closed-loop
return. Do not add a joint bank on the strength of cancellation tapes. Look
for a decision where the bounds actually admit a rank reversal **and**
matched draws exist before proposing a live change.

```sh
# From development/ with the usual dependencies
python -m eval.scripts.cue_coupling_probe
```

## 3. Where it actually makes sense to save resources

### Implemented local simplifications

The current implementation no longer accumulates, resets, or archives the
second moment of onset rewards: the rollout used the upper forecast but
ignored the variance. The first moment and all reward-bin statistics remain.
The planner also takes the maximum over continuation actions before clipping
that maximum to `[0,1]`, rather than clipping every candidate. The standalone
[specification](V60_SPEC.md) reflects both changes.

A before/after comparison matched all 192 actions and both binary64 Q hashes
in §9.1. It also matched fixed and policy Q values and retained reward state
at all 13 queries on three exposed forced cancellation tapes, five with an
active joint law. A focused test checks that a regime recall restores onset
counts, amounts, and the upper forecast. Timings did not establish a speedup.
These are local conformance checks, not a new gate or a new accepted freeze.
The [reward partition experiment](REWARD_PARTITIONS.md) remains separate from
the default controller.

### Action reward with an unchanged context

The original arrival-only reward law gave equal values to actions when the
sensory vector stayed constant, even if their realized rewards differed.
The current controller can select an action partition in this scope. It
uses the existing replay and compares a pooled histogram with separate
histograms on the same rows. Replay must match the current bits and lag
features, all onsets must be zero, there must be no live control slot, and
all actions must have repeated contacts. Changed arrivals are not evidence
for this split. The same one-atom update as in contextual readout affects
only the action just taken, before comparing the next actions.

The [paired closed-loop probe](../../development/eval/scripts/v60_action_reward_probe.py)
uses five constant sensor bits, two actions paying `.9` or `.1`, and three
seeds. Its control disables only this split:

| Case | Steps per seed | Control mean | Action split mean |
| --- | ---: | ---: | ---: |
| Fixed better action | 120 | .473 | .873 |
| Better action reverses every 200 steps | 800 | .495 | .819 |
| First 50 steps after each reversal | 150 in total | .502 | .568 |

A worse post-switch segment remains: `.388` versus `.532` on seed 1 after
the third reversal. This is not a change detector. With missing recent
contacts the split can drop and the legacy calculation resumes; that is
not a guarantee of prompt exploration or recovery. The two §9.1 tapes
still match their 192 actions and both Q hashes; the 13 exposed cancellation
queries still match fixed and policy values. These are scoped comparisons,
not the qualification of a new freeze.

Run from `development/` with the regular dependencies:

```sh
python -m eval.scripts.v60_action_reward_probe
```

### Larger savings are still hypotheses

The [registered warm profile](../../development/results/ar7/assurance-v60/assurance.json)
uses two synthetic sensory worlds, not a cancellation workload. At 12
sensory channels, `learn` takes **8.67 ms** on average versus **4.76 ms** for
`step` (which also prepares the real prediction cache). The reported
75.57 MB for the factor and 2.34 MB for the reward are separate sums of
direct NumPy arrays, **not** complete resident-memory measurements. From
the [factor allocation](../../development/models/factor.py) (`N=200`,
`K=10`, 12 sensory channels, two actions, ten feature positions), the main
banks account for 74.88 MB of the factor subtotal:

| Bank | Decimal MB | Why it matters |
| --- | ---: | --- |
| Table counts `q0`, `q1` | 38.592 | One bank for each particle and regime, including unused slots |
| Log scores and cached weights `ev`, `_E` | 10.944 | Cached weights avoid repeated exponentials; memory is not free |
| Additive covariances `Sg` | 23.232 | Full feature covariance per particle/regime/channel |
| Additive means `mu` | 2.112 | Same capacity across particles and regimes |

**First memory candidate: allocate capacity for occupied regimes plus the
one next candidate, instead of storing ten dense rows from the start.**
Keep the current ten-regime limit, particle count, priors, precision, and
update order. A short diagnostic measured the high-water mark of
`max(particle.used)`: **4** in `s-2x3` and **5** in `s-4x3` over 360 steps
per world (agent seed 0, environment seed 1000, as in the profile), and
**3** in the exposed 320-step serial prefix. With one next candidate, six
rows would have sufficed for the larger scaling fixture; that is an
estimated **29.95 MB** reduction of those banks there, not a measured RSS
saving or a claim about long episodes. Growth has its own cost and may
briefly raise peak memory while copying; include that in the measurement.
Reducing N or diagonalizing covariance would alter inference and is
**not** the first optimization.

**First CPU step: measure where an expensive real update goes**, separately
from `step` and separately for a case with live joint choices. The synthetic
profile cannot establish that refreshing the joint window is cheap or
expensive in cancellation. If the selector is hot, try reusing invariant
bin/code preparation or exact count work with unchanged replay and candidate
support. Do not throttle selection or shrink the replay horizon on the
basis of these profiles. If the factor kernels dominate, optimize the
representation or the exact loops there. Compare warmed timings and exact
predictions/action values on short existing fixtures, including capacity
growth, regime recall and replay eviction where applicable.

## 4. Order of work

The behavior changes are staged separately from exact cost reductions:

- **Shipped locally:** action-conditioned reward in a matched, unchanged
  context, with the paired control above.
- **Tested, not adopted:** context-local reward changes. The
  [closed-loop partition check](REWARD_PARTITIONS.md#closed-loop-check-reward-change-on-a-rare-revisit)
  recovers from a visible reversal, but cannot detect better payoffs on an
  untried action and shows little benefit under noisy rare revisits. Keep the
  readout and evidence-collection questions separate before changing v60.
- **Tested, not adopted:** coupled arrivals and decision-relevant cues.
  The [fixed-value bounds](#what-the-coupling-check-found) prove that this
  interface can lose rankings, but also rule out any fix from coupling alone
  on the tested cancellation queries. The largest distal-delay failure
  instead occurred before the learner ever saw the cue.
- **Next candidate:** intervention coverage. The
  [distal contact check](#the-largest-distal-delay-loss-happens-before-cue-interpretation)
  isolates an untried motion that the policy did not revisit. The
  [reset comparison](#a-mode-reset-does-not-necessarily-make-repeated-trials-wasteful)
  favors active-bank coverage over a historical witness mask on the short
  serial prefixes. Neither a [revisit delay](#waiting-for-a-revisit-after-recoding-does-not-remove-the-cost-of-probing)
  nor [one or two credits](#a-hard-probe-budget-caps-interventions-not-their-reward-cost)
  reliably protects early reward. Keep the budget offline; prioritize
  deciding when a probe is worth its cost, and treat startup and recovery
  as different cases.
- **Not adopted:** full-vector outcome tables or multiple forgetting experts
  as a wholesale replacement for v60's sensory and regime models.

For exact cost reductions and evidence:

1. **Measure without changing behavior.** For the memory question, record
   regime high-water mark and bytes. For the serial question, use a short
   exposed segment around a reversal and break `Q` into fixed value,
   optimistic value, and information option alongside the regime, control
   row and reward state. Do not average away the transition at which they
   first disagree.
2. **Separate two tracks.** An exact representation change can be checked
   against frozen outputs and should not need a fresh qualification merely
   to test it. A change to the learned law or action ranking is a new model.
   Test one named failure and its controls first; do not carry accepted
   evidence over to a changed policy.
3. **Require a counterexample before adding complexity.** Start either with
   the reversal trace (strongest observed deficit) or the conditional cue
   toy (cleanest architectural limitation). Retain the successful delayed
   and cancellation cases as guardrails. There is no current evidence that
   a longer generic rollout, a new layer, or fewer particles is the right
   answer. Blind T4 remains a separate decision.
