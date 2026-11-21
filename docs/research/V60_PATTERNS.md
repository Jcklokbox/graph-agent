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

One falsifying toy for the interface: let a continuation succeed exactly
when a controlled bit `c` and a cue `z` agree. One first action yields
`(c,z)=(0,0)` or `(1,1)` with probability `1/2` each; another yields
`(0,1)` or `(1,0)` equally often. Both have identical separate marginals;
combining them independently values agreement at `1/2` for either action,
whereas the true values are 1 and 0. This is an algebraic boundary case,
not a claim that v60 incurred this error on DEV. On a saved decision state,
compare the existing calculation with an explicit conditional law for
**one** decision-relevant cue at fixed arrival, holding the reward law and
transition marginals constant. If it cannot move the action margin, do not
add a joint representation for that case. If it can and matching real
support exists, improve that interface rather than adding a reward bonus.

## 3. Where it actually makes sense to save resources

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
