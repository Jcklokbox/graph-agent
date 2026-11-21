# Reward partitions: a local replacement experiment

This experiment replaces **the fixed reward readout**, not the v60 agent. It
keeps the learned arrival codes, motion model, proposed cue/moderator basis,
and sensory projection. Neither the live policy nor its training is changed.
The diagnostic omits legacy unary/signed fallback, soft root refinement and
exploration. It does not suggest deleting those paths from v60 yet.

## One law per arrival

For an arrival code, assign each transition to a group. The pooled model has
one group; a unary model partitions by a cue bit at a given lag; a joint model
partitions by the moderator bit at the corresponding cue onset, with a third
group for everything outside onset. On the *same* transitions for every
candidate, maximize

\[
S(P)=\sum_{g\in P}\log p(n_g\mid\operatorname{Dir}(1/B,\ldots,1/B))
      -\log p(n\mid\operatorname{Dir}(1/B,\ldots,1/B)) - \log M_P .
\]

Here \(n_g\) counts reward bins for group \(g\), and the likelihood is for
an observed sequence, without a multinomial coefficient. \(M_P\) is the
number of proposals in the candidate's family (one for pooled). A candidate
with an unobserved group cannot win; pooled wins ties with splits. This reuses
v60's common-support criterion, but a unary winner is now an actual forecast
rather than a threshold for joint admission. It is still conditional on the
existing candidate discovery mechanism, not a general interaction search.

Only the winning partition defines the readout. Every group has the same
predictive recipe: unit total Dirichlet prior over bins, with the bin center
as its prior representative; observed representatives use the sums of actual
reward amounts. If an atom of value \(r\) is observed, the second-step mean
shifts by \((r-\mu)/(1+m)\) only for that arrival and group, where \(m\) is
posterior mass before the atom. Thus the planner does not have to choose
between separate update procedures. For a stationary partition, the forecast
may estimate its groups using a different window; this never reopens model
selection.

## Short checks

Deterministic tests cover a pooled winner, a unary winner without any joint
proposals, a rare joint signal missing entirely from a recent window,
independent forecast refitting, and an action reversal for a supplied
lag-one joint candidate. The last is transfer of the readout to another lag,
not evidence that discovery would autonomously propose the candidate.

The replay diagnostic uses four previously exposed, forced tapes. For each
prefix it compares v60's stored fixed margin to the experiment's fixed margin
for action 0 minus action 1, **not** the full policy's chosen action or return.
The diagnostic fits to the last 96 rows or up to 192 rows (reseeded from
current replay when the regime or control slots change). The candidate basis
and motion are held fixed for the four window combinations at each query;
all arrivals, not just the inspected code, use the corresponding windows.

| Tape | Queries | Winning partition at inspected arrival (short → long) |
| --- | ---: | --- |
| `canc-noise3`, 15, same | 4 | Joint → joint throughout; margin flips on cancellation |
| `canc-noise3`, 15, opposite | 4 | Unary → unary throughout; positive margin |
| `canc-noise3`, 15, neutral | 4 | Unary → unary throughout; positive margin |
| `canc`, 6, same | 5 | Joint/unary/unary/pooled/pooled → joint throughout |

On `canc`, seed 6, the window ablation is:

| Tick | Stored v60 | Short selection / short forecast | Short / long | Long / short | Long / long |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 168 | −.3859 | −.3847 | −.3869 | −.3847 | −.3869 |
| 173 | +.3718 | +.3572 | −.2771 | +.3762 | +.3718 |
| 178 | +.3719 | +.3604 | −.2769 | +.3794 | +.3719 |
| 183 | +.3720 | −.0594 | −.3095 | +.3557 | +.3720 |
| 188 | −.3860 | −.0594 | −.3093 | −.3788 | −.3861 |

The respective pairwise sign agreements over all 17 queries were 16, 14, 17,
and 17. Merely giving the selected model more data to estimate its means did
not fix the failure here. Keeping older **selection evidence** did, even
when the forecast still used the short window. The longer window is a
counterexample to equating selection support with forecast memory, not a
recommended lifetime. The following experiment tests that limitation.

## Time as a split, not a second reward model

Besides a static context partition, allow a subset `S` of its `G` groups to
be split into past and current visits. Evaluate both parts on the original
rows. Relative to the stationary score for that partition, add

```text
d_g = ell(past_g) + ell(current_g) - ell(unsplit_g) - log(N_g - 1)
gain = sum(d_g for g in S) - log(binomial(G, |S|))
```

where `ell` is the same Dirichlet log marginal used for context splits and
`N_g` counts visits to `g`. This charges for the group subset at a fixed size
and for each boundary among successive visits to its group. It is a
penalized comparison, not a normalized joint posterior over all models.
Only nonempty cuts are proposed. The gains add: for a given subset size,
the best boundaries are found per group and the highest gains are selected,
subject to the alias restriction below. A singleton subset recovers the
previous `log(G)` group cost.

The forecast uses only the later part of each split group; other groups
retain their counts and sums. A gap with no visits is not evidence of
change. The supplied tape window still bounds how far back the diagnostic
can see.

**A failure of unrestricted competition.** On the exposed opposite and
neutral tapes, the learner first has 16 visits to the inspected arrival
with cue absent and reward 0, then four with cue present and reward 1.
A pooled model with a time cut beats unary by about 1.06 log-score units
under the proposal costs, even though both induce exactly the same grouping
of rows. The time interpretation would now predict the high reward *even
with the cue absent*. The signs of the measured action margins did not catch
this loss: they agreed on all 17 queries before the restriction below.

The implemented restriction excludes a temporal candidate if its grouping
of the observed rows is identical, up to relabeling, to an available static
context candidate. This deliberately prefers a context explanation under
exact observational aliasing; it is not a claim that the data identify the
true cause. With that restriction, the context-and-time readout makes no
cuts on the four exposed tapes and reproduces all 17 margins of the
stationary long-window readout exactly.

**When a cut does help.** On a deterministic reward-only reversal fixture,
a joint group's reward changes from `.9` to `.1` after 100 rows visiting
only another group. An unchanged control with the same silence does not
split. After the second contradictory visit, the time split changes the
forecast from `.767` (stationary history) to `.233` (recent visits), opposite
sides of the safe `.4`; the other group's statistics remain unchanged.
Removing the 100 quiet rows preserves the affected group's forecast on this
fixture. Without any useful context feature, a reward reversal can also
select a pooled time split.

### What changes: the mix of contexts or the reward in a context?

The marginal reward law is a mixture of the conditional reward laws with
weights given by context exposure. A change in that mixture is neither
necessary nor sufficient for a change in any component. To interpret the
evidence, compare before/after rewards **within** context before pooling
across contexts. A context with no visits on one side gives no such contrast:
its unpenalized split gain is zero because `ell(empty) = 0`.

Two deterministic checks isolate the distinction:

| Fixture | What is visible after pooling | What the context-and-time readout finds |
| --- | --- | --- |
| Exposure shift versus reward shift, with the exact same ordered reward sequence | Both go from mean `.0625` to `.9375`. In the exposure-only case each context is still visited twice in its rarer period. | Exposure-only: static unary, no cut. Actual change with balanced context exposure: pooled time cut. |
| Opposite reversals | Each period has the exact same alternating payoff tape and the same context counts. | Unary with a cut in **both** groups; the current means are `.971` and `.029`. |

For the second fixture, the pooled log marginal prefers no cut at the true
boundary by 7.76 log units (before search costs). Conditioning on context
instead gives +20.21 log units for each local reversal. The earlier
one-group-only model left one group's mean at `.5`, on the wrong side of
a safe `.4` when that group's reward dropped to zero. Allowing multiple
cuts preserves the useful contrasts without introducing another detector.

**Remaining limits.** An omitted cue makes time/context aliasing invisible;
the test suite demonstrates that failure explicitly. Exact equality does not
resolve arbitrary partial overlap, and the clean exposure fixture is not a
noise robustness result. Reward movement entirely inside one bin also cannot
favor a cut under this likelihood. A single cut per group cannot represent
recurring regimes in general. The batch search is a diagnostic, not a claim
about per-step speed or a replacement for v60's motion-based regime bank.

## Closed-loop check: reward change on a rare revisit

Both policies choose their own actions, using the same reward readout with
or without temporal cuts. Contexts are given: one occurs every eighth
step; all other visits use a separate stable law. Actions do not affect the
next context, so each policy chooses by immediate predictive mean. Each
gets two bootstrap samples per action in each context. In the probing arm,
every fourth rare visit deliberately tries the action not currently preferred.
In noisy runs, rewards are binary draws with the stated amounts as means,
using shared random draws indexed by step and action for the two policies.

Run 320 steps on seeds 0, 1, and 2, with a change halfway through. There are
20 rare visits after the change per seed. Payoffs on those visits:

| Rare-context scenario | Probes | Noise | No cuts | Local cuts |
| --- | --- | --- | ---: | ---: |
| Stable `.9` vs `.1` | No | No | .900 | .900 |
| Stable `.9` vs `.1` | Yes | No | .700 | .700 |
| Stable `.9` vs `.1` | Yes | Yes | .700 | .700 |
| `.9/.1` reverses to `.1/.9` | No | No | .100 | .740 |
| `.9/.1` reverses to `.1/.9` | Yes | No | .300 | .620 |
| `.9/.1` reverses to `.1/.9` | Yes | Yes | .283 | .383 |
| Only the unchosen action improves, `.6/.1` to `.6/.9` | No | No | .600 | .600 |
| Only the unchosen action improves, `.6/.1` to `.6/.9` | Yes | No | .675 | .765 |
| Only the unchosen action improves, `.6/.1` to `.6/.9` | Yes | Yes | .633 | .633 |

Common-context rewards agree between policies in each pair. No cut was
selected on a stable rare context. In the deterministic reversal, the first
six rare visits improve from `.100` to `.500` without probes. In the noisy
reversal, the improvement comes from one seed; the other two tie. In the
unseen-improvement control without probes the better action is **never**
tried again. The entire observed history then agrees with a world in which
nothing changed; temporal inference cannot fix missing interventions.
Probes make that improvement visible but have a real cost: in the stable
case they reduce rare-context reward from `.900` to `.700`.

This is a closed-loop check of the readout on known contexts, not of the
whole v60 policy or its structure discovery. It supports keeping detection
and evidence collection distinct, not adding a fixed probing schedule or
switching on temporal cuts in default v60 based on these fixtures alone.

Run from `development/`:

```sh
python -m unittest -v tests.test_reward_partitions tests.test_factor_joint_reward
python -m eval.scripts.reward_partition_probe
python -m eval.scripts.reward_change_probe
```

Neither diagnostic writes tracked evidence or registers a new agent.
