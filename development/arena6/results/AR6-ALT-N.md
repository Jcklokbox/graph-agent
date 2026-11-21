# AR-6 alt-N: research notes

Exposed-seed evidence for the AC-6/AR-6 replica. Development only; no formal gate row was run,
no holdout or T4 stream was touched, and every comparison uses the exposed DEV seeds 0-7.

## AR-6 alt-N — research note (first turning point)

**Target.** On `canc` the maximizing action is `left` at a cue when the cancellation lamp is off
(return 1.0), `stay` while the lamp is on or when there is no cue (0.5), and never the far side
(0.0). Oracle tail 0.555–0.568, always-stay 0.500; the whole reward contrast is **0.055** in tail
mean, obtained once per 5 steps, so a policy that merely avoids shocks scores ~0.0 and one that
walks the wrong way at cues scores −1.0 (five DEV seeds: 6, 7, 8, 12, 13).

**Measured cause, three independent probes (seeds 6/7/8/12/13, 420 steps, exposed DEV only).**

1. *Free-discovery arm.* v48 never reaches a payoff at all on 4/5 failing seeds
   (`first_payoff = None`; seed 8 first payoff at t=98, seed 12 at t=13). It never chooses `left`
   at a cue without the lamp: at 30–40 late cue-without-lamp opportunities per episode it plays
   `left` 1–3 times and stays 29–38 times.
2. *Scripted exposure arm* (the max-reward action played for the first 8 cue-without-lamp
   opportunities, then free; ordinary observations only). Score jumps 0.455 → 0.525–0.545, the same
   optimal action is reused at 34–41 of 32–42 later opportunities, late shocks drop 18 → 5–8.
   **Reuse after exposure is intact.**
3. *Yoked arm* (identical count and timing of scripted actions, but played to the far side, so no
   payoff is possible). Score stays at 0.455–0.458 on 4/5 seeds; seed 12 is a genuine
   counter-example where spontaneous discovery happens before the yoke.

So the defect is **payoff discovery**, not credit assignment, and it is not the actuator/state
factor: the same episode shows correct local actions and a learned one-step control transition.

**Mechanism.** In the unified object `provisional_values`, the branch is
`incumbent = max(r, (bound*phase).max(1))`, where `bound` is the provisional upper payoff and
`phase` is the *onset* mask. At the cue-without-lamp state the agent's own factor has never seen a
payoff, so `r = 0.5`; the state is not an onset, so `incumbent = 0.5` for every action. The sibling
state cue+lamp is an onset carrying `bound ≈ 0.89`, and it is reached with high probability by
`stay` (0.97) but low probability by `left` (0.04). The optimistic bound is therefore a *reward for
arriving* at the sibling, which makes exploitation self-consistent: stay is repeated, the
cue-without-lamp state is re-armed 5 steps later and dropped again. Opportunity is charged to the
onset, not to the decision that reveals it, and a payoff requires an action at the *non-onset* cue
state. This is an inference deviation of the kind L-1 names.

**Alternative that this does not establish.** A pure exploration-rate deficit would look similar,
but the exposure/yoke contrast shows the missing variable is realized payoff, not action entropy:
with one reward the same controller exploits correctly for the rest of the episode.

**Next falsifiable step.** One coherent change inside the same object: replace the
"optimistic reward for arriving" term by an *optimistic payoff of the intervention at the current
decision*, i.e. push the provisional upper payoff through the learned control transition of the
current action before it enters the incumbent, and keep the branch correction unchanged. Prediction:
free-discovery seeds should find the first payoff and then reuse it; guard rows (aut/dist families)
should be unchanged because the incumbent is already backed by a real learned law there.

## AR-6 alt-N — research note (second turning point)

**Failed hypotheses, kept as evidence.** (a) Reallocating the bound through the learned control
transition (`edge = T@bound / T@(bound*n)`) — wrong space: `bound` is (state, feature) and cannot be
averaged over states. (b) Removing the novelty term entirely: score locks to exactly 0.500 on all
five failing seeds and no payoff is ever discovered, so all exploratory drive lives in that one term
and the defect is its direction, not its size. (c) Unary-coordinate cells (`v51`): fires for every
coordinate including geometry, so it is wrong about what an opportunity is (canc-noise3 shock count
70). (d) Configuration cells over all free channels (`v50`, `v53`): fixes `canc` 8/8 but a full
configuration over iid noise is never revisited, so its exposure never anneals; on `dist` it
suppresses the real payoff law (score 0.37, `dist-delay3` 0.19). (e) Room bound `1 - r_cell` applied
to a per-state room (`v59`): in `aut` the predictor's `r` is exact, so the door closes and the score
locks to 0.500 — the bound must not touch a cell whose payoff the law already predicts.

**Established.** The predictor screens the cancellation lamp as a real reward parent
(`slots[14]` = lamp lag 0 in every checked seed), so representation is not the blocker: the missing
object is the domain on which provisional payoff is counted. The optimistic headroom currently
multiplies the *state's* onset payoff by a *feature's* transient rarity, so (i) an onset already
explained by the law still claims `(1-m)*plausibility` and (ii) the optimism is monotone in the state
mean, which is why staying at cue+lamp dominates the mid payoff of the decision that can realize the
maximum.

**Fix frozen as v60 (one object, one statistic).** Every cell is a (outcome state, configuration of
the payoff's own screened parent channels). Its provisional payoff is the Jeffreys width of that
cell's realized payoff `sqrt(m(1-m)/(2+n))` — positive only while the cell is mid and poorly
determined, shrinking with its own exposure, zero where the payoff is already pinned. Nuisance
channels are not payoff parents, so they cannot multiply the cells; the marginal term and its onset
gate are unchanged. Exposed DEV, seeds 0-7: `canc` 0.958 (v48 0.698), `canc-noise3` 0.601 (0.457),
`aut` 0.989 (0.994), `dist` 0.961 (0.961), `aut-serial300` 0.797 (0.786), `aut-fooddelay2` 0.798
(0.863 reported at 16 seeds), `dist-delay3` 0.959 (0.952).

## Third turning point: controlled discovery test and gate results

**Manipulation (exposed seeds 0,1,3,4,6,7,8,9,12,13,15 on `canc`, 420 steps).** Three arms on the
same (variant, seed): free, scripted exposure to the maximum-reward action for the first eight
cue-without-lamp opportunities, and a yoke with the same count/timing played to the far side.

| agent | free | exposure | yoke | exposure - free | exposure - yoke |
| --- | ---: | ---: | ---: | ---: | ---: |
| v48 | 0.5037 | 0.5395 | 0.4783 | +0.0357 | +0.0612 |
| v60 | 0.5382 | 0.5396 | 0.5360 | +0.0015 | +0.0036 |

Under v48 a payoff the agent cannot find for itself is worth +0.036. Under v60 that advantage
collapses to +0.002, because free exploration now reaches the payoff on its own; the residual is the
yoke's small cost. This is the mechanism witness: the change acts on discovery, not on reuse.

**Gates completed on this tree.** 116/116 unit tests; 15/15 controller isolation
(`tests.test_factor_control` + `tests.test_factor_reward`); predictor parity exact
(`max |P_v48 - P_v60| = 0.0` over 200 steps on one identical observation/action stream, and the
3b-1 predictor unchanged with or without a reward head); act+learn scaling ratio 2.091
(s-2x3 2.862 s -> s-4x3 5.983 s, 360 steps, N200), i.e. below the registered 2.50 bound
(v48 on the same machine: 2.037). No formal gate row (DEV-16 block protocol, held-out, T4) was run;
no unseen seed was spent.

**Exposed DEV-16, seeds 0-7, candidate / v48.** canc 0.958/0.698, canc-noise3 0.601/0.457,
aut 0.989/0.994, dist 0.961/0.961, aut-half 0.980/0.994, aut-noise4 0.997/0.991,
dist-delay3 0.959/0.949, aut-serial300 0.797/0.795, aut-noise4-sticky 0.989/0.994,
dist-noise2-sticky 0.961/0.955, dist-noise2 0.962/0.949, aut-fooddelay2 0.798/0.798,
dist-delay3-noise2 0.947/0.955, dist-noise3-p4 0.951/0.953, aut-fooddelay3-noise1 0.910/0.910.
Per-cell episode counts are 8 seeds x 1 episode = 8 per cell (canc and canc-noise3 also carry the
11-seed x 3-arm controlled probe above). Two cells move down by 0.014 (aut-half, aut-noise4-sticky),
one by 0.008 (dist-delay3-noise2), offset by gains of 0.260 and 0.144 in the target cells.
