# Online posterior graph: implementation status and results

Everything here concerns `development/posterior_graph.py` (`PosteriorGraph`, v10 since Round 12). The legacy
`BlindGateGraphBrain` was not run. Reproduce with:

```
cd development
python -m unittest -v test_posterior_graph.py                  # 30 tests, OK (~5 s)
PYTHONPATH=<numpy> python transfer_suite.py --dev              # transfer benchmark, DEV split (15 variants, ~15 min on 2 CPUs)
PYTHONPATH=<numpy> python transfer_suite.py --dev --seeds 4    # screening only (first 4 DEV seeds)
PYTHONPATH=<numpy> python trace_fingerprint.py --check trace_fingerprint_ref.txt  # bit-exact check of a perf refactor (~95 s)
PYTHONPATH=<numpy> python transfer_suite.py --held5            # fresh held-out set of Round 12, report only
PYTHONPATH=<numpy> python transfer_suite.py --held4            # Round-11 held-out set (no longer clean)
PYTHONPATH=<numpy> python transfer_suite.py --held3            # Round-10 held-out set (no longer clean)
PYTHONPATH=<numpy> python transfer_suite.py --held2            # Round-8 held-out set (no longer clean)
PYTHONPATH=<numpy> python transfer_suite.py --held             # Round-7 held-out set (no longer clean)
PYTHONPATH=<numpy> python run_posterior_scenarios.py --dev     # 8 DEV seeds only, ~35 s on 2 CPUs
PYTHONPATH=<numpy> python run_posterior_scenarios.py --suite   # everything, ~100 s on 2 CPUs
```

Episodes run in a process pool (`PG_PROCS=<n>`, `PG_PROCS=1` = serial; output is identical).

## Round 12: controllability on the attended context, earlier regime storage (v10)

**Protocol.** Selection on DEV-15 (8 seeds). Because the per-seed spread of the noisy variants is
about ±.4 (standard error of an 8-seed mean ≈ .14), the noisiest variants were additionally
screened on DEV seeds 8-15 (env 108+, brain 8+; still DEV, never held-out). A **fresh HELD5**
(10 new variations, env seeds 15000+, brain seeds 1500+, 12 seeds) was written into
`transfer_suite.py` before any run on it and run **once** on the final code; v9 was then run on
it as a control (report only).

### Changes kept (2)

| finding (instrumented) | change |
|---|---|
| **Controllability fragmented by noise and confounded by memory.** The controllability BF was a sum over *exact* contexts: in dist-noise3-p4 every noise combination paid its own Occam penalty and the cue channel (which appears only at position 2) was judged uncontrollable (P = 1.0), so the action-pooled cell predicted the cue after staying at position 1 and the planner imagined "cue at position 1 -> food". Conversely in aut-fooddelay2 the food channel (purely Pavlovian) was judged *controllable* (P_unc = 0.0): the policy acts on memory of the precursor, which the current context does not contain. | **Controllability evidence on the attended context plus open windows**: once attention exists, the within-context action-vs-shared BF is computed per (attended part of the context, open expectation windows), from the summed exact belief-state cells of its members (rebuilt when attention grows; exact contexts before attention). Noise channels no longer split the comparison and the open windows remove the memory-policy confound. After 600 steps: dist-noise3-p4 cue P_unc 1.0 -> 0.0, aut-fooddelay2 food 0.0 -> 1.0. |
| **No regime stored in half of the serial runs**: the robust change detector reached only P(change) ≈ .35-.45 at the first reversal in 4/8 seeds, so nothing was archived and no later reversal could be a reinstatement. | **Store earlier than discount** (`regime_store` = .25): the current model is archived once per run of surprise as soon as P(change) > .25 (previously .5, the point where most memory is already discounted). Storage is cheap insurance; a stored copy is only reinstated if it later predicts surprising transitions better by 3 nats. In the other 14 DEV variants not a single extra regime was stored (scores identical). |

Tried and rejected (all removed): per-context ("local") controllability with a hierarchical rate
of controllable contexts (dist-noise3-p4 −.16); hierarchical BF over exact contexts (Round 11);
onset salience without the timing part, and inheritance of salience by a renewing cue (16-seed
screen: +.01 on the noisy variants, fooddelay2 −.09; not kept for simplicity); event_explore .75
(dist-noise3-p4 .45 on 8 seeds, .57 on seeds 8-15, but aut-fooddelay3-noise1 and canc-noise3
drop; DEV-15 .720); `regime_store` .1 (8/8 serial seeds with savings, but false storage and
regime flip-flops on the delayed-food variants; DEV-15 .748 vs .750 at .25).

### DEV (8 seeds; selection)

| variant | v9 | **v10** |
|---|---|---|
| aut | .886 | .852 |
| canc | .995 | .995 |
| dist | .996 | .992 |
| aut-half | .886 | .852 |
| aut-noise4 | .803 | .783 |
| dist-delay3 | .938 | .972 |
| aut-serial300 | .576 | **.706** |
| aut-noise4-sticky | .800 | .840 |
| canc-noise3 | .698 | .687 |
| dist-noise2-sticky | .823 | .876 |
| dist-noise2 | .703 | .723 |
| aut-fooddelay2 | .627 | .630 |
| dist-delay3-noise2 | .605 | .590 |
| dist-noise3-p4 | .455 (7/8) | .226 (5/8) |
| aut-fooddelay3-noise1 | .205 (6/8) | **.382** (7/8) |
| **mean DEV-13 / DEV-15** | .795 / .733 | **.808 / .740** |

16-seed screen (DEV seeds 0-15) of the noisiest variants, v9 vs the coarse-controllability step
(before `regime_store`, which does not change these variants, and before the empty attended
context was included): dist-noise3-p4 .20 -> .23, aut-fooddelay3-noise1 .20 -> .33,
aut-fooddelay2 .65 -> .72, dist-delay3-noise2 .52 -> .48, dist-noise2 .75 -> .71, canc-noise3 .75 -> .74.
The 8-seed drop of dist-noise3-p4 (.455 -> .226) is mostly seed luck of v9 on seeds 0-7 (v9 gives
−.055 on seeds 8-15).

Savings on serial300 (mean reward in the first 100 steps of phases 1-5): v9
[.446 .473 .535 .528 .530] (savings in 4/8 seeds), v10 [.446 .474 .568 .581 .581]; in 7/8 seeds
every reversal from the third phase on is a reinstatement within 0-10 steps.

### HELD5 (fresh, 12 seeds, run once; v9 = control, run afterwards)

| variant | v9 | **v10** | v10 wins |
|---|---|---|---|
| dist-noise4-p3 | −.022 | **.146** | 7/12 |
| dist-delay3-noise3 | −.025 | −.031 | 4/12 |
| aut-fooddelay4-noise1 | −.013 | −.011 | 8/12 |
| aut-fooddelay3-sticky3 | −.235 | **−.141** | 2/12 |
| aut-serial450 | .758 | **.803** | 12/12 |
| canc-noise4-p2 | **.184** | .060 | 9/12 |
| aut-noise6 | **.765** | .733 | 12/12 |
| dist-sqrt | −.039 | −.039 | 0/12 |
| aut-fooddelay2-noise3-p3 | −.265 | **−.093** | 6/12 |
| dist-delay4-noise2-sticky | .446 | **.617** | 12/12 |
| **mean** | .155 | **.204** | 72/120 (v9 66/120) |

**Honest reading.** v10 is better than v9 on fresh variations (+.05 mean, 6 of 10 variants
better, 2 worse), and the serial savings now transfer (serial450 .803). But HELD5 is a much harder
set than HELD4 and the absolute level is low: delayed food with noise, long-delay dist with
iid noise and cancellation with sparse noise stay near the "stay" baseline.

### New failure found on HELD5 (diagnosed, not tuned)

* **dist-sqrt (0/12, both versions).** With w -> sqrt(w) the reference point is the stay reward
  (.707) and the symmetric frame puts shock at z = 0 but food only at z = .71: a food trip gains
  +.21 and risks −.5 relative to the reference, so the planner never pays the two-step
  exploration cost (no food in 800 steps), whereas dist-affine (linear) is solved. The value
  frame is symmetric about the reference; the asymmetry of gains and losses then rescales the
  *optimism* of exploration, which is a design question of the frame, not a parameter.

### Remaining failure mechanisms

* **Fixed optimism vs deep exploration.** With correct controllability, dist with iid noise needs
  a two-step sacrifice (left, left) after the cue; with the 2-step planner and event_explore .5
  the depth-1 window bonus can never exceed staying (it would need an untested-window bonus
  > .98). v9 solved some of these seeds only through the controllability error (an imagined cue
  at the wrong position). More optimism solves dist-noise but costs the delayed-food variants.
* **Delayed food with noise**: improved (DEV .21 -> .38) but still costly exploration in rare
  contexts; HELD5 versions remain at the baseline.
* **Noise-channel controllability** is still decided from sparse within-context action contrasts
  (a deterministic policy rarely tries other actions), so an iid channel can look weakly
  controllable (P_unc .16-.95) and then keeps its split in the timing cells.

## Round 11: irrelevance instead of relevance, prediction-error-gated regime retrieval (v9)

**Protocol.** As in Round 10. The HELD3 failure families entered DEV with other parameters:
`dist-noise3-p4` (3 iid channels, p = .4) and `aut-fooddelay3-noise1` (food 3 steps after the
precursor, one iid channel); DEV is 15 variants x 8 seeds. All choices were made on DEV (plus DEV
seeds 8-15 for screening of the noisiest variants). A **fresh HELD4** (10 new variations, env
seeds 13000+, brain seeds 1300+, 12 seeds) was written into `transfer_suite.py` before any run
on it and run **once** on the final code; v8 was then run on it as a control (report only).

### Changes kept (3 small edits of v8, all general)

| finding (instrumented, DEV seeds) | change |
|---|---|
| The unexplained aut regression of v8 (.911 -> .789): the reduced timing key kept only channels *shown* to be relevant, so a channel never tested on one side (e.g. position under a rare action) was marginalised, and timing cells learned in one position produced spurious hazard boosts (food expected) in another. | **Irrelevance, not relevance, is what must be shown**: a context channel is dropped from the reduced key only if it is uncontrollable, has been tested on both sides, and the split is not credible (chi2 <= max(3.84, log N)). Untested channels keep their split. aut .789 -> .874. |
| Regime retrieval was evaluated every step; on well-predicted steps small likelihood differences between stored and current tables accumulated and sometimes reinstated an old regime without any reason (and cost one likelihood pass per stored regime per step). | **Retrieval is driven by prediction error**: stored regimes are compared only on transitions that the current model finds surprising (joint log LR of fresh prior vs current cells > 0); otherwise the evidence only decays. Also removes most of the per-step regime work. serial300 .462 -> .576. |
| The expected-event bonus counted tests of a timing window in the exact context, so every new combination of noise channels made the same test "novel" again (endless exploration in noisy variants). | **Tests are counted in the reduced context** of the same timing cell (`_reduced_ctx_counts`). dist-noise2 .626 -> .703, dist-noise3-p4 −.057 -> .455. |

Tried and rejected on DEV (all removed): leaky surprise accumulator in the global change
detector (all .749 vs .790 on DEV-13: canc, dist, dist-delay3 break), robust per-channel mixture
in the change LR (`GC_PI`, worse on dist-noise2 and fooddelay2), no lazy quiet-period discount,
Student-t predictive density for continuous cells (worse on fooddelay2, serial, dist-noise2),
an extra attended-only action-pooled memory cell (helps dist-noise2 / fooddelay3-noise1 but
costs serial and aut; DEV-15 .718 vs .733), a hierarchical (pi ~ U(0,1)) controllability BF so
that noise-fragmented contexts do not each pay an Occam penalty (DEV screen: dist-noise3-p4
.455 -> .247, fooddelay2 and dist-noise2 also worse), a lower UCB coefficient (explore 1.5:
fooddelay2 better, dist-noise variants worse).

### DEV (8 seeds; selection)

| variant | v8 | v8 + irrelevance | **v9 (+ gated retrieval + reduced test count)** |
|---|---|---|---|
| aut | .789 | .874 (×16 .871) | **.886** |
| canc | .995 | — | .995 |
| dist | .996 | — | .996 |
| aut-half | .789 | — | .886 |
| aut-noise4 | .771 | — | .803 |
| dist-delay3 | .946 | — | .938 |
| aut-serial300 | .462 | .470 | **.576** |
| aut-noise4-sticky | .817 | — | .800 |
| canc-noise3 | .688 | — | .698 |
| dist-noise2-sticky | .876 | — | .823 |
| dist-noise2 | .660 | .626 | **.703** |
| aut-fooddelay2 | .726 | .652 | .627 |
| dist-delay3-noise2 | .542 (7/8) | — | **.605** (8/8) |
| dist-noise3-p4 (new) | — | — | .455 (7/8) |
| aut-fooddelay3-noise1 (new) | — | — | .205 (6/8) |
| **mean DEV-13 / DEV-15** | .774 / — | — | **.795 / .733** |

(The intermediate irr + gated retrieval, without the reduced test count, gave DEV-13 .790.)
Savings on serial300 (mean reward in the first 100 steps of phases 1-5): v8
[.449 .464 .509 .528 .520], v9 [.446 .473 .535 .528 .530]. In 4/8 seeds the first reversal is
detected (c > .5), a regime is archived and every later reversal is a reinstatement within 2-5
steps (first-100 reward .58-.60); in the other 4 seeds the detector never reaches c > .5 and
there are no savings.

### HELD4 (fresh, 12 seeds, run once; v8 = control, run afterwards)

| variant | v8 | **v9** | v9 wins |
|---|---|---|---|
| dist-noise5-p2 | −.207 | −.178 | 4/12 |
| aut-fooddelay3-noise2 | −.281 | −.317 | 2/12 |
| dist-delay4-sticky1 | .840 | .803 | 12/12 |
| aut-serial400 | .700 | .735 | 12/12 |
| aut-fooddelay5 | .442 | .369 | 10/12 |
| canc-noise2-sticky9 | .854 | .856 | 12/12 |
| aut-noise3-p2 | .703 | .722 | 12/12 |
| aut-fooddelay2-sticky2 | .354 | .423 | 12/12 |
| aut-square-noise2 | .783 | .779 | 12/12 |
| canc-affine | .982 | .982 | 12/12 |
| **mean** | **.517** | **.517** | 100/120 (v8 101/120) |

**Honest reading.** The DEV gain (+.02 on DEV-13) does not transfer: on HELD4 v9 equals v8 in
the mean; per-variant differences are within seed noise (per-seed spread on the noisy variants is
±.4, so an 8-seed DEV mean has a standard error of ~.14). What the round did achieve is
explanatory: the aut regression of v8 has a cause and a fix, retrieval no longer runs on
predictable steps, and the bonus no longer treats noise combinations as new tests. The two
failure families persist on HELD4 with new parameters: **much iid noise in dist** and **delayed
food plus noise in BlindGate**.

### Open failure mechanisms (diagnosed, not solved)

* **Controllability fragmented by noise.** In dist-noise3-p4 (seed 5) the cue channel 3 (which
  appears only at position 2) is judged uncontrollable, P = 1.0: the controllability BF is a sum
  over *exact* contexts, and every noise combination pays its own Occam penalty, which outweighs
  the few contexts where the action matters. The action-pooled cell then predicts the cue with
  p = .26 even after staying at position 1, the planner imagines "cue at position 1 -> go left ->
  food" (.60), and the agent idles at position 1 (.45) instead of returning to 2 (.50). The
  hierarchical BF fixed the probability (P = .19) but was worse on DEV overall; computing
  controllability on the attended (coarse) context is the next candidate.
* **Exploration in delayed food with noise.** In aut-fooddelay3-noise1 the agent times the food
  correctly when the precursor window is open, but takes costly exploratory moves in rare
  exact contexts (posterior sd .1-.24 from forgetting and fragmentation), ~100 shocks per 600
  steps. A lower UCB coefficient helps here and hurts dist-noise; no general rule found.
* **Change detection on sparse surprise.** Surprises on reversals are tied to rare events; a
  well-predicted step in between resets the CUSUM, so in half the serial seeds no change point is
  confirmed. Leaky accumulation fixes serial (.725) but breaks canc/dist.

## Round 10: noise fragmentation, delayed food, change points and savings (v8)

**Protocol.** The HELD2 failure families entered DEV with *other* parameters: `dist-noise2`
(2 iid channels), `aut-fooddelay2` (food 2 steps after the precursor), `dist-delay3-noise2`.
DEV is now 13 variants x 8 seeds; configurations were chosen on DEV only. A **fresh HELD3**
(10 new variations, env seeds 11000+, brain seeds 1100+, 12 seeds) was written into
`transfer_suite.py` before any run on it and run **once** on the final code; the v7 code was run
on it afterwards as a control (report only). HELD2 is no longer clean (its failures motivated
the round) and is shown only to compare with the Round-8 goals.

### What the diagnosis showed, and the general fixes

| finding (instrumented, DEV seeds) | general fix |
|---|---|
| dist + iid noise: knocking out the noise channels from the context key of the timing cells alone lifts dist-noise2 from .21 to .67 (oracle knock-out). Each noise configuration opens a new exact context; with only `context_shrink`=2 pseudo-counts from the generic cell, the timing posterior there is too wide to pass the credibility test, so the delayed food is never planned for. Filtering by learned attention is not reliable (.28). | **Reduced-context back-off for timing cells**: exact context -> reduced context -> generic cell. A context channel stays in the reduced key if splitting the generic cell's children on it is credible (chi2 > max(3.84, log N), BIC for one extra rate) or if the channel is controllable (by the existing controllability BF). The reduced rate is the prior of the exact cell with empirical-Bayes strength (beta-binomial method of moments among the exact contexts sharing the reduced key; >= `context_shrink`). |
| aut-fooddelay2: the "precursor 2 steps ago -> food" memory cell exists only for the action actually taken (a1 = stay); for a0/a2 the agent has almost no belief-state cells, so it never learns that leaving the centre at the right time is safe/food. | The **action-pooled exact context of uncontrollable channels now also carries open windows** (belief-state memory pooled across actions), weighted by P(uncontrollable). |
| Global change point fired falsely in single spikes on fooddelay2 and dist-noise2 (one surprising transition gave c ~ 1 and wiped all memory), while on serial reversals the evidence was split over several steps. | **Outlier-robust, accumulated global change point**: per-step LR is a mixture with an outlier rate eps = .05 (at most log 20 per step), accumulated as CUSUM; memory is discounted to (1 - c) once per run of surprise instead of compounding every step; in quiet periods the prior hazard is accumulated and applied lazily. |
| Serial reversals: after each reversal the whole model was relearned (no savings). | **Contextual inference over latent regimes**: on a confirmed change the current tables are archived (at most 4 regimes); every step each stored regime's predictive log-likelihood of the transition is compared with the current one (exponential memory .8); with a 3-nat advantage the stored model is reinstated. Test: return to a known regime in 5 steps vs 25 without regime memory. |

Tried and rejected on DEV: reward-model value for unvisited successor states (`MBNEW`: −.32 on
fooddelay2), windows informative for any channel in the value key (`OCANY`), plain
empirical-Bayes pooling to the generic cell without channel selection (dist-noise2 −.06),
channel selection without keeping controllable channels (aut .79, serial .46), a lower bound on
negative per-step evidence in the change detector (worse on fooddelay2 and aut). All flags were
removed; the final code has no experimental switches.

### DEV (8 seeds; selection)

| variant | v7 | A = robust GC + regimes + pooled memory | A + reduced timing | **final (A + reduced timing, controllable kept, EB strength)** |
|---|---|---|---|---|
| aut | .911 | .875 | .794 | .789 |
| canc | .995 | .995 | .995 | .995 |
| dist | .986 | .996 | .992 | .996 |
| aut-half | .911 | .875 | .794 | .789 |
| aut-noise4 | .686 | .771 | .691 | .771 |
| dist-delay3 | .907 | .925 | .976 | .946 |
| aut-serial300 | .475 | .573 | .464 | .462 |
| aut-noise4-sticky | .836 | .800 | .805 | .817 |
| canc-noise3 | .717 | .744 | .709 | .688 |
| dist-noise2-sticky | .490 (6/8) | .593 | .867 | .876 |
| dist-noise2 | .211 (5/8) | .463 (5/8) | .606 (7/8) | .660 (8/8) |
| aut-fooddelay2 | −.198 (2/8) | .534 (7/8) | .438 | .726 (8/8) |
| dist-delay3-noise2 | .106 (4/8) | .449 (7/8) | .567 (7/8) | .542 (7/8) |
| **mean all / variants** | .618 / .514 | .738 / .673 | .746 / .692 | **.774 / .728** |

(The intermediate "controllable kept, without EB strength" variant gave .766 / .722.) Wins are
8/8 unless shown. Costs on DEV: aut .911 -> .789 (consistent over all 8 seeds: later in the
episode a little more shock, the cause was not found), and the serial gain of A (.573) is lost
again with reduced timing (.462): in such runs the robust detector often does not reach c > .5 at
the reversal, so no regime is archived.

### HELD3 (fresh, 12 seeds, run once; v7 = control, run afterwards)

| variant | v7 | **v8** | v8 wins |
|---|---|---|---|
| dist-noise4-p4 | −.281 | **−.119** | 3/12 |
| aut-fooddelay4 | −.803 | **.236** | 9/12 |
| dist-delay5-noise1 | .449 | **.633** | 11/12 |
| aut-serial350 | .554 | **.648** | 12/12 |
| aut-fooddelay2-noise2 | −.218 | **.017** | 6/12 |
| dist-delay3-sticky2 | .617 | .578 | 11/12 |
| aut-noise5 | .770 | .744 | 12/12 |
| canc-noise5-p3 | .525 | .564 | 12/12 |
| aut-sqrt-noise2 | .538 | .482 | 12/12 |
| dist-affine | .999 | .987 | 12/12 |
| **mean** | **.315** (85/120 wins) | **.477** (100/120 wins) | |

### HELD2 (not clean any more; compared with the Round-8 goals)

| variant | Round 8 (v7) | v8 |
|---|---|---|
| dist-noise3 | −.199 | .217 |
| aut-fooddelay3 | −.592 | .437 |
| dist-delay5 | .254 | .400 |
| dist-delay4-noise2 | .116 | .444 |
| aut-serial250 | .387 | .414 |
| aut-noise6-sticky95 | .734 | .728 |
| canc-noise4-sticky | .901 | .823 |
| aut-affine | .890 | .880 |
| aut-sqrt | .376 | .399 |
| aut-dupaction-noise3 | .669 | .700 |
| **mean** | .354 | .544 |

**Savings.** Mean reward in the first 100 steps after each phase start, aut-serial300, 4 DEV
seeds: v8 .449 / .464 / .509 / .528 / .520 vs v7 .449 / .490 / .470 / .481 / .496 — a rising
trend over reversals instead of a flat one, but small; regime reinstatement fires in some
seeds only (see the DEV cost above). The unit test shows the mechanism works when the change is
detected.

### Algorithmic work

* The global discount of every cell (`_discount_all`, ~10% of time in Round 9, called on most
  steps) now runs once per run of surprise plus a lazy prior-hazard discount (every ~10 quiet
  steps at H = .001).
* New cost: the reduced-context analysis of timing cells (per decision, per generic cell), and
  regime evidence (one likelihood per stored regime per step). After memoising the analysis per
  decision (fingerprint-identical refactor) v8 is still slower than v7 on noisy tasks
  (fingerprint set 79 s vs 70 s with 13 vs 10 variants; HELD3 aut-noise5 41.5 vs 27.0 s/episode
  before the last refactor). The planner prediction count is unchanged and remains the main cost.

### Open

* Many iid noise channels in dist (dist-noise4-p4 −.12) still fail: with 4 channels at p .4 the
  split tests and the exact-context exploration bonus are still fragmented (the event bonus counts
  tests per exact context by design).
* Delayed food with noise (aut-fooddelay2-noise2 .02) is only at the stay level.
* aut regression on DEV (.91 -> .79) and unreliable regime archiving under reduced timing.
* Hand-made cancellation/blocking/run-length rules remain.

## Round 9: where the benchmark time goes (behaviour unchanged)

**Measurement.** One aut-noise4 episode under cProfile (exact call counts) and under the
sampling profiler py-spy (shares without instrumentation overhead), plus `line_profiler` on the
hot functions. Before the round: ~70% of the time is the two-step planner in `step()`, ~25% is
`learn()`.

* The planner is `_q2` for 3 actions -> `_expected_continuation` over up to 2^4 imagined
  successor observations -> `_best_next` over 3 actions -> for the value channel,
  `_model_based_value` predicts *every* channel of the next state. Predictions are already
  memoised by (active set, action, target, depth); after that dedup aut-noise4 still needs
  **~1200 distinct channel predictions per step** (729k in 600 steps; 16 channels, 3 actions,
  ~25 distinct successor active sets). This count is the model itself (the expectation over the
  next observation and the reward model R(y)) and was not touched.
* **Implementation losses** inside each prediction:
  1. `_excitation` scanned the whole `births` set for every active cue of every prediction to
     compute the residual rate (O(|births|) per cue): ~25% of the run.
  2. Per-cue terms (residual rate, background, signed strength) depend only on
     (cue, action, target) but were recomputed for every imagined active set.
  3. `_inhibition_probability` re-derived every cue's cancellation contrast per prediction;
     `_hazard_boost` re-scanned all open windows and timing cells per prediction.
  4. `_discount_all` (global change point, 375 of 600 steps) made one Python method call per
     cell (12k+ exact-context cells).
* **Fixes** (pure caching/inlining, same floating-point operations in the same order):
  births indexed per (action, target, cue) in the per-decision cache (updated when a pair is
  born); per-cue excitation terms memoised per decision; per-target cancellation contrasts and
  per-(target, action, depth) supported timing windows memoised per decision; decay inlined in
  `_discount_all`.
* **Fundamental, left as is:** the planner's prediction count (2^rollout_branch x actions x
  channels), the global change-point discount of every cell (`_discount_all`, ~10% after the
  round; a lazy global scale would not be bit-identical), and the Granger attention test of all
  pairs x lags on every step (~8%).

**Identity check.** `trace_fingerprint.py` hashes every root Q value (full `repr`) and action of
one episode per DEV variant: identical to the pre-round code for brain seeds 0, 1 and 2 (30
episodes). The full DEV table is identical to Round 8 in every column except s/ep
(all .792 / variants .718). 26 tests OK.

**Speed.** DEV wall time on 2 CPUs: **~450 s -> 287 s** (x1.57); the fingerprint set 112 s -> 70 s.
Per-episode: aut-noise4 25.3 -> 13.8 s, aut-serial300 21.1 -> 14.6 s, aut-noise4-sticky 23.7 ->
13.1 s, canc-noise3 13.5 -> 9.0 s, aut 6.0 -> 4.5 s. The pool was already saturated (longest
jobs first, `chunksize=1`, tail < 5%), so process-level parallelism had nothing left to gain.

**Process.** `--seeds K` (DEV only) runs the first K seeds as a screen for discarding clearly
worse ideas; configuration choices remain on all 8 DEV seeds, held-out splits always on all 12.
`trace_fingerprint.py --check` replaces a full DEV run when a change is meant to be a pure
refactor.

## Round 8: trajectory diagnosis of dist, complex noise and delays > 2

Each change below comes from an instrumented trajectory, not from a parameter sweep. The DEV
set gained three noise forms that had failed on the Round-7 HELD set, with other parameters:
`aut-noise4-sticky` (4 sticky channels, keep .9), `canc-noise3` (3 i.i.d. channels) and
`dist-noise2-sticky` (2 channels, p .3, keep .8). Because the Round-7 HELD results motivated
these families, that set is no longer clean. A **fresh HELD2** (10 new variations, env seeds
9000+, brain seeds 900+, 12 seeds) was defined before any run on it and run once at the end.

### What the trajectories showed, and the fixes

| finding (trajectory) | general fix |
|---|---|
| A frame change of the value code re-expressed `factors` / `context_factors` / `reward_cells` but **not** the Round-7 coarse cells: the coarse prior of the value lived in a stale scale | coarse cells and the spread statistics are re-expressed too (a bug fix; aut .823 → .863, noise4 .345 → .436) |
| `dist`: every wrong move to the far side happened with open windows at *stale* ages ((3,5), (0,3), …): an unvisited belief-state cell inherited the exact-context mean, which the policy biases (the agent mostly moves there when food is due) | a window enters the belief-state key **only at lags with credible non-baseline timing** (`_lag_informative`); without evidence it promises nothing beyond the baseline |
| aut / noise: rare exact contexts with an unattended noise channel had sd ≈ .3 forever: τ² (how much a child can differ from its attended parent) was estimated per parent from 2–3 children plus a fixed pseudo-child at .25 | **hierarchical τ²**: the spread of children around parents over *all* parents of the target is the prior of the local spread (learned irrelevance generalises); noise4 .44 → .75–.81 |
| aut: the continuation of the 2-step plan was evaluated in a single thresholded "modal" successor, producing chimeras (position 0 *and* a precursor; both precursors at once) worth .96 | **continuation = expectation over the next observation** (E over successors of max over a′; up to 4 uncertain attended channels are enumerated) |
| delay 3: at the age where food is not due yet, moving to the middle (.45) was predicted as .03: the negative timing lift multiplied the *whole* value mean, although the belief-state cell had already learned .45 | the timing lift is a **prior that the belief-state cell overrides**: h · m/(m + n_mem) |
| delay 3: no food ever found: the event bonus was cut at lag ≤ 2 (`event_horizon`, a constant matching the base task) | **cut-off removed** (windows live up to `max_age`); H = 4/6/12 were within noise of each other on DEV, the parameter-free version was kept |
| sticky noise was attended: noise→noise pairs fired (Yule's nonsense correlation of two autocorrelated series); the action-pooled table was confounded by the agent's own policy | **Granger-style attention**: i(t+1−k) → j(t+1) within each action and each stratum of j(t), log BFs of the strata summed, k = 1..max_age (delayed causes earn attention), threshold over all pairs, actions and lags |
| `dist` with sticky noise: windows anchored on the agent's own moves (e.g. "left position 2 one step ago") carried policy-confounded timing (+.7 lift) | **timing counts weighted by the surprise of the anchoring onset** (associability, Pearce–Hall): a predicted / self-caused onset adds almost no evidence |

Rejected on DEV (removed): an action-pooled Granger table in addition (.671 vs .694); dropping the
transient-channel anchoring rule (run length ≤ 3) once salience weights existed (.751 vs .792),
so that hand rule stays; a midpoint clip of the negative lift (dist .295).

### DEV (8 seeds; incremental path, each line includes the previous ones)

| step | aut | canc | dist | aut-half | noise4 | delay3 | serial300 | R7 set all / variants |
|---|---|---|---|---|---|---|---|---|
| Round 7 final | .823 | .990 | .335 | .823 | .345 | .010 | .305 | .519 / .371 |
| + affine fix of coarse cells | .863 | .990 | .335 | .863 | .436 | .010 | .313 | .544 / .405 |
| + lag-informative windows | .823 | .990 | .446 | .823 | .410 | .001 | .348 | .549 / .395 |
| + hierarchical τ² | .724 | .990 | .422 | .724 | .809 | −.010 | .405 | .580 / .482 |
| + expected continuation | .840 | 1.000 | .386 | .840 | .802 | .004 | .489 | .623 / .534 |
| + timing lift as prior | .843 | .995 | .949 | .843 | .849 | .109 | .501 | .727 / .575 |
| + no event horizon | .851 | 1.000 | .982 | .851 | .842 | .632 | .475 | .805 / .700 |
| + Granger attention (lag 1) | .891 | .976 | .900 | .891 | .763 | .631 | .447 | .786 / .683 |
| + Granger lags 1..max_age | .857 | 1.000 | .944 | .857 | .806 | .650 | .440 | .793 / .688 |
| **+ salience-weighted timing (final)** | **.911** | .995 | **.986** | **.911** | .686 | **.907** | .475 | **.839 / .745** |

New DEV noise forms (final vs. before Granger attention): `aut-noise4-sticky` −.111 → **.836**
(8/8), `canc-noise3` .603 → **.717**, `dist-noise2-sticky` .156 → **.490** (6/8). Full DEV (10
variations): **.792 / .718**. `aut-noise4` fluctuates between .69 and .85 across configurations.

### HELD2 (fresh, 12 seeds, run once)

| variation | score | wins |
|---|---|---|
| aut-noise6-sticky95 (6 channels, p .3, keep .95) | .734 | 12/12 |
| canc-noise4-sticky | .901 | 12/12 |
| dist-delay5 | .254 | 7/12 |
| dist-noise3 (i.i.d., p .3) | −.199 | 3/12 |
| dist-delay4-noise2 | .116 | 7/12 |
| aut-fooddelay3 | −.592 | 1/12 |
| aut-affine (.2 + .3 w) | .890 | 12/12 |
| aut-sqrt (√w) | .376 | 12/12 |
| aut-dupaction-noise3 | .669 | 12/12 |
| aut-serial250 | .387 | 12/12 |
| **mean** | **.354** | |

Round-7 HELD set, now with the Round-8 model (not clean any more, shown for continuity): mean
**.536** (Round 7: .114; without attention −.140). aut-high .921, aut-square .856,
aut-noise8-sticky .812 (was −.341), canc-noise6 .585 (was −.403), dist-delay4 .694 (was .006),
dist-shift .999, dist-noise4 −.213, aut-fooddelay2 −.409, aut-dupaction .829, aut-serial200 .284.

### Honest status

* Solved in this round: the `dist` regression (.335 → .986), sticky noise (−.111 → .836 DEV, .73–.81
  on held sets), noise in cancellation, delays of 3–4 steps in `dist` (DEV .907, old HELD
  delay4 .694).
* Partly: delay 5 (.254); serial reversal (.475 DEV, .29–.39 held); no savings mechanism yet.
* **Not solved:** i.i.d. noise in `dist` (−.20 on both held sets) and delayed food in BlindGate
  (−.41 / −.59). Brief diagnosis on DEV seeds (no changes made, so HELD2 stays clean):
  in `dist`+noise the agent almost never finds the food and cue 3 earns attention late or never;
  i.i.d. noise onsets are unpredicted (salience about .5), open windows everywhere and take the
  direction out of the event bonus. In BlindGate with delayed food, shocks stay high from the
  start and the precursor → food timing is not learned.
* Still hand-set: the transient-channel anchoring rule (run length ≤ 3), the cancellation and
  blocking constants, the rollout budget (4 channels) and the 2-step depth.
* Cost: about 2.5–4× slower per step (lag tables, enumeration of successors); 16–20-channel
  noise variants take 25–58 s per episode.

## Round 7: transfer to task variations as the main criterion

The Round-6 audit found assumptions tuned to the three base tasks (reward scale, horizon 2,
noise, erasure on reversal, no generalisation, hand rules, expectation windows). The quality
criterion is now **transfer to unseen variations**, not new seeds of the same task.

### Benchmark (`development/transfer_suite.py`)

Score = (brain − stay) / (oracle − stay) over the last 200 steps (1 = oracle, 0 = stay,
< 0 = worse than doing nothing); wins = seeds with brain > stay.

* DEV (configuration choice, env seeds 100+, brain seeds 0+, 8 seeds): `aut`, `canc`, `dist`,
  `aut-half` (all rewards ×0.5), `aut-noise4` (+4 random channels), `dist-delay3` (food 3 steps
  after the cue), `aut-serial300` (reversal every 300 steps).
* HELD-OUT (report only, env seeds 7000+, brain seeds 700+, 12 seeds, run once at the end):
  `aut-high` (reward shift), `aut-square` (nonlinear reward), `aut-noise8-sticky` (8 sticky noise
  channels), `canc-noise6`, `dist-delay4`, `dist-shift`, `dist-noise4`, `aut-fooddelay2`,
  `aut-dupaction` (duplicated action), `aut-serial200`.
* Sanity check: the oracle beats stay in all 17 variations.

### Change 1: reference-point value coding (replaces the fixed reward scale)

The value channel is coded relative to an adaptive reference point (the median of observed
values) with a scale S = max(2·max(hi − ref, ref − lo), `value_scale`), z = .5 + (w − ref)/S.
When the frame changes, every value posterior is updated affinely, so learning is invariant to
affine changes of reward. A purely range-based normalisation (tried first) broke `dist`
(0/8): there the relevant differences (.45 vs .5) are small compared with the range. The
reference point makes them visible, like adaptation in dopamine coding. An empirical-prior
variant and `value_scale` .25 / 1.0 were worse on DEV and removed.

### Change 2: learned attention as a coarse level of the model (against noise)

Attention is learned, not given: a channel becomes *attended* once a 2×2 Beta-Binomial Bayes
factor shows that it depends on another channel at the next step. The test runs per action and
pooled over actions, with a multiple-comparison threshold log(n(n−1)(n_actions+1)/α).
Attention only adds a **coarse level**; the exact tables still see every channel:

* `coarse_factors[(attended part of the context, a, target)]` becomes the prior of an exact
  context. This is partial pooling: the variance τ² among the visited exact children, plus one
  pseudo-child at maximal spread, sets both the uncertainty of an unvisited child and how
  strongly a visited one is shrunk (m = within-cell variance / τ²). Children are weighted by
  min(1, n);
* an unvisited predicted state takes its reward from its attended siblings (`coarse_reward`);
* the uncertainty q about the next state counts only attended channels.

A context that differs only in irrelevant channels is therefore *not novel* (an animal does not
re-learn a lever because the lighting changed), while a channel that does matter keeps its
exact table.

Rejected on DEV (removed from the code):

| variant | DEV mean all / variants |
|---|---|
| hard filtering of all table keys by attended channels | .240 / .021 (dist .096, noise4 −.551) |
| same, only reward / hazard / q | .253 / .034 |
| attention via hazard / expectation-window evidence | noise channels became attended |
| "ignore only proven-independent channels" instead of "attend proven-dependent ones" | .451 / .268 |
| attention also over lags 2..max_age | .495 / .348 |
| children weighted 1 (not min(1, n)) | .361 / .104 (noise4 −.676) |

Ablation of the final version (4 DEV variations: dist / noise4 / aut / serial300): all on
.335 / .345 / .823 / .305. Without the coarse prior: .377 / −.022 / .732 / .111. Without q over
attended channels: .397 / .185 / .818 / .278. Without coarse reward: .347 / .307 / .805 / .298.
The cap on m (2, 10, 50) and a floor on the sd of unvisited children do not change `dist`.

### Change tried and rejected: longer event horizon

`event_horizon` (the window age up to which open expectations get an exploration bonus) was 2,
matching the base task. On DEV, H = 4 gave .508 / .364 (delay3 .069, 7/8 wins; dist .284) and
H = max_age = 12 gave .494 / .355. Both are worse overall than H = 2, which was kept. `dist-delay3`
needs a structural change (see open problems).

### DEV results (8 seeds)

| variation | v5 (Round 5) | + reference point | + attention (final) |
|---|---|---|---|
| aut | .640 8/8 | .717 8/8 | **.823** 8/8 |
| canc | .995 8/8 | .990 8/8 | .990 8/8 |
| dist | **.484** 8/8 | .453 8/8 | .335 8/8 |
| aut-half | .513 8/8 | .717 8/8 | **.823** 8/8 |
| aut-noise4 | −.276 1/8 | −.330 1/8 | **.345** 8/8 |
| dist-delay3 | .065 3/8 | −.017 0/8 | .010 1/8 |
| aut-serial300 | .049 6/8 | .072 8/8 | **.305** 8/8 |
| mean all / variants | .353 / .088 | .372 / .111 | **.519 / .371** |

On 16 additional seeds (env 300+, brain 50+), comparing without and with attention: `dist`
.390 → .303 (16/16 wins in both), `canc` .997 → .994, `aut-noise4` −.421 (0/16) → .337 (16/16).
**The `dist` regression is systematic, not noise.** Late code paths are identical once all
relevant channels are attended (ak = full context), so the loss comes from the early phase via
learning on predictions (onset salience → timing model). Food intake is unchanged; there are
more late errors (moving to the far side without an open window).

### HELD-OUT (12 seeds, run once; control = the same code with attention switched off)

| variation | no attention (control) | final |
|---|---|---|
| aut-high | .634 12/12 | .714 12/12 |
| aut-square | .720 12/12 | .718 12/12 |
| aut-noise8-sticky | −.397 2/12 | −.341 1/12 |
| canc-noise6 | −2.002 0/12 | −.403 1/12 |
| dist-delay4 | −.003 1/12 | .006 1/12 |
| dist-shift | .338 12/12 | .275 12/12 |
| dist-noise4 | −.843 0/12 | −.543 0/12 |
| aut-fooddelay2 | −.360 3/12 | −.090 3/12 |
| aut-dupaction | .542 12/12 | .701 12/12 |
| aut-serial200 | −.031 5/12 | .101 9/12 |
| **mean** | **−.140** | **.114** |

What transfers: reward shift and nonlinearity (aut-high, aut-square), duplicated action, serial
reversals (weakly). Attention improves 8 of 10 held-out variations and the mean by +.25.

What does **not** transfer: noise outside the DEV form is still worse than stay
(sticky noise, noise in canc and dist). On `aut-noise4` (DEV) attention fixed noise: the
noise channels are reliably not attended. Sticky noise and noise in tasks with rare or delayed
events are not yet handled: attention is pooled only at lag 1, and before attention is
established the agent explores the noise combinations. Longer food delays (`dist-delay3/4`,
`aut-fooddelay2`) remain at about stay.

### Open problems (honest status of the Round-6 bottlenecks)

| bottleneck | status |
|---|---|
| reward scale | solved: reference point plus affine re-coding (aut-half = aut, aut-high, aut-square) |
| noise / scaling | partly: random noise in aut is solved; sticky noise and noise in canc/dist are not |
| no generalisation | partly: coarse level over irrelevant channels; no generalisation over time lags |
| erasure on reversal (savings) | not done; serial reversal improved only indirectly (.07 → .31 DEV) |
| horizon 2 / expectation windows | not solved: a longer event horizon is worse; delay ≥ 3 at stay |
| hand rules (cancellation, blocking), slow avoidance | not touched in this round |

## Round 5: speed, a minimal kernel, one optimism budget

### Speed (mathematics unchanged)

Profiling showed the planner re-deriving the same posterior quantities: within one decision the
two-step tree asks for the same (context, action, target, depth) prediction, the same excitation
and hazard terms, the open cues and the pending windows of a target many times; every such call
rebuilt the active set. Changes, all exact:

* per-decision memo for `_predict_core` and caches for excitation, pending windows, open cues and
  window informativeness, cleared by `_invalidate()` on every `learn()` (and by the public
  `predict`/`predict_next`); tests that edit tables by hand call it explicitly;
* the active set is computed once per query and passed down; early exit in `_hazard_boost` when
  nothing is pending;
* `run_posterior_scenarios.py` runs all episodes of the suite in one process pool, longest first.

Before the optimism change the behavioural fingerprint (300 steps × 6 runs, every action and
prediction hashed) was bit-identical to Round 4, and the `--suite` output was byte-identical.

| | Round 4 | Round 5 | speed-up |
|---|---|---|---|
| fingerprint 300 × 6, one process | 9.1 s | 4.1 s | 2.2× |
| `--suite` (2 CPUs) | ~316 s | ~100 s | 3.2× (≈1.6× serial work, ≈2× pool) |
| new `--dev` check | — | ~35 s | |

Tried and dropped: a persistent version-stamped excitation cache (no gain) and sharing memo
entries between depths 0 and 1 (≤ 10 % of calls).

### Minimal kernel

Removed together with their code paths and tests: posterior sampling (`TS`, `ucb+ts_timing`),
`explore_depth`, `idle_horizon`, `event_pu`, Beta-sd event bonus, `event_uninformative_only`,
the `novel_state_prior` switch (now always on) and other dead switches. Constructor:
`n_inputs, n_actions, value_index, gamma, pair_alpha, pair_min_samples, inhibit_threshold,
explore=2, max_age=12, change_hazard=.01, global_hazard=.001, event_explore=.5, event_horizon=2,
tie_seed, seed`.

### Double counting of optimism

On one transition two optimism terms could both fire at the root: `explore·SD[w']`, whose SD
for a never-seen predicted state is the novel-state prior (≈ 0.29 → +0.58), and the open
(cue, τ) expectation bonus. Both are uncertainty about the *same* reward of the *same* transition
(will the predicted state be good / will the expected event occur), so adding their SDs counts a
transition that is both novel and an open test twice. Independent uncertainties add in variance;
`max` would drop a genuine second source. In the rollout there is nothing to merge: imagined
states use plain means, only the event term exists. Diagnosis on DEV (late BlindGate shocks, 8
seeds × last 200 steps): with Round 4, 49 of 87 shocks happened at decisions where a predicted
state was novel at the root or at the second step; without the prior, 37 of 68.

DEV seeds only (8 seeds; tail mean of the last 200 steps; late = shock fraction last 200 steps):

| Variant | autonomous (late sh.) | cancellation | distant (late sh.) | reversal before/after |
|---|---|---|---|---|
| Round 4 (sum of bonuses) | 0.571 (0.054) | 0.552, 8/8 | 0.526 (0.014) | 0.562 / 0.555 |
| no novel-state prior | 0.578 (0.042) | 0.552, 8/8 | 0.528 (0.009) | 0.552 / 0.541 |
| novel-state prior at root only | 0.580 (0.036) | 0.552, 8/8 | 0.526 (0.014) | 0.570 / 0.547 |
| max(novel, event) per transition | 0.577 (0.046) | 0.552, 8/8 | 0.523 (0.018) | 0.558 / 0.557 |
| **one budget sqrt(a² + b²)** | **0.570 (0.049)** | 0.552, 8/8 | **0.535 (0.001)** | 0.562 / 0.550 |
| one budget, no novel-state prior | 0.586 (0.032) | 0.552, 8/8 | 0.500, 4/8 (0.053) | 0.561 / 0.547 |
| one budget + prior at root only | 0.562 (0.058) | 0.552, 8/8 | 0.535 (0.001) | 0.555 / 0.552 |

Differences on BlindGate are within seed noise (SD across seeds ≈ 0.015). Chosen: the single
budget — it is the theoretically motivated one-line rule, no variant keeps all three tasks at 8/8
and is clearly better, and it is equal or better than Round 4 on every DEV figure except reversal
after (−0.005). "No novel-state prior" also passes DEV but the prior is needed as soon as the
bonuses are merged (distant collapses to 4/8), so both mechanisms stay.

### Final model, held-out seeds (12, chosen before looking)

| Task | Round 5 | late shocks | whole-run shocks | oracle | > stay | Round 4 |
|---|---|---|---|---|---|---|
| autonomous | **0.576** | **0.045** | 0.086 | 0.611 | 12/12 | 0.563, late 0.065, whole 0.095 |
| cancellation | 0.551 (= oracle) | 0.000 | 0.027 | 0.551 | 12/12 | 0.551, 0.000, 0.029 |
| distant-value | 0.532 | 0.005 | 0.056 | 0.573 | 12/12 | 0.529, 0.007, 0.056 |
| reversal before/after | 0.558 / 0.548 | | | | | 0.561 / 0.545 |
| timing lag 2/3/4 | 0.31 / 0.48 / 1.00 (true 0.33 / 0.50 / 1.00) | | | | | same |

Sign test p = 0.0002 on every task. Equivariance 1440/1440, probes unchanged (AND 1 birth,
noise 0, A/AX/B/BX 0.999/0.0001/0.99996/0.13). The BlindGate gap to the oracle (0.035) and the
exploration cost before the first distant meal remain.

## Round 4: stuck cancellation seeds and the distant-value barrier

Same protocol: every setting chosen on the 8 DEV seeds (env 100+, brain 0+); every table below is
**HELD-OUT** (12 seeds, env 5000+, brain 900+, last 200 steps). New defaults:
* `novel_state_prior=True`;
* `event_explore=0.5`, `event_horizon=2`;
* the imagined successor never activates the value channel;
* `idle_horizon=0` (tested, not needed).

### Final model, held-out seeds (stay = 0.5)

| Task | Round 4 | late shocks | oracle | > stay | sign-test p | Round 3 |
|---|---|---|---|---|---|---|
| autonomous (BlindGate) | 0.563 | 0.065 | 0.611 | 12/12 | 0.0002 | 0.564, 12/12, late shocks 0.043 |
| cancellation | **0.551** (= oracle) | 0.000 | 0.551 | **12/12** | 0.0002 | 0.532, 8/12 |
| distant-value | **0.529** | 0.007 | 0.573 | **12/12** | 0.0002 | 0.499, 0/12 |
| reversal, 100-step windows from t = 300 | 0.562 → **0.454** (flip) → 0.520 → 0.546 → 0.543 → 0.567 → 0.568 | | | | | frozen control 0.20–0.23 |
| timing (lag 2/3/4) | 0.31 / 0.48 / 1.00 (true 0.33 / 0.50 / 1.00) | | | | | same |

Whole-run shock fractions from `--suite`: autonomous 0.095, cancellation 0.029, distant 0.056.
The distant cost is the exploration before the first meal (around t = 55). Closed-loop
equivariance: 1440/1440 exact. Probes are unchanged (AND 1 birth, noise 0, A/AX/B/BX
0.999/0.0001/0.99996/0.13).

### Ablation, held-out (for reporting only)

| Config | autonomous (late sh.) | cancellation | distant |
|---|---|---|---|
| A: Round 3 final | 0.564 (0.043) | 0.532, 8/12 | 0.499, 0/12 |
| B: A + novel-state prior + imagined-value fix | **0.580** (0.038) | 0.550, 12/12 | 0.499, 0/12 |
| C: B + expected-event bonus (**final**, chosen on DEV) | 0.563 (0.065) | 0.551, 12/12 | 0.529, 12/12 |
| D: A + imagined-value fix + event bonus, no novel-state prior | 0.576 (0.047) | 0.550, 12/12 | 0.530, 12/12 |

Distant trips after a cue at the base (t ≥ 400, 12 held-out seeds):

| | full (L, L) | half | late | meals in the last 200 steps |
|---|---|---|---|---|
| A (Round 3) | 0.00 | 0.07 | 0.01 | 0 on every seed |
| C (final) | **0.50** | **0.00** | **0.00** | 15–18 on every seed |

**Not hidden:**
* On held-out, D beats the chosen C on BlindGate (0.576 vs 0.563, late shocks 0.047 vs 0.065).
  On DEV the difference was within noise. C stays the reported model because switching now
  would be selection on the test seeds.
* The event bonus raises late BlindGate shocks from 0.038 (B) to 0.065. Exploration is not free.

### 1. Stuck cancellation seeds: what was actually wrong

* **The failing seeds never tried "left on cue alone".** Their only left trial was cue + trap →
  shock, and food had never been seen at position 0. The factor prior for (cue, left) was
  therefore 0.06 ± 0.25, about 0.05 short of stay.
* **Fix 1: novel-state reward prior.** Through controllability pooling (Round 3), the model
  already predicted "after left on the cue: position 0 + food visible". That state had never been
  observed, so the reward model returned nothing. It now returns the uninformed prior (0.5 ± 0.29)
  combined by precision. Cue + trap still predicts "position 0, no food", a familiar state worth
  0, so the inhibitor keeps that action unattractive.
* **Fix 2: imagined-value bug.** In imagined successors, a predicted value mean of 0.503 was
  thresholded as "value channel active", a context that never occurs. It produced spurious
  continuation values (0.89 after "stay" in distant).
* **Neither fix works alone.** DEV: prior alone 6/8, bug fix alone 6/8, both 8/8. Held-out: 8/12
  → 12/12 at oracle level, late shocks 0.003.
* **Idle-time confidence decay (the suggested alternative) was not needed.** I implemented
  hyperbolic survival T/(T+d) for unvisited cells. T = 100–300 brings back repeated shocks; T ≥ 1000
  helps only together with the novel-state prior, and the bug fix makes it unnecessary. The option
  remains, off (`idle_horizon=0`).

### 2. The 0.45 < 0.50 barrier in distant value

Before the first meal, nothing observable says the corridor pays off: cue 3 predicts no channel
until the agent is at position 0 at lag 2. What got through the barrier was a selective bonus for
*untested combinations (state, open window (cue, τ), action)* in both the root and the step-2
evaluation. Each restriction below was needed:

| Restriction | Reason it is needed (DEV evidence) |
|---|---|
| applied at **both** depths with the same count | the plan and its second step agree, so there are no half trips (0.00) |
| count of the **exact** (cue, lag, action, context) cell | a generic (cue, lag, action) count is satisfied by "late" trips from the base and reintroduces τ-confusion |
| weight = **salience** = 1 − P(onset) before it happened | food after its cue, or a position after the agent's own move, has salience ≈ 0. Weighting by P(uncontrollable) instead was unstable in distant (0.28–0.98 across seeds and time), because the cue is only shown at the base |
| only cues with **no located effect** (no informative lag) | without it (and before the horizon limit), BlindGate cue windows at lags 3–10 caused about 7% bonus-driven shocks |
| only lags within the **planning horizon** (≤ 2) | noise / trap windows at long lags: with no horizon, BlindGate shocks were 0.15 and cancellation 0.08 (whole run). With horizon 2: 0.087 and 0.033 |
| imagined value channel never active | without it 5/8 DEV seeds; three seeds lost exploitation for 400+ steps after the first meal |

The bonus is (1 − E[w]) · salience / (1 + n): the headroom times count-based novelty, an
R-max-like optimism that fades after a few tests. Once the agent has eaten, exploitation comes
from the real posterior (hazard lift and τ-aware memory). On DEV, κ ∈ {0.5, 1, 1.5} are all 8/8
on distant; 0.5 was chosen for the lowest shocks.

### Status after Round 4

| Ability | Held-out |
|---|---|
| cue-following | ✅ 0.563, 12/12 (late shocks rose to 0.065) |
| conditional inhibition (cancellation) | ✅ 0.551 = oracle, 12/12, late shocks 0.000 |
| distant value (two-step timing) | ✅ 0.529, 12/12 (oracle 0.573), first meal around step 55 on every seed |
| reversal | ✅ dip 0.454, recovers to 0.57 (frozen control 0.21) |
| timing calibration | ✅ |

Open problems:
* The distant gap to the oracle (0.529 vs 0.573) is mostly the return trip and waiting for the
  next cue.
* The event bonus costs BlindGate safety. It is a noisy-TV-type cost, partly controlled by
  salience.
* The novel-state prior interacts negatively with the bonus on BlindGate (C vs D on held-out).

---

## Round 3: three principles (surprise forgetting, τ-synchronisation, observational inhibition)

Defaults were fixed on the DEV seeds: `change_hazard h = 0.01`, `global_hazard H = 0.001`, no
blind decay, root UCB κ = 2, reward model on. All tables below are **HELD-OUT** (12 seeds, env
5000+, brain 900+, last 200 steps).

### Final model, held-out seeds (stay = 0.5 everywhere)

| Task | PosteriorGraph | late shocks | oracle | brain > stay | sign-test p | previous round |
|---|---|---|---|---|---|---|
| autonomous (BlindGate) | **0.564** | 0.043 | 0.611 | **12/12** | 0.0002 | 0.545, 12/12 |
| cancellation | **0.532** | 0.004 | 0.551 | 8/12 (4 seeds exactly 0.5) | 0.19 | 0.519, 12/12 |
| distant-value | 0.499 | 0.000 | 0.573 | 0/12 | 1.0 | 0.473, 0/12 |
| reversal (100-step windows from t = 300) | 0.556 → **0.466** (flip) → 0.524 → 0.542 → 0.558 → **0.583** → 0.576 | | | | | frozen control 0.22–0.24 |
| variable delay (hazard calibration) | 0.31 / 0.48 / 1.00 vs true 0.33 / 0.50 / 1.00 | | | | | 0.30 / 0.46 / 1.00 |

Probes: AND → 1 birth, p = 0.9995; noise → 0 births; A = 0.999, AX = 0.0001, B = 0.99996,
BX = 0.13 (previously 0.06: transfer is still clear but weaker). Closed-loop equivariance: 1440/1440
exact.

### Ablation on held-out seeds (used only for reporting, not for choosing)

| Config | autonomous | late shocks | cancellation | late shocks | distant | late shocks |
|---|---|---|---|---|---|---|
| A: previous round (blind decay 0.995) | 0.542 (12/12) | 0.080 | 0.517 (12/12) | 0.056 | 0.479 (0/12) | 0.057 |
| B: + change-point forgetting + τ memory | 0.555 (12/12) | **0.034** | 0.522 (7/12) | **0.001** | 0.499 (0/12) | 0.000 |
| C: + controllability + reward model | **0.564** (12/12) | 0.043 | **0.532** (8/12) | 0.004 | 0.499 (0/12) | 0.000 |

Reversal recovers in all three configs. C reaches the highest post-reversal level (0.583).

### Principle 1: surprise-modulated forgetting — confirmed, with one cost

Implementation:
* **Local:** before absorbing y, each cell discounts its counts by
  `c = σ(logit h + S)`, where `S` is a CUSUM of `log p₀(y)/p(y|cell)`. Evidence used up by a
  discount is spent.
* **Global:** a joint likelihood ratio over all channels of the transition gives
  `P(global change)`, which discounts *every* cell, including actions that are no longer taken.

Findings:
* **Repeated shocks fall** from 0.080 to 0.034 (BlindGate) and from 0.056 to 0.001
  (cancellation). Blind decay was indeed making the agent retry known-dangerous actions.
* **Local forgetting alone could not support reversal.** The new good action had a long
  confident history of shocks and was never retried (recovery stalled at about 0.53). With the
  global level, the change posterior is about 0 in stationary phases (Σc ≈ 0.01 per 100 steps),
  spikes at the flip (Σc ≈ 1.1), and recovery goes beyond the pre-flip level.
* A one-step change-point approximation adapts too slowly (12 surprises → p = 0.19). The
  accumulated CUSUM version halves the memory after 2 surprises.
* **Cost:** without regrowing uncertainty, an early pessimistic estimate is permanent. In
  cancellation, 4 of 12 seeds get locked at exactly 0.5: after an early trap shock, *left* on the
  cue alone is never retried, and the reward model has no cell for "position 0 + food visible".
  The mean rises while reliability falls (12/12 → 8/12).
* Memory of a perfectly predicted cell levels off near 2/h ≈ 200 **visits** (the change-point
  prior), but a cell with no visits does not forget at all.

### Principle 2: τ-synchronisation and deep exploration — partly confirmed

* **The planner was already partly correct.** Hazard is looked up at lag `τ + 1 + depth`, so τ
  advances in the rollout, and context-specific timing cells exist.
* **The gap was the belief-state level.** It keyed only on *which* windows were open, so "one
  step after the cue" and "two steps after" were the same state. It now keys on informative
  windows `(cue, τ + depth)`, where informative means a credibly non-baseline hazard at some lag.
* **Consequence:** once the memory stopped pooling timed successes with late failures, the trip
  was no longer discovered by luck. The agent now **never** attempts it (shocks 0.000), instead
  of attempting it badly (shocks 0.057). The score is stay-equivalent (0.499).
* **Deep exploration results** (8 DEV seeds, trips after a cue at t ≥ 400):

| Exploration | full (L,L) | half (L, back) | late (wait, L) | meals | global effect |
|---|---|---|---|---|---|
| root UCB (final) | 0.00 | 0.07 | 0.01 | 0 | — |
| root UCB, blind decay (previous round) | 0.26 | 0.15 | 0.08 | 81 | — |
| TS, commitment 1 | 0.31 | 0.12 | 0.12 | 81 | shocks 20% on all tasks |
| **TS, commitment 3** | **0.32** | **0.04** | 0.10 | **86** | shocks 23% on all tasks |
| UCB in the continuation (κ_c = 1, 2) | — | — | — | — | worse on every task |
| UCB + TS on timing cells only | — | — | — | — | shocks 18–19%, no gain |

  **Committed posterior sampling does remove the half trips** (0.15 → 0.04) and gives the most
  meals. But applied to the whole value function it injects optimism everywhere, and shocks rise
  to about 20% on every task. Sampling only the timing uncertainty doesn't work either: some
  open-window cell is always unsampled, so it acts as noise. All variants keep exact equivariance
  (per-action quantiles from the mapped tie-noise stream). The default stays root UCB. **Safe
  deep exploration remains open.**

### Principle 3: conditional inhibition by observation — confirmed, but it needed two links

* **The inhibitor already learned by observation.** In cancellation, the trap → food-lamp
  (channel 5) contrast reached 0.96 from 47 trials observed at base, with no shock.
* It did not reach behaviour for two reasons:
  1. **Action slicing.** P(food | cue, *left*) = 0.08 against 1.0 under stay.
  2. **No link between the observed food event and value.**
* **Added: controllability.** The within-context Beta-Binomial BF (action-specific vs shared
  rate) classifies channels without hints: positions and w are controllable, while cue, food,
  trap and noise come out uncontrollable (P ≥ 0.99). Pavlovian channels share an action-pooled
  exact context. P(food | cue, left) is now 0.96, and P(food | cue + trap, left) = 0.0.
* **Added: a reward model.** `R(next observation) → w`, applied to the predicted next state, is a
  precision-weighted prior for the value. Its SD includes the spread of w given the state; an
  earlier version without that spread overrode inhibition, which a unit test caught.
* **Effect (B → C):** cancellation 0.522 → 0.532 (7 → 8 of 12 seeds), BlindGate 0.555 → 0.564,
  and the best reversal recovery. The remaining 4 locked seeds have never been at "position 0
  with food visible", and an exact-state reward model can't generalise to an unseen state.

### Decision for this round

| Ability | Status (held-out) |
|---|---|
| autonomous cue-following | ✅ 0.564, 12/12, best so far |
| cancellation / conditional inhibition | ⚠️ best mean (0.532) and near-zero shocks, but 4/12 seeds stuck at stay (p = 0.19) |
| reversal | ✅ dip and recovery to 0.58 (frozen control 0.22) |
| timing (hazard) | ✅ best calibration so far |
| distant value | ❌ stay-equivalent; committed sampling fixes half trips locally but is unsafe globally |

Open problems: (1) exploration that regrows only where the value of information justifies it, so
that both lock-in and repeated shocks are avoided; (2) a reward model that generalises over
states (factorised R instead of exact keys); (3) safe deep exploration for the two-step trip.

---

# Previous round (kept for comparison)

## Model

All statistics are conjugate counts on [0, 1] (Beta-Bernoulli for 0/1 outcomes, Welford moments
for fractional ones). There is no logistic regression and no iterative optimisation. Every
counter decays by `decay = 0.995` per step.

**Prediction is a hierarchical back-off:**

```
factor prior  ->  exact observed context  ->  exact context + open expectation windows
(generalises)     (what was actually seen)    (belief-state memory)
```

Each level is the prior of the next, with `context_shrink = 2` pseudo-counts. The factor prior's
pseudo-count is scaled by `n_min/(1+n_min)`: it is only as reliable as its least-supported active
cue under this action. The exact-context keys are visited sets (linear in experience, not a
powerset). They are sets of channel indices, so relabeling equivariance is preserved.

| Mechanism | Implementation |
|---|---|
| Action slices | Factors are keyed `(context, action, target)`. |
| Pair births | A 2×2 interaction Bayes factor (both-active rate vs exactly-one-active rate, Beta-Binomial). The threshold `log(C(n,2)·\|A\|/α)` corrects for all pairs × actions. |
| Singleton combination | Present vs **absent** posteriors, shrunk toward the slice root. Enabling cues combine by noisy-OR; suppressive cues only lower the baseline. Born pairs own their trials (singletons use residual counts). Kamin blocking applies when a strong cue already explains the outcome. |
| Inhibition (summation) | `p = p_exc·(1 − P_inhib)`. `P_inhib` is a target-level 2×2 cancellation contrast (X present vs absent, given Y expected). It is tied to Y, not to the pair AX, so it transfers to BX. The excitor is protected only when an established inhibitor is present. |
| Hazard timing | Windows open on rising edges of transient channels. On recurrence of the cue, the open window is right-censored and a new one starts (renewal). Each `(cue, target, lag, action)` cell is backed off to a cell that is also conditioned on the exact context. An unobservable target is censored. An event consumes its window. The lift over the root event rate applies only when it is credible (more than 2 SD). |
| Planning | `Q(b,a) = E[r\|b,a] + κ·SD[r\|b,a] + γ·max_a' E[r\|b̂',a']`, with γ = 0.9 and κ = 2. Optimism applies **only at the root**: imagined successors use plain means. |
| Ties | Explicit per-action `tie_noise`, which gives exact closed-loop coupling under relabeling. |

## Component results

| Probe | Result |
|---|---|
| AND structure / pseudo-random noise | 1 birth, P = 0.995 / 0 births ✅ |
| Summation: A+, AX−, B+ → BX? | A = 0.99, AX = 0.0002, B = 0.999, **BX = 0.06** ✅ |
| Hazard, delay U{2,3,4}, 5 seeds | 0.30 / 0.46 / 1.00 vs true 0.33 / 0.50 / 1.00 ✅ |
| Closed-loop equivariance | 3!×3! relabelings × 40 steps = 1440 paired steps, exact ✅ |

## Behavioural results

The hyperparameters κ and decay were selected on the **DEV** seeds (8, env 100+). The **HELD-OUT**
seeds (12, env 5000+, brain 900+) were never used for any decision. Scores are the mean of the
last 200 steps. Stay earns 0.5 everywhere.

| Task | split | PosteriorGraph | shocks | stay | random | oracle | brain>stay | sign-test p |
|---|---|---|---|---|---|---|---|---|
| autonomous (BlindGate) | dev | 0.549 | 0.10 | 0.5 | 0.24 | 0.61 | 8/8 | 0.004 |
| | **held-out** | **0.545** | 0.11 | 0.5 | 0.25 | 0.61 | **12/12** | **0.0002** |
| cancellation | dev | 0.517 | 0.08 | 0.5 | 0.21 | 0.55 | 8/8 | 0.004 |
| | **held-out** | **0.519** | 0.07 | 0.5 | 0.20 | 0.55 | **12/12** | **0.0002** |
| distant-value | dev | 0.470 | 0.10 | 0.5 | 0.31 | 0.57 | 0/8 | 1.0 |
| | **held-out** | **0.473** | 0.10 | 0.5 | 0.32 | 0.57 | **0/12** | 1.0 |
| reversal | held-out | 0.550 before → 0.547 after | | | | | | |
| variable delay | actions have no effect; judged by hazard calibration | | | | | | | |

**Reversal** (held-out, 50-step windows from t = 300): 0.545, 0.561, **0.422** (right after the
flip), 0.499, 0.508, 0.550, 0.553, … The agent recovers in about 150 steps. A control with
learning frozen after the flip scores 0.335 at t = 600–800.

### Honest qualifications

* **Cancellation is solved by the exact-context level, not by the inhibitor.** Late in a run
  the agent goes left on the cue alone (17/17, reward 1.0) and stays when the trap is shown
  (18/18). But `P_inhib` stays 0: exploration avoids trap+left, so that cell gets only about 2
  trials and the inhibitor's contrast requires at least 3. Inhibition and transfer are
  demonstrated only by the component probe.
* **Distant-value fails, but not trivially.** The agent learned the left-left trip after the
  cue: 65 meals in 4×200 steps, about half the oracle's rate. The losses come from mistimed
  trips (most often one step late) and attempts with no cue pending. Each costs 0.5 relative to
  staying, which cancels out the meals. Fixing this needs deep (multi-step, committed)
  exploration and timing that is sharp enough to separate lag 2 from lag 3.
* Margins are small in absolute terms (oracle − stay = 0.05–0.11). The agent recovers about 40%
  of the oracle margin on BlindGate and about 37% on cancellation.

## What changed in this round (each change from an instrumented failure)

1. **Exact-context back-off.** Noisy-OR double-counted correlated cues (Q(stay at center) = 0.58
   where the truth is 0.5), and `predict_next` was blurry. Result: BlindGate 0.50 → 0.515,
   reversal adapts.
2. **UCB on the posterior SD** instead of 1/(1+n). On its own it was worse: optimistic values of
   imagined successor states favoured *stay*. **Root-only optimism** fixed that.
3. **Decay 0.99 → 0.995.** About 100 steps of memory was shorter than the time needed to sample
   the relevant (context × action) cells.
4. **Prior weight tied to the least-supported cue.** The factor prior "center+left = shock"
   wrongly suppressed exploration of left when cue3 had never been seen under left.
   Cancellation went 0/8 → 8/8.
5. **Context-specific timing cells.** The trial at the same lag from position 2 diluted the
   timing cell for the trip from position 1.
6. **Renewal bug.** A window that never closed blocked re-anchoring, so later events were logged
   at lag 7 instead of lag 2. Found by a new unit test.
7. **Belief-state level** (exact context + open windows). After a timed meal, "position 1, left"
   pooled timed successes with untimed failures. Cancellation held-out went 11/12 → 12/12.
8. **Removed:** adding the timing-cell SD to the bonus. Once renewal was fixed, every action
   always had some unsampled cell, so it acted as uniform noise.

## Decision

Four of the five abilities hold up statistically on held-out seeds: autonomous cue-following,
cancellation behaviour, reversal adaptation, and timing calibration. Equivariance and the
component probes (structure, summation) pass. **Distant-value does not pass.** The next step is
deep exploration (for example posterior sampling with synchronised randomness, which preserves
exact equivariance) and more precise lag-specific timing for the trip.
