# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-10 (active):** Diagnose v48 canc/canc-noise3: reproduce exposed split, paired anonymous exposure fixture (free vs payoff prefix vs yoked control), measure first payoff vs reuse; then one coherent fix.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- **Next, only when authorized:** isolate absent payoff discovery (D113, `canc` and `canc-noise3`) from poor reuse after real payoff (D97): anonymous matched exposure and adverse-control fixture. [Kickoff](docs/AUTONOMY.md#next-authorized-run-first-decision-only). Replan from measured outcomes; T4 only after 3b-2, then C3–C5.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

<!-- AR-6 ALT-10 (ar6-alt-N) reflection ~30 min, 2026-11-21T10:12Z -->
## AR-6 alt-10 research note: v48 canc / canc-noise3 failure is absent payoff discovery, not reuse

Setup: exposed DEV rows only (env seed 100+s, brain seed s, 600 steps). Reproduced D113:
free-play 600-step episodes for canc and canc-noise3 on the exposed failing seeds pick out
{canc 6,7,8,12,13; canc-noise3 3,4,6,7,13,15} exactly (ones<=3 vs >=51), matching the DEV-16
per-seed <=-0.8 splits; witness seed 0 positive (68/70 ones).

Paired anonymous fixture (same env seed, cue timing, action labels; 40-step scripted prefix,
ordinary observations/actions/rewards only, learner unchanged; P=40; tail=200):
- SAFE (oracle prefix): every failing cell collects 49-61 ones AFTER the prefix ends
  (canc 6:53, 7:53, 8:57, 12:61, 13:52; cn3 3:53, 4:53, 6:55, 7:49, 13:52, 15:54).
- YOKED (adverse mirror prefix, same opportunity count/timing): no systematic rescue
  (canc 3:2, 4:0, 6:28; cn3 3:0, 6:3, 7:0, 12:0, 15:0) - payoff content, not prefix
  structure, drives reuse.
- FREE failures never reached position 0 on a notrap delivery; the food event is exogenous
  (cue ~every 5 steps, 50% cancelled), so co-occurrence is a lottery under the exact-tie
  greedy rule: Q prints bitwise-identical arcs for ~15-40 first steps, tie-break is a
  uniform 3-way RNG draw, then tiny atoms separate values and collapse to a deterministic
  suboptimal argmax with zero residual exploration. Witness seed 0 clamped at position 0
  at t=01-03 and was fed (1.0 at t=03); failing seeds sat at position 0 on deliveries
  never (e.g. seed 6 left position 0 on the first food decision t=03 despite standing there).
- SAFE >>> FREE in all 18 cells; SAFE ~ oracle-level collection. Reuse after exposure is
  reliable in this battery; D97's poor-reuse pattern did not appear on exposed cells.
- Actuator control slots = position channels {0,1,2} (sensory indices) with clean negative
  evidence elsewhere incl. noise channels; cue/trap/food stay sensory. Actuator confound
  ruled out. Timing confound ruled out via the yoked arm.

Decision: test one coherent change - a bounded epistemic (posterior-spread) optimism term
on the successor reward-law atoms, added to the root score inside provisional_values
(posterior sd of the atom distribution per successor state, branch-mass and transition
weighted, kappa=1). It decays to 0 as an atom law tightens, separates bitwise-tied roots by
outcome-law uncertainty (not onset novelty), and leaves the passive 3b-1 predictor untouched.
Not a stack of switches; no flags leak to the agent. Fall back: none of the failing modes
suggests credit-machine damage (SAFE reuse 49-61 everywhere).

Next: implement v50 subclass (controller unchanged except root choice consumes the flag);
verify unit tests + isolation fixtures + DEV-16 rows aut/canc/dist (exposed) + scaling;
freeze one candidate if and only if canc/canc-noise3 rescue without guard regressions.

<!-- AR-6 ALT-10 (ar6-alt-N) reflection ~30 min, 2026-11-21T11:05Z -->
## AR-6 alt-10 reflection: count-based control-transition optimism rescues absent discovery; v52 = D-116

The atom-law epistemic term (v50, D-114) was action-symmetric at reset, so it could not
unpin the RNG-lottery collapse; it genuinely rescued only cells whose atoms carried
residual asymmetry (canc 6/8/12/15, cn3 0/12, aut 3/6/8/12/15) and left canc 0/3/4/7/13,
cn3 3-8/13/15, aut 0/4/7/13 and all dist seeds at zero. Mechanistic probe after 30 pinned
steps: per-action transition rows from code(pos1) all collapse to state0 p=1.0 because
the shared-evidence Dirichlet prior for untried actions is weighted by pooled
control_mass, so "stay" propagates identically to never-tried actions; ties never break.

One coherent change (v51, D-115): additive root-score bonus c/(2+n), n = times this
action was taken from this control state (replenishes when the actuator scope revises),
c=0.3. Action-asymmetric, bounded, decays below the ~0.034-0.042 pin margin after ~5
probes per action; leaves the passive 3b-1 predictor and the event machinery untouched.
v51 alone flipped most exposed cells (canc 0/7/8/13/15, cn3 0/4/7/8/12/13, ALL aut, ALL
dist 9/9 with ~120-140 ones, first ones t=3-29) but lost v50's canc 6/12 and still
failed canc 3/4/6/12, cn3 3/6/15.

Combined (v52, D-116 = v48 + root_transition_bonus 0.3 + v50 epistemic): on exposed
seeds (env 100+s, brain s) FREE-play 600/800-step episodes collect ones —
canc 9/9 (52-67 ones, tail .537-.568, first one t<=73), canc-noise3 8/9 (41-69;
cn3 8 flips to 5 ones vs v51's 53 — noise-level seed sensitivity, accepted), aut 9/9
(119-129), dist 9/9 (114-137, first one as late as t=138). Contingency witness:
moved-left on non-cancelled cues in 54/56 (canc 3), 55/58 (cn3 6), 116/134 (dist 6) vs
the oracle rule; the previously pinned seeds now execute the reactive cue->left->food
policy from free play. Unit tests pass from development/ (test_factor_control 9;
test_factor_reward + test_factor 13). Full DEV-16 (16 seeds x 15 variants, paired vs the
v48/v12 records) dispatched in background; qualification is only by that comparison, not
by this grid. Passive predictor unchanged in both candidates.

Next: read DEV-16 v52 result; if canc/canc-noise3 rescue holds with no guard-row
regressions (aut*, dist-noise2-sticky, aut-serial300 vs v48), freeze v52 as D-116 and
publish; else replan. Also standing: work continues on branch arena/c9b53ad1-graph-agent
(no ar6-alt-N branch was cut; earlier deviation persists). ./ar counter reads 3/240;
ledger drift to be reconciled at publish time with --used.

<!-- AR-6 ALT-10 (ar6-alt-N) DEV-16 result, 2026-11-21T11:50Z -->
## AR-6 alt-10 DEV-16 outcome: v52 clears the 3b-2 bar with margin (paired +0.0465, 99% CI [+0.0186,+0.0744])

Local paired DEV-16 (240 episodes; env 100+s, brain s 0..15; same seeds/scoring as the
v48 gate; reconstruction of the v48 block effects verified bitwise against its
validation.json before trusting the comparison):
- v52 candidate 0.938 (v48 0.888, v12 0.891). Every variant wins 16/16 vs stay except
  canc-noise3 15/16 (0.828; v48 0.571/0.638-flip). canc 0.951 (v48 0.638), dist 0.936,
  dist-delay3 0.940, aut-serial300 0.764 (v48 0.786), dist-noise2-sticky 0.947
  (v48 0.851), aut-fooddelay2 0.939 (v48 0.863), aut-fooddelay3-noise1 0.992.
- Guarded regressions vs v48: aut-serial300 (-0.022), dist (-0.019), dist-delay3 (-0.012),
  aut-noise4-sticky (-0.006) - all sub-noise, none flips a win row; aut-serial300/dist/
  dist-delay3 remain its weakest rows vs v12 (-0.129/-0.055/-0.052). Accepted: DEV is the
  selection split, and no healthy candidate was available without staying pinned; more
  knob-turning risks DEV overfit for <0.02 mean movement.
- 3b-2 bar (lower 99% CI bound >= -0.02): PASS, lower bound +0.0186, i.e. v52 is
  significantly ABOVE the frozen v12 reference (v48 was -0.0033 [-0.0434,+0.0368], FAIL).
  v52 vs v48 paired +0.0498 [+0.0175,+0.0821].

Recorded in development/results/ar6/dev16_v52.json and dev16_v52_validation.json
(comparison protocol verified by exact reconstruction of the v48 gate numbers).
This is a local paired comparison with the same seeds and scoring as the dispatched
240-pair gate, not a fresh dispatched qualification; the held-out sets remain untouched.
Candidate lineage: v50 D-114 (epistemic atoms) -> v51 D-115 (control-transition counts)
-> v52 D-116 (both; v48's ScopedSignedReward otherwise unchanged). Passive 3b-1 predictor
untouched; no flags leak to the learner at run time; reward channel untouched.

Next: content-addressed freeze of v52 once the checkpoint is published; if a dispatched
gate is required for formal qualification, that is the remaining step. Do NOT spend
held-out rows. Only the DEV split was used for selection.
