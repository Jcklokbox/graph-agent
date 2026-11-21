# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-5 (active):** Diagnose v48 canc discovery vs reuse; trace seed6 vs seed0; test one coherent M_factor fix
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- **Next, only when authorized:** isolate absent payoff discovery (D113, `canc` and `canc-noise3`) from poor reuse after real payoff (D97): anonymous matched exposure and adverse-control fixture. [Kickoff](docs/AUTONOMY.md#next-authorized-run-first-decision-only). Replan from measured outcomes; T4 only after 3b-2, then C3–C5.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## Research note (ar6-alt-5, wall ~500s): v48 canc failure is discovery deadlock, not reuse

**Causal finding.** On `canc` seed 6 (env 106, brain 6) v48 gets 0 payoffs in 600 steps (60 cues, 25 payoff opportunities, tail 0.50) while seed 0 gets first payoff at step 3 and tail 0.565. Matched exposure (scripted oracle prefix 150 steps, then free) rescues v48 to tail 0.5525 with 58 payoffs; adverse stay-prefix delays first payoff to step 158 but still discovers (13 payoffs, tail 0.5225). So post-exposure reuse is intact; free discovery is absent. D97 poor reuse (v36) is a separate witness, not v48's failure.

**Trace mechanism.** First cue (t=3, no trap) is a three-way Q tie (1.583 each); tie-break is a lottery (seed 0 picks left and hits, seed 6 picks right and misses). After the miss, cue marginal evidence never exceeds the parent threshold (`rule_w` all False, `evidence_type` 0), so D-108's `context_requires_parent` gate blocks `pair_context` (offer 0.0 always, no signed splits). Unary provisional optimism is pooled over trap/no-trap, so trap shocks (r[left]≈0.05 vs r[middle]=0.5) extinguish it; Q(stay) > Q(left) by ~0.04 thereafter. v44 (same stack without the gate, `same|~any`) discovers seed 6 at step 83 (tail 0.5525) and all DEV-16 failing clean seeds (6,7,8,12,13). Guards probed (aut, dist, aut-noise4, dist-noise2, aut-serial300, dist-noise2-sticky, aut-fooddelay2; seeds 0,6) are identical v44 vs v48. Noisy `canc-noise3` seed 6: v44 discovers at 83 (tail 0.5175, score 0.318) vs v48 0.0; v48 with payoff exposure reaches 0.55, so noisy reuse after discovery is partial (36/40 oracle-correct in tail, trap split learned with n_trap=1).

**Implication.** Parent admission by marginal reward evidence deadlocks configural discovery: payoff needs configural optimism, optimism needs parent, parent needs payoff. The fix is sensory qualification: configural hypotheses are proposed by sensory association (`assoc`, already thresholded) and evaluated by reward; when no reward parent is earned, uncertainty is maximal so hypotheses are qualified. This removes a scope condition (L-9) rather than adding one. Next: implement as v50 (one change in reward qualification), test clean/noisy canc + guards, then scaling/isolation/parity.

## Research note (ar6-alt-5, wall ~1500s): v50 implements sensory qualification; checks pass; freeze pending

**Change (one, D-114).** `development/agents/v50.py`: `SensoryQualifiedReward(ScopedSignedReward)` with `context_requires_parent = False`. Configural (`pair_context`, `same|~any`) hypotheses are proposed by the sensory `assoc` statistic (already thresholded at log1p(J)) and evaluated by the reward path exactly as before; when no reward parent is earned, uncertainty stays maximal so hypotheses remain qualified. This removes the D-108 scope condition instead of adding a switch (L-9/L-1/L-4). No other line differs from v48.

**Evidence (exposed seeds only, 600-step episodes, tail = last 200).**
- `canc` (env 100+s, brain s): seed 0 first=3/pay=68/tail=0.5650; seed 6 first=83/pay=54/tail=0.5525; seed 7 48/52/0.5600; seed 8 73/54/0.5625; seed 12 13/61/0.5550; seed 13 98/46/0.5450. All five DEV-16 failing clean seeds now discover; every number equals v44 on the same cell.
- `canc-noise3`: seed 0 3/70/0.5625; seed 3 38/51/0.5375; seed 4 8/54/0.5175; seed 6 83/53/0.5175; seed 7 48/51/0.5400; seed 13 18/53/0.5475; seed 15 78/50/0.5125 — again identical to v44; v48 was 0 payoffs / tail 0.50 on seeds 4, 6, 7, 15.
- Guards (seeds 0,6): aut 0.6075/0.6100, dist 0.5700/0.5705, aut-noise4 0.6075/0.6075, dist-noise2 0.5697/0.5703, aut-serial300 0.6100/0.6100, dist-noise2-sticky 0.5698/0.5680, aut-fooddelay2 0.5900/0.5925 — bit-identical to v48 and v44 on every cell.

**Checks.** Scaling: registered C = sec(s-4x3)/sec(s-2x3) at N200/360 steps — v50 2.02919 (same box v48 2.04389; stored D111 v48 row 1.78443) ≤ 2.50. Controller isolation: `tests.test_factor_control` + `tests.test_factor_reward` = 15/15 OK. Predictor guards: `tests.test_factor` 7/7 OK; `parity_3b.py` fails identically (max abs 0.5209722818109068) with and without v50 on a clean stash — pre-existing in this sandbox (numpy 2.4.6), not caused by v50, which does not touch `FactorModel`. No formal gate rows run; no unseen seeds spent.

**Freeze (D-114):** v50 frozen at commit `9f2b374cb0139a49a73e29e225b17a4d16ec8ff9` on `arena/126fc4e1-graph-agent`.
