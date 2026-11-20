# STATE — SPEC-3b
Active SSOT; historical Markdown under `archive/` is excluded from the active tree. No merge.
Env: NumPy 2.4.6 at `/tmp/graph-agent-deps`; from `development/`, `PYTHONPATH=/tmp/graph-agent-deps:. OPENBLAS_NUM_THREADS=1`.
Evidence/manifests: `development/results/spec3b/`; compute via `harness.bg start [--tree H]` + one `wait` per invocation. Locked harness unchanged.

## [0] Hot path and localization
- Kernel checkpoint `806c933`, s0, N200: ε=0 (260 transitions, stream seed 42); savings reference 9.115 s vs 4.56 s (N200) / 1.58 s (N25): 2.00× / 5.77×. Batch fixed-lag messages by (K,width,lag); reuse forward priors, emissions and windows; vectorize cells and member evidence.
- Base `98008dfd0fd6`, s0: `q(place,z) ∝ p(place)p(z|place)∏_j Bern(y_j; Σ_M w_zjM p_z/place,jM(x,lag))`; fast evidence changes place. Reinstatement A .50483−.54008=−.03525 vs lc-sub-add +.09254; after free events and home, new regime 3 mass .987 (control .878).
- Sessions: `p(z|place)` had no elapsed absence term; extinction mass .964 long vs .849 short, effect −.02116 vs lc-sub-add +.03817. Inhibition: `μ_new=0, Σ_new=s₀I`; BX/BY .0791/.3978 vs .0172/.9876; binned conditional admission and marginal eviction were incomparable.
- d3-and s0: decompw has groups (0,1)(2,3)(4,5)(6) at t=1000 and updates a joint-emission likelihood per family; old Factor used `∏_j Bern(Σ_M w_zjM p_zjM)` and reweighted process marginals again per regime. `stage0-trace.jsonl`; `eval/scripts/localize_3b.py` with base snapshot.

## [1] Canonical model / decision
- `factor-L4-P-C`: lags 0…4, own lags 0…2, 5 slots, K=10, α=β=1, noise=.25, s₀=1, hazard-grid floor .003; only N differs in dev, 25→200. No rejected production variants or v12 tables in FactorModel.
- Hard slow view (empirical active dwell ≥12); gaps relax saved posterior by `retain=(1−h)exp(−Δ/64)`. Boundary transitions do not train the old place. Shared signed regression center plus regime deviations; parent score is squared centered BMA residual covariance over input variance plus ridge, one slot per source.
- Parent-missing mixture .12 (.40 in fresh regimes) and normalized quiet-event marks: silence supplies timing, not a false regime shift. Process evidence shared across regimes (switch .002), committed groups keep the process likelihood when no slow scope is found. d3 S1 GC .94930±.03857 SD, n=4 (CI [.88838,1.01022]); .00070 below .95; not closed in screening.
- T2 S0/S1 n=1/4, native σ fixed, screen −3.7328. Same tree `cf38b91bb916`, 68/68 keys, dup/bad=0/0. S1 passed all requested z floors; 71 snapshot tests passed.

| row | Δ S0 | Δ S1 | σ native | z S1 |
|---|---:|---:|---:|---:|
| blocking | .4249 | .4074 | .0746 | 10.92 |
| cond-inhibition | .8548 | .4326 | .3044 | 2.84 |
| contingency | .4137 | .4052 | .1176 | 6.89 |
| pree | .0136 | −.0045 | .0981 | −.09 |
| reinstatement | .1352 | .0550 | .0339 | 3.25 |
| renewal-ABA | .5072 | .5291 | .0592 | 17.88 |
| renewal-ABC | .3009 | .3106 | .0093 | 66.80 |
| savings | .3204 | .0774 | .0915 | 1.69 |
| spont-recovery-sessions | .1706 | .1838 | .0684 | 5.37 |
| spontaneous-recovery | .0447 | −.0062 | .1909 | −.06 |

## [2] Frozen Gate 3b-1 — cf38b91bb916
- A: **FAIL**, 8/10 PASS, 0 REV; reinstatement .0655 (SD .0404) and inhibition .4640 (SD .3944) PASS, sessions .1817 (SD .0190) PASS; pree .0219 [−.0066,.0504] and spontaneous recovery .0056 [−.0128,.0239] FAIL. n=16, 272/272 tuples, 0 dup/bad; nominal α=.05, no correction.
- B, **INCOMPLETE**: T1 v1 GC .75294 (SD .05025, n=2/6 complete seeds; 20/48 tuples), d2 no complete seed (3/24 tuples: .96939/.78684/.94369), d3 .93316 (SD .04779, n=3/6; 10/18 tuples). Below-target provisional v1/d3 estimates are not final verdicts.
- D, **NOT RUN**: h1/h2 retained their one-shot status (conservative estimate 1017.6 s vs 149.1 s left). No code edits after freeze.
- C, seed 0 × 600 transitions, N200: 15.526/4.927 = 3.151 (limit 2.5) **FAIL**; no CI at n=1. Strict linear scaling not established (pair-screening cost is quadratic).

## [3] Checkpoint
- S0/S1 used only dev seeds; screen-selected and S2 seed IDs overlap by protocol, but N differs; no holdout selection, post-freeze edit, or cross-tree tuple reuse. Screen significance uses the prescribed σ, not a post-selection confirmation. Other rows were screened together; uncertainty carried to final verdicts.
- Open: A lost PREE and same-place recovery; C exceeded cost cap; B/D incomplete. Next: review this frozen result without merge; fix scaling and same-place interval inference in a new checkpoint, not on D.
- Runtime: S2 ≈ 789 s of 800 s; max staged invocation 161 s (d3 S1), one wait each, no runner polling. Stage journals: 0 duplicate/bad tuples.
