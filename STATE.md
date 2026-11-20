# STATE — SPEC-3b.3 / Gate 3b-1
Only active Markdown; historical files under `archive/`. Evidence: `development/results/spec333/`; baseline `30b7370` evidence in `results/spec3b/`.
Environment: `pip install --target /tmp/graph-agent-deps -r development/requirements.txt`; one BLAS thread, single-threaded Numba kernels, immutable `harness.bg` snapshots.

## Root causes and decisions
- A: fixed per-observation hazard overwrote regime memory during silence; active cue likelihood was diluted by the marginal mark model. Anchor the regime posterior at last informative marks, propagate elapsed time once, and use undiluted known-regime associations on active input. Quiet/free events still compare rate marks.
- B: sticky dwell classification made false places in noisy stationary streams, precluding established joint process likelihoods. Compare persistent-copy hazard-grid evidence against Jeffreys continuation with sparse 1:J prior odds (reversible MAP role); use committed process grouping where there is no slow view. Do not passively downweight deterministic targets by their marginal rate.
- C: materialized per-particle predictions/counts, repeated fixed-lag messages, covariance copies; active regimes and grouping/search cost grow with stream width. Fuse contiguous-bank mixture/read, sufficient-statistics and sparse Gaussian updates; packed fixed-lag rings preserve scalar HMM updates within 1e−12 (test). The earliest committed merge is at t=700, outside C’s 600-step window; pairwise discovery still has quadratic worst case.
- PR #15 `9455399`: useful scale-role diagnosis; its S1 session z=−5.11 and inhibition z=.78 blocked adoption. Kept conditional residual screening, signed shared center and learned place/regime separation.
- Canonical `factor-L4-P-C`, lags 0–4, own 0–2, 5 slots, K=10, α=β=1, noise .25; prior hazard grid and gap scale 64. Production N=200, S0/S1 N=25. No task names, tables, or production variants.
- Same seed-0 savings workload CPU: scalar ref 8.641 s vs warmed 1.907 s (N200), .939 s (N25): 4.53×/9.20×; different inferential model, not a bitwise old-model claim. Freeze `5316b34416d7` (`484c51f`), 72/72 snapshot tests.

## Screening (Δ, native σ, z; S0 n=1, S1 n=4)
| Row | Δ S0 | Δ S1 | σ | z S1 |
|---|---:|---:|---:|---:|
| blocking | .4367 | .4266 | .0746 | 11.44 |
| cond-inhibition | .9742 | .9720 | .3044 | 6.39 |
| contingency | .4570 | .4448 | .1176 | 7.57 |
| pree | .0402 | .0143 | .0981 | .29 |
| reinstatement | .1618 | .0922 | .0339 | 5.44 |
| renewal-ABA | .0153 | .2551 | .0592 | 8.62 |
| renewal-ABC | .1180 | .1577 | .0093 | 33.92 |
| savings | .2062 | .2432 | .0915 | 5.32 |
| spont-recovery-sessions | .1752 | .1003 | .0684 | 2.93 |
| spontaneous-recovery | .2304 | .2305 | .1909 | 2.42 |

## Frozen verdicts (α≤.05; all row-level Δ, σ, z, n in gate JSON)
- A: **PASS** 9/10, 0 REV, n=16. Inhibition .92813 (SD .08769), reinstatement .06106 (SD .04597), same-place recovery .26188 (SD .07953) pass; PREE .03017 [−.00508,.06541] fails. 272/272 tuples.
- B: **PASS**, (GC, SD, z versus floor, n): v1 (.82342,.01497,3.83,6), d2 (.90365,.01704,1.96,6), d3 (.97653,.00845,7.69,6). All 90/90 tuples; point thresholds, not confidence-bound tests.
- C: **FAIL**, 9.60538 / 2.81927 = **3.40704** (Δ=+.90704 versus 2.50; n=1, σ/z not estimable), one pinned core, 600 transitions each.
- D: **FAIL**, h1 **5/6** (PASS), h2 **1/5** (FAIL; US preexposure −.227, superconditioning −.222 REV). n=16 each, 160/160 tuples per set. One attempt on the frozen tree, no tuning; these holdouts have now been consumed.

## Closure checklist
- [x] Immutable tree/seed provenance, risk comparison, nominated effects and uncertainty; screen overlap with S2 seed IDs is by protocol, not independent confirmation.
- [x] Code frozen before S2/D; no edits or D tuning after launch. `harness.bg`/`staged` unchanged; stage starts/waits in `frozen-gate.log`.
- [x] All 684 expected frozen tuples present exactly once; 0 corrupt/duplicate records. Max gate stage 78.1 s; one wait per stage, estimated work ≤150 s; gate finished at elapsed 1245.8 s.
- Gate **FAIL**: C and h2 miss their bars. PR #16 stays open; #15 untouched. Next: address uncommitted-process scaling and evaluate any future model on newly preregistered holdouts, not by tuning on these D results.
