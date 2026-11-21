# Autonomous run history (UTC, historical)

| Run | Durable result and operating lesson |
| --- | --- |
| AR‑1, 2026‑11‑19/20 (~2.5 h, stopped under then-rule) | Calibrated context-tree + VOI v12 .891 vs v11 .770, paired +.121 [.093,.150]. A′ .81 GC; learned HMM/window decomposition d1→d2; hard labels/birth scoring and sticky fast merges do not give calibrated inference. Early context patches v13/v14 traded rows; v14 silence could favor extinction in a recovery gap. |
| AR‑2, 04:20–09:20 | v15 .83 vs .66 v1; no new DEV claim. v12 profile: all pairs 38%, `_discount_all` 18%, timing 10%. F4 changes architecture after three unproductive cycles. D13/14 conditional emissions; D16–19 anonymous causes with bounded parent averaging and additive credit. D20 unfinished at 08:50 under contention; isolated >=20% headroom. D21 v18 fixed candidates; 56/56 tests (68 s). Witness ≠ agent. |
| AR‑3, started 09:35:52; stopped 12:27 vs 13:35 plan | Accepted slice 2 v18, v1 .76, +.103 vs v12; v1 T2 10/10/h1 4/6; DEV historical equivalence, ×3.8 cost. h2 fixed after control-only correction; F11→F12 separates place and contingency, not completed 3b. Early stop violated contract; shortlist new short decisions rather than end after planned bullets. See [ledger](../../archive/research/runs/AR-3.md). |
| AR‑4, 13:08:06–16:53:26 report (planned 17:08) | D32 h2 falsification; D33 delay attribution; D34–38 view/lag interference. ~3h35 working, 340/450 CPU-min (~75%), of which ~320 long trials and ~15–20 probes. Observed 14:12–16:54: ~90–95 min waiting, 8 min forced idle, ~40 min probes, ~15–20 min decisions, ~3 min tests. Pre-14:12 action timings unknown; 16:41–16:43 interruption mechanism unknown. See [ledger](../../archive/research/runs/AR-4.md). |
| AR‑5, D39–D113 | D40 spent after signed/background reversal; D51 C 2.522914 fails; D53 exposed guard pass not fresh transfer; D54–64 bounded credit/clock and D61 preregistered independently controlled split. D65 accepts 3b‑1: C 2.080340, A 10/10, h1 6/6, B .809077/.921421/.959937; 740/740 in 47 stages, new D61 3/5 no REV and D62 guard passes. D66–109 controller mechanism screens, no teacher/legacy fallback. v48 D111 15/15 fixtures and act+learn ratio 1.78443; D112 complete DEV‑16 fails lower 99% CI [−.043362,+.036767], Δ−.003298. D113 missed payoff exposure in selected trajectories; D97 observed early payoffs with poor reuse in a different run. See [3b-1 evidence](../../development/results/ar5/gate3/validation.json), [DEV](../../development/results/ar5/gate-dev16-v48/DEV.json) and source excerpts (`excerpt D-112`). |
| AR‑6, 10,800 s including final reserve | Matched payoff exposure restored hard reuse. Soft real-atom forecasts and nonadditive information improved exposed behavior. Exact parity vetoed v57; v58/v59 preserve canonical sensory law; exact shared projection cuts active CPU 28–34% while registered work passes. Fresh v58 DEV-20 **incomplete** (456/600 verified, no CI); dispatch underbudgeting, not statistical failure. Same-support joint evidence counterfactual repairs the lagged outcome alias; no deployed joint controller or T4. [Register and handoff](AR6.md). |

## AR‑2 T2 v1 (16 seeds, signed contrasts)

| Row | v12 | lc-f-E | lc-anon | lc-sub | lc-sub-add |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blocking | .578 | 0 | 0 | .473 | .473 |
| CI | .429 | 0 | 0 | 0 | .970 |
| Contingency | .428 | .434 | .418 | .460 | .460 |
| ABA | .972 | .942 | .963 | .877 | .862 |
| ABC | .653 | .484 | .482 | .439 | .432 |
| SR | .001 | .404 | .201 | .197 | .203 |
| SR-sessions | −.002 | .185 | .126 | .145 | .150 |
| Savings (direction-only) | −.595 REV | .048 | .053 | .118 | .120 |
| Reinstatement | −.006 | .339 | .298 | .141 | .110 |
| PREE | −.218 REV | .207 | .273 | .307 | .303 |

## Reported bottlenecks and evidence limits

- AR4 P1–7: all-at-end trial; both CPUs occupied (even 1+1 became two trials); code mutated between chain stages (D36); untracked probes had BX/BY overwrite, wrong AAB key, missing `--set`, uncomputed q; hand-paired files; unscaled noisy metrics; artificial final-window idling. Agent errors M1–5: early judge, no probe CPU reserve, passive polls, unmeasured extrapolation (d2 157 vs 40 s; D37 37 vs 30 min), narrative before trace. D32 249 s was needed as one-shot, D33 measured instrumented contribution; D36 T2 59 CPU-min vs 20 s seed-0 trace; D37 37 vs short probes; D36 d2/d3 need pairing (≈64 CPU-min). Interruption/latency and early internal timings were not recoverable from mtime.
- Improvements in order proposed: seed-major prefix and keyed resumable journal with unchanged arithmetic; nice priorities while keeping cost isolated; development smoke before powered judge; frozen code and manifest; integrated traces; result registry matched on seed/steps; raw nats/noise scales; queue offline design during compute; no forced sleep. For the current contract see [autonomy](../AUTONOMY.md), not historical stop, language or commit rules.
- AR5 reported tool cliff 276 calls, not an introspectable quota. Diagnose discoveries by local trace and keep compaction recovery separate from experiment evidence.

Historical details: AR3 exact speedup ×3.1 enabled a 49-min DEV screen versus a 5-h planning estimate (not a completed 5-h run). Before h2 the F12 ≥3/5 and non-worse-than-parent forecast had ~55% confidence; D32 falsified it. F13 conditional P(regime | place) was proposed to recover ABA while preserving AAB/sessions (~50% confidence), not a result. v19 canceled at 13:07; D33 tested whether predictive tables, not just the controller interface, remained necessary. AR4 meta-audit counts ~7 CPU-h by rough reconstruction (distinct from the detailed ~340 CPU-min accounting); ~4 CPU-h on D36/37/38 gave two decisive verdicts, while ~15 CPU-min of probes produced r4/r5, sessions/inhibitor explanations and `-C/-X` candidates. A two-order-of-magnitude efficiency estimate from those different purpose workloads is descriptive, not a controlled causal comparison.
