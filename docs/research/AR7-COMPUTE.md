# Turn 2 compute brief (preparation only)

W1 and W2 were left blank. No new episodes, holdout IDs, cache, or candidate freeze were opened in preparation. The session branch preserves all AR-6 artifacts; DEV-20 is spent. This is an operational brief, not a research plan.

| Timed evidence | Observed wall cost | Throughput |
| --- | --- | --- |
| v12, 29 successful 8-episode stages, 2 workers | 111.40 s mean, 137.35 s max | 0.0718 episodes/s |
| v58, 28 successful 8-episode stages, 2 workers | 68.44 s mean, 101.81 s max | 0.1169 episodes/s |
| v59, 23 exposed episodes pinned to 1 logical CPU | 323.54 s summed episode wall, 14.07 s mean, 33.42 s max | 0.0711 episodes/s; no measured 2-worker stage |

v59 exposed counts: 7 cancellation, 7 noisy cancellation, 3 serial, 3 late food, 3 distal. Timing provenance: `development/results/ar6/dev20-v58/stages.jsonl` and `18-{v59,guards-v59}.jsonl`. Linux exposes 2 allowed logical CPUs, both listed as siblings of the same core; the current cgroup has no CPU quota. Host physical allocation is not independently visible. Use the observed 2-worker throughput, not a 2-core assumption.

Turn 2 starts with warmed 1-vs-2-worker, identical 8-row timing packs on already-exposed seeds, including a serial-heavy pack. Pin CPUs and single-thread numerical libraries; measure stage wall time including startup/journaling, and retime the eventual exact candidate. Balance long tasks between workers inside each registered stage; prohibit contention during qualification. Do not shorten the final stage to fit a deadline.

A new candidate-independent ref would cover 20 fresh seed blocks at 15 variants each, 300 actual v12 episodes. Pin executable/source, environment, dependency versions, seed program, raw denominators and journal SHA-256. Keep reference outcomes unopened during model selection; publish the checksum and retain immutable receipts. No DEV-20 entries or exposed exercises can populate it. Precomputing moves work off the gate path, not outside the wall budget. No new reference was built; its planning cost is based on measured v12 stages.

Provisional W2 allocation, before retiming: 5,400 s reference, 450 s exposed check, 3,350 s candidate-only gate, 600 s audit; development plus timing preflight gets W2 minus 9,800 s if feasible. Full 300-row stages count is 38 with batches of up to 8. Reference/candidate total budgets use 38 mean stages plus 25% and startup rounded up; provisional per-stage caps 175/130 s use measured maxima plus 25%, rounded up. The 450 s exposed allowance uses 323.54 s plus 25% and at least 45 s overhead. These are planning numbers, not a registered lock: v58 is only a proxy until the finalist is timed. Refuse dispatch if W2 cannot fund the revised complete run and reserve. Keep the acceptance thresholds, cohort size, isolation, predictor parity and scale criteria unchanged. No new work until explicit execution authorization and the wall limits are supplied.
