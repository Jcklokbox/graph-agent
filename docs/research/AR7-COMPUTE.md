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

## Executing AR-7 (authorized separately)

The replacement gate is independent of the spent AR-6 DEV-20 draw. Its 20
hash-derived seed blocks (same 15 DEV variants) have disjoint seed IDs; no fresh
outcome has been produced during protocol preparation. Run the actual v12 and
one preselected candidate, pairing on the same environment/brain seeds and raw
normalizers. The qualification statistic remains the mean of 20 complete
within-block mean differences, with two-sided 99% Student interval and lower
bound at least -0.02. There is no intermediate score or retry.

The provisional 8-row packing is superseded by 30 full ten-row stages per
model. Greedy stage balancing, and largest-first queue order within a stage,
use only per-cell wall times from the spent AR-6 stage receipts. Run every distinct full production stage shape (six per model) on exposed
DEV-16 seeds on the final immutable tree. Ranking only by summed duration
would use three near-duplicates and miss the v12 two-serial-job stages. Register the per-model watchdog at no less than
1.25 times its largest measured stage. Fund all 60 complete watchdogs, four
seconds per stage of termination and ledger allowance, 30 seconds of launch
allowance, and the separate 600-second audit reserve. The wall estimate is
not based on cached payoff timings. Never reduce a cap to fit the clock.

D65 exactness is checked by actual fixed-tape playback of `ac37d503f3d0`
(reconstructed tree `6f619568dc90`), against the active candidate on 320 sticky
and 320 noisy cancellation transitions each. Byte equality and the forcing
stream are checked separately. This is not a new 740-tuple D65 gate. The D111
registered 15-test suite is the actual nine named controller and six named
reward cases in `gate-dev16-v48/fixtures.log`. Retain all the identities and
add the selected model to both actual-agent isolation cases (no-legacy under
poison imports, and nonmutating deterministic planning). Candidate full
act+learn process scaling is the registered warm N200 width pair; do not
substitute an isolated update ratio for it.

Observed in AR7-01 (8 episodes per row, exposed seeds 6/13, same four
variants and warm cache): v59 took 124.064 s at one worker and 59.273 s at
two (2.093×); v12 took 258.332 s and 136.158 s (1.897×). Non-timing outputs
match across worker counts for all 16 pairs. Guest topology reports CPUs
0 and 1 as siblings, but it did not predict measured throughput, and the
v59 result above 2× is not a claim about physical parallelism (one pack per
setting, run order and speed drift not controlled by replication). Use two
workers, one numerical thread apiece, no other sustained jobs during timed
stages. See `development/results/ar7/worker-cost.tsv` and raw receipts.
