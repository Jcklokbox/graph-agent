# STATE — AR-5, reflection 2 (~60 min)
Live anchor. Register `archive/research/runs/AR-5.md`; audit `archive/BRIEF-CE.md`; prior fixed SPEC in `development/results/spec333/`.

## Contract and clock
- Branch `arena/b8e242cb-graph-agent`, PR #16 open, main base `3f932b7`. Merge only after 3b-1 and factor-only 3b-2 pass. Preserve all frozen journals; no h2 confirmation.
- Monotonic T0 **1545.582119**; dispatch ≤**15645.582119**; hard ≤**15945.582119**, last 300 s reserved. Two pinned CPUs, immutable snaps, atomic JSONL, max blocking 180 s. Next cadence **6945.582119**; major verdicts get their own commit. Read STATE and log first after compaction.

## Evidence
- D-51 frozen C FAILED 2.522914; do not retime for acceptance. D-53 frozen C 2.438522, A 10/10, h1 5/6, v1/d2/d3 .809077/.921171/.959979 PASS. D-40 FAIL 3/5, 2 REV (background −.122, inhibitor −.563), valid controls, 812/812 total, max stage 82.648 s in `development/results/ar5/gate2`. D-40 now exposed; never reopen for acceptance. No 3b-1 closure.
- D-60 support-sensitive signed continuation; D-63 exact once-per-vector conversion fixes tuple/list O(J²) flaw; D-64 exact skip unused quiet emissions; D-65 shared latent tonic dwell grid. D-63/64 two replay arrays exact vs immediate parents. D-65 screen C 2.335, A N200 n4 10/10 provisional, controlled signed +.348 [.039,.658], background +.128 [.070,.185], both anchors valid, 88 tests pass + 1 DEV-cache skip (outside snapshot passes). Rejected branches D-55/57/58/59 archived.
- **D-65 tree `6f619568dc90`**, to freeze `/tmp/ar5-gate3`, all official checks pending. D-61 seeds 83100–83123 and hashed schedules, no candidate output; release requires frozen C/A/h1/v1/d2/d3 and D-62 (=credit) together. D-61 ≥3/5, no REV, signed AND competing hazards, valid anchors. Not a D-40 rerun.
- v20 archived as nonisolated; 3b-2 not executed. `archive/research/ar5-controller-spec.md`, theoretical fixture, exact cache protocol baseline DEV-16 .8912708030, paired 99% lower ≥−.02. Keep no v12 predictor/value reads or hidden fallbacks.

## Next
- Start frozen supervisor on `6f619568dc90`. Progress with bounded milestone waits; no contention on timed C. Preserve complete evidence in `development/results/ar5/gate3` (new gate only, no changes to prior). If 3b-1 passes, implement validated factor-only bounded reward, joint branches and scratch posterior controller then paired DEV; otherwise isolate first failure and continue before reserve. No merge until both actually clear.
