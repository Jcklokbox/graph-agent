# STATE — AR-5: 3b-1 accepted, 3b-2 active
Live anchor. Register `archive/research/runs/AR-5.md`; audit `archive/BRIEF-CE.md`; prior SPEC `development/results/spec333/`.

## Contract and clock
- Current branch `arena/b8e242cb-graph-agent`, PR #16 OPEN, main base `3f932b7`. Merge only after both milestones pass. Preserve frozen evidence; h2 exposed, not repeated for confirmation.
- T0 monotonic **1545.582119**; dispatch ≤**15645.582119**; hard ≤**15945.582119** with final 300 s reserved. Two pinned CPUs; immutable snapshot/batch, atomic keyed journals, blocking ≤180 s; next reflection **8745.582119**. Read STATE and git log first after compaction.

## 3b-1 verified
- D-51 frozen cost FAIL 2.522914. D-53 frozen all exposed passed, D-40 FAILED (background −.122, inhibitor −.563 REV); preserve `development/results/ar5/gate1,gate2`. D-40 exposed, never use for confirmation again.
- **D-65 `6f619568dc90` / `ac37d50` PASS**: C 2.080340 (4.682722/2.250941); A 10/10 no REV (PREE .124170 [.081228,.167111], inhibition .982190, reinstatement .475673); h1 6/6; v1/d2/d3 .809077/.921421/.959937. D-62 signed .348 [.039,.658], background .128 [.070,.185]. D-61 sole new confirmatory run **3/5 no REV**, signed .308 [.204,.412], competing hazards .096 [.077,.115], context interaction .570 [.525,.615], valid controls. 740/740, 47 stages, max 85.878 s; `development/results/ar5/gate3/validation.json`. Snapshot unit 88 pass + 1 external DEV-cache skip; passes outside. D-63 full-vector normalization resolves tuple/list O(J²), D-64 exact scoped quiet bypass.

## 3b-2 active, not accepted
- D-66–80 isolated reward/control failures are archived in AR-5. D-78b live-state byte equality, passive parity, coupled exact two-step value and legacy-import poison. D-88 12/12 incl. exact sparse projection and real-observation atom conditioning.
- D-81–89: typed finite reward, retractable dwell-based actuation, elapsed onsets, one-row optimism. D-86 `v34` exposes strong stationary scores; serial −.123. D-90 `v36` recalled control modes: serial seed0 .7985, other main scores preserved, seed1 exposure in progress. D-91 v34 N200 full-step C1.872/2.008 process/dense; J18 dense verified diagnostic. D-92/93 exact support pruning/tie/label-permutation candidates not yet exposed. Need whole-step cost and throughput of actual freeze, seed-diverse diagnostics.
- No official DEV-16 dispatched; fixed .8912708030, 240 pairs /177600 steps, conservative 99% lower ≥−.02. Freeze supervisor `research/ar5_dev_lock.py` validates exact cache/stages. Latest dispatch for completion depends on measured wall throughput.
- Arena restoration lost old epochs/tmp/local commits; published `f10421e` restored predictor gate3 and working prototype history. Conservative origin deadline **03:57 UTC dispatch / 04:02 UTC hard** (2026-11-21 observation clock); next reflection ~03:30 UTC. PR #16 remains OPEN, no merge without both gates.
