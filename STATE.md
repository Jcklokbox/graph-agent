# STATE — AR-5: 3b-1 accepted, 3b-2 active
Live anchor. Register `archive/research/runs/AR-5.md`; audit `archive/BRIEF-CE.md`; prior SPEC `development/results/spec333/`.

## Contract and clock
- Current branch `arena/b8e242cb-graph-agent`, PR #16 OPEN, main base `3f932b7`. Merge only after both milestones pass. Preserve frozen evidence; h2 exposed, not repeated for confirmation.
- T0 monotonic **1545.582119**; dispatch ≤**15645.582119**; hard ≤**15945.582119** with final 300 s reserved. Two pinned CPUs; immutable snapshot/batch, atomic keyed journals, blocking ≤180 s; next reflection **8745.582119**. Read STATE and git log first after compaction.

## 3b-1 verified
- D-51 frozen cost FAIL 2.522914. D-53 frozen all exposed passed, D-40 FAILED (background −.122, inhibitor −.563 REV); preserve `development/results/ar5/gate1,gate2`. D-40 exposed, never use for confirmation again.
- **D-65 `6f619568dc90` / `ac37d50` PASS**: C 2.080340 (4.682722/2.250941); A 10/10 no REV (PREE .124170 [.081228,.167111], inhibition .982190, reinstatement .475673); h1 6/6; v1/d2/d3 .809077/.921421/.959937. D-62 signed .348 [.039,.658], background .128 [.070,.185]. D-61 sole new confirmatory run **3/5 no REV**, signed .308 [.204,.412], competing hazards .096 [.077,.115], context interaction .570 [.525,.615], valid controls. 740/740, 47 stages, max 85.878 s; `development/results/ar5/gate3/validation.json`. Snapshot unit 88 pass + 1 external DEV-cache skip; passes outside. D-63 full-vector normalization resolves tuple/list O(J²), D-64 exact scoped quiet bypass.

## 3b-2 formal qualification active, not accepted
- D-108 v48: signed refinements require an admitted local unary cause; noisy cancellation .875/.938 in exposed seeds1/2. D-110 stopped at 14/15 seed0 to reserve time (all positive), wall 99–118 steps/s. D-111 exact snapshot `621c56d67713`, 15/15 reward/control tests, N200 full act+learn C **1.78443** (3.234842/5.772399 CPU s low/high; 360 each). Frozen predictor remains unchanged, gate3 validator reproduced exactly.
- Freeze `development/results/ar5/gate-dev16-v48/`, exact DEV baseline .8912708030, 240 pairs /177600 steps, 99% lower ≥−.02; no borrowed exposed records. Runner `ar-5-frozen-paired-dev-16-gate-bbfd07d3`, started 03:27:03; 24/240 by 03:29:54, ETA 03:54–03:56. Use `research.ar5_dev_lock` verdict and `archive/research/ar5_dev_audit.py` independent verifier. Do not edit candidate or retune from in-progress results. No merge or next capability except after both gates verified.
- Conservative **03:57 UTC dispatch / 04:02 UTC hard** (2026-11-21 observation clock), reflection 5 recorded ~03:30 UTC; next at formal completion or cutoff. Current branch only, PR #16 OPEN; effects and prior failures archived. If stage coverage incomplete, report not qualified.
