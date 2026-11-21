# STATE — AR-5: 3b-1 accepted, 3b-2 active
Live anchor. Register `archive/research/runs/AR-5.md`; audit `archive/BRIEF-CE.md`; prior SPEC `development/results/spec333/`.

## Contract and clock
- Current branch `arena/b8e242cb-graph-agent`, PR #16 OPEN, main base `3f932b7`. Merge only after both milestones pass. Preserve frozen evidence; h2 exposed, not repeated for confirmation.
- T0 monotonic **1545.582119**; dispatch ≤**15645.582119**; hard ≤**15945.582119** with final 300 s reserved. Two pinned CPUs; immutable snapshot/batch, atomic keyed journals, blocking ≤180 s; next reflection **8745.582119**. Read STATE and git log first after compaction.

## 3b-1 verified
- D-51 frozen cost FAIL 2.522914. D-53 frozen all exposed passed, D-40 FAILED (background −.122, inhibitor −.563 REV); preserve `development/results/ar5/gate1,gate2`. D-40 exposed, never use for confirmation again.
- **D-65 `6f619568dc90` / `ac37d50` PASS**: C 2.080340 (4.682722/2.250941); A 10/10 no REV (PREE .124170 [.081228,.167111], inhibition .982190, reinstatement .475673); h1 6/6; v1/d2/d3 .809077/.921421/.959937. D-62 signed .348 [.039,.658], background .128 [.070,.185]. D-61 sole new confirmatory run **3/5 no REV**, signed .308 [.204,.412], competing hazards .096 [.077,.115], context interaction .570 [.525,.615], valid controls. 740/740, 47 stages, max 85.878 s; `development/results/ar5/gate3/validation.json`. Snapshot unit 88 pass + 1 external DEV-cache skip; passes outside. D-63 full-vector normalization resolves tuple/list O(J²), D-64 exact scoped quiet bypass.

## 3b-2 active, not accepted
- D-66–71 factor-only rollouts; D-69 aut .860/canc .815/dist −.203. D-72–76 successor/phase/exception screens fail broadly. D-77 shows false control roles erase actual cue/reward evidence; D-79 retraction and bounded replay yield aut 1.000 and dist-delay3 .966 but other exposed slices negative. D-80 one distal success not reused, hard evidence threshold. D-81 provisional reward and finite reachable-state sum first observed aut .349; remaining stage interrupted.
- D-78b scratch forecast matches real readout, coupled toy exact, live bytes and RNG unmodified, legacy modules poisoned; 10/10. Whole-step scaling still needed. No official DEV-16 dispatched: exact cache .8912708030, 240 pairs /177600 steps, 99% lower ≥−.02.
- Arena restored workspace on same branch, but local commits, old monotonic epoch and `/tmp` jobs/journals disappeared. Index recovered to published `ac37d50`, complete frozen gate3 still present. Remaining deadline conservatively **03:57 UTC dispatch / 04:02 UTC hard** (2026-11-21 observation clock); next reflection ~03:30 UTC. Do not restart the 4-hour clock.
