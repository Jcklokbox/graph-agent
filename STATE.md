# STATE — AR-5, D-40 complete (3b-1 open)
Only active Markdown. Register/reflections `archive/research/runs/AR-5.md`; audit `archive/BRIEF-CE.md`; frozen prior SPEC-3b.3 in `development/results/spec333/`.

## Contract and clock
- Branch `arena/b8e242cb-graph-agent`, PR #16 open, main base `3f932b7`. Merge only if 3b-1 and factor-only 3b-2 clear. Preserve prior journals; h1/h2 exposed, never re-use h2 for confirmation.
- Monotonic T0 **1545.582119**; dispatch ≤**15645.582119**; finish ≤**15945.582119**; final ≥300 s reserved. Two pinned CPUs, immutable snapshots, atomic JSONL, one bounded wait per stage (≤180 s). Cadence next **5145.582119**; milestone commits too. After any compaction read this file + git log first.

## Verified and in flight
- D-39–53: bounded parent/group search and finite locality, no fixed-age process cutover; compare warm/fresh vs merged evidence; context-owned free outcomes and regime cue retrieval, revised role null and hazard survival. Exact ordered live-row gathers, native slot decisions and state-local memo. D-51 first freeze C **FAIL 2.522914**, untouched.
- **D-53 `814b0333575c` / `f41e9c2`**: C **PASS 2.438522** (4.484223/1.838910); A **PASS 10/10, 0 REV**, 272/272, PREE .120169 [.077175,.163164], inhibition .985575, reinstatement .475673. 83 snapshot tests passed + 1 isolated external-cache skip (passes outside). D-53 playback exact vs D-51 on sticky/savings; cost proof bounded degree/state, finite runtime varies with live regimes.
- D-40 FAIL on D-53: 288/288, 3 favorable, 2 REV: background −.12235 [−.16486,−.07984], inhibitor −.56301 [−.62796,−.49805]; valid direct and summation controls. Frozen 812/812, 45 stages, max 82.648 s. `development/results/ar5/gate2/validation.json`. D-40 now exposed, never reopen for acceptance. h1 5/6 and T1 v1/d2/d3 .809077/.921171/.959979 all passed.
- v20 archived (legacy predictive reads); candidate-neutral DEV protocol (`.8912708030`, paired 99% lower ≥−.02), `archive/research/ar5-controller-spec.md`. 3b-2 not executed.

## Next
- Local: D-54 diagnose cue-credit routing, compare place-conditioned predictive baseline and signed component on independent probes. Existing exposed suites and timing remain regression guards.
- Global/dynamic: 3b-2 and merge blocked. D-53 is exposed reference; any new eligibility requires new independently designed preregistered holdout. Next cadence mono 5145.582119.
