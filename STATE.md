# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-N (active):** Diagnose v48 canc/canc-noise3 failure: separate absent-payoff discovery from post-exposure reuse; freeze at most one coherent M_factor+controller fix.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- **Next, only when authorized:** isolate absent payoff discovery (D113, `canc` and `canc-noise3`) from poor reuse after real payoff (D97): anonymous matched exposure and adverse-control fixture. [Kickoff](docs/AUTONOMY.md#next-authorized-run-first-decision-only). Replan from measured outcomes; T4 only after 3b-2, then C3–C5.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## AR-6 alt-N — notes

Diagnosis, rejected mechanisms and the frozen fix: [docs/research/AR6-ALT-N.md](docs/research/AR6-ALT-N.md).
Short form: `v48` fails `canc` by *absent-payoff discovery*, not reuse (scripted exposure lifts canc
+.036 over free and +.061 over a matched yoke; one payoff is then reused at 34-41/32-42 later
opportunities); the provisional optimism is an onset-frequency statistic over control states, so its
headroom is monotone in the state mean. Fix `v60`: per-(state, payoff-parent-configuration) Jeffreys
room of the realized payoff, marginal term and onset gate unchanged.
**Frozen 3b-2 candidate:** `v60` = `development/agents/v60.py` + rollout `cell_room` hook,
commit `59c4070` (`1001904` adds only a `.venv` ignore). Gates: 116/116 tests, 15/15 isolation,
predictor parity |ΔP|=0.0, scaling 2.091 ≤ 2.50. Exposed DEV seeds 0-7: canc .958 (v48 .698),
canc-noise3 .601 (.457), other 13 cells within ±.014 — `development/arena6/results/summary.json`.
No formal gate row, held-out stream or unseen seed was used.
