# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-N (active):** Diagnose v48 canc/canc-noise3 failure: separate absent-payoff discovery from post-exposure reuse; freeze at most one coherent M_factor+controller fix.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], bar ≥−.02. `development/results/ar5/gate-dev16-v48/`. PR #16 merge is not qualification.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## AR-6 alt-N — notes

Diagnosis, rejected mechanisms and the frozen fix: [docs/research/AR6-ALT-N.md](docs/research/AR6-ALT-N.md).
Short form: `v48` fails `canc` at *absent-payoff discovery*, not reuse (scripted exposure +.036 over
free, +.061 over a matched yoke; a found payoff is reused 34-41/42 later chances); the optimism is an
onset-frequency statistic, monotone in the state mean. Fix `v60`: Jeffreys room per (state,
payoff-parent configuration); marginal term and onset gate unchanged.
**Frozen:** `v60` (`development/agents/v60.py` + rollout `cell_room` hook), commit `59c4070`
(`1001904` = `.venv` ignore only). Gates: 116/116 tests, 15/15 isolation, |ΔP|=0.0, scaling 2.091.
Exposed seeds 0-7: canc .958/.698, canc-noise3 .601/.457, other 13 cells ±.014 —
`development/arena6/results/summary.json`. No formal gate row, holdout or unseen seed used.
