# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-11 (active):** Run DEV-16 qualification for v50 and validate
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## Reflection (ar6-alt-11)

- **v48 diagnosis:** `ElapsedEventReward` inflates `pos=1` (`base=0.5`) above `pos=0` after 1–3 shocks (`canc s=6,7`); `context_requires_parent=True` (`L-1` violation) + `P=1` preemption (`canc-noise3 s=3,4`) kill configural `signed_rules` and `pair_context` under early adverse cues.
- **v47 diagnosis:** Full `_reset_context_counts()` on unrelated anchor churn + `0.25*log1p(N)` variance-blind BIC cliff + ungrounded `0.5` `pair_context` prior induce `late0=4..11` shocks on `canc-noise3`.
- **v50 (`GroundedSignedReward(CredibleEventReward)`):** Selective `changed`-slot replay, empirical-variance `signed_rules` (`var=(0.25+ss_w)/(1+N)`, `cost=log1p(N)+(~parent)*log1p(F-1)`), baseline-anchored empirical-Bernstein `pair_context`. Exposed DEV (`s=0..7`): 118/120 near-oracle (`canc` 8/8 `late0=0`; `canc-noise3` 8/8; `aut-fooddelay3-noise1` 8/8; `aut-serial300` 8/8; `dist*` 48/48). Next: unit tests, N200 scaling, DEV-16 dispatch.
