# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-11 (closed):** Resolve one_window/provisional pos=1 vs dist tail variance on v50
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified (v48):** D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`.
- **3b-2 unqualified (v50):** DEV-16 240/240; `0.908829` vs `v12` `0.891271` (Δ `+.017558`, 99% CI `[−.060548,+.095665]` FAIL vs ≥`−.02`; scaling `1.8800`). `development/results/ar5/gate-dev16-v50/{DEV,validation}.json`.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## Reflection (ar6-alt-11)

- **Fixed `canc` & `canc-noise3`:** `v50` (`GroundedSignedReward(CredibleEventReward)`) selective `changed`-slot replay + empirical-variance `signed_rules` + baseline-anchored `pair_context` lifted `canc` to `0.993` (`+.006` vs `v12`) and `canc-noise3` to `0.983` (`+.069` vs `v12`), plus `aut-fooddelay2` (`0.930`), `aut-fooddelay3-noise1` (`0.882`), and `aut-serial300` (`0.819` vs `v48` `0.786`).
- **Remaining CI blockers (`sd=0.106`):** (1) `one_window=True` traps `pos=1` (`aut` s=0,15, `aut-noise4` s=14, `aut-fooddelay2` s=2) and correlates with `dist*` tail collapses (s=9,13); `one_window=False` fixes `aut` s=0 (`0.6075`) but reintroduces `canc` late shocks (`late0=5..8`). (2) `~parent` splits with `n=[2,1]` permit spurious `splits` on `dist-noise2-sticky`. Next: eliminate step-1/step-2 `pos=1` `provisional` inflation under `one_window=False` or prevent `pos=1` trapping under `one_window=True` without breaking `aut-fooddelay3-noise1 s=1`.
