# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-1 (active):** Run matched prefix diagnostic on exposed seeds, then inspect reward-parent and actuator traces
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- **Next, only when authorized:** isolate absent payoff discovery (D113, `canc` and `canc-noise3`) from poor reuse after real payoff (D97): anonymous matched exposure and adverse-control fixture. [Kickoff](docs/AUTONOMY.md#next-authorized-run-first-decision-only). Replan from measured outcomes; T4 only after 3b-2, then C3–C5.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## AR-6 initial decision (replica 1)

Working from `08d61ea` on the fixed Arena session branch; `ar6-alt-1` is the run identifier, not a branch. No external replica or formal gate stream is being read. v48's DEV-16 result is inconclusive/non-inferior failure, not proof of a uniformly inferior mean: the 99% lower bound is -0.043362 versus -0.02. D113 seed 6 lacks realized maximum payoffs; D97 is a distinct v36 run with four early payoffs and poor later reuse, not evidence of reuse failure on the same v48 trajectory.

Prediction: positive exposure may remain unexploited because the controller only selects a signed moderator after a sufficiently supported unary reward parent; no positive reward in the left controlled state means the unary cue likelihood can be admitted for adverse outcomes, potentially closing its novelty hypothesis before the uncancelled contrast is sampled. Competing explanation: even if reward inference is correct, learned transitions or being outside one-step reach prevent the target action. Falsify with ordinary real outcomes in matched scripted prefixes, equal timing for positive versus null, motor excursions to adverse cues, then free behavior and separate one-step reach, transition mass and optimistic/ordinary action values. Script is an experimental intervention only; the agent never sees the fixture's cue labels or intervention flag. Four prefix events, 100 steps, then 140 free steps; seeds 0 and 6, clean and noise3. Adverse events cannot have identical cue polarity at identical times under the same exogenous stream, so compare this arm cautiously rather than claiming perfect motor matching.
