# Active checkpoint

**English. If compaction just occurred: `./ar resume`, then [active index](docs/INDEX.md).**
During an authorized run, keep the original clock and 220+20 call budget. Use the ~30-minute cue to reflect in this file.

<!-- managed-next -->
- **Run ar6-alt-N (active):** Diagnose canc/canc-noise3: separate payoff discovery from post-exposure reuse with anonymous matched-exposure controls; then one coherent change inside factor reward+controller.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740, cost 2.080340. `development/results/ar5/gate3/validation.json`.
- **3b-2 unqualified:** v48 D112 DEV-16 240/240; Δ −.003298, 99% CI [−.043362,+.036767], lower bar ≥−.02. `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`. Archival PR #16 merge is not qualification.
- **Next, only when authorized:** isolate absent payoff discovery (D113, `canc` and `canc-noise3`) from poor reuse after real payoff (D97): anonymous matched exposure and adverse-control fixture. [Kickoff](docs/AUTONOMY.md#next-authorized-run-first-decision-only). Replan from measured outcomes; T4 only after 3b-2, then C3–C5.
- [Pipeline recovery audit](docs/research/RECOVERY-AUDIT.md); [historical evidence](archive/README.md).

## AR-6 research note 1 (≈30 min): the `canc` deficit is a discovery failure, not poor reuse

Exposed rows only (dev seeds 0–7, 600 steps). `v48` per-episode sweep
(`development/eval/scripts/ar6_sweep.py`): `canc` scores 0.96/0.94/0.94/0.96/0.88/0.91/**0.00**/**0.00**;
`canc-noise3` 0.93/0.88/0.94/**−0.04**/**0.00**/0.96/**0.00**/**0.00**. The zero
episodes record **0 payoffs in 600 steps** while seeing 57–58 cue-without-lamp
opportunities; every other episode gets 50–70 payoffs with the first one at
step 3–73. So the deficit lives in *reaching the payoff at all*, not in using it.

Matched-exposure controls (`ar6_probe.py`, seed 6, 120-step scripted prefix,
then free): oracle prefix → tail 0.55/0.56 with 0.95 score; always-stay prefix
(22–40 payoffs, null yoke by length) → 0.55/0.53; **matched** prefix (same
position-0 entry steps, payoff decoupled from the cue by a coin flip; only 5
payoffs) → 0.50/0.50; off-goal prefix (cue timing, wrong actuator) → 0.50.
A *real* payoff with the cue present is necessary and, at ≥20 payoffs with the
cue present, close to sufficient. Seed 6 has 58 cue opportunities and no payoff:
it never leaves/enters position 0 on a cue-without-lamp step.

Mechanism under test: the provisional onset optimism
`u[c,f]=m[c,f]+(1−m[c,f])·strength[f]·(1−prevalence[f])/√(1+n[c,f])` is a
**marginal** function of one feature at one arrival control state. Position 0
pays only on the *conjunction* (cue ∧ ¬lamp); every other visit to that state
(quiet, or cue∧lamp) returns 0 and is entered into the same cell `onset[c,f]`,
`onset_reward[c,f]`. So adverse trials at the payoff state drive `m→0` while
`n` grows: the bonus at the payoff state decays *monotonically in experience*,
whereas the middle state (which pays .5 unconditionally) keeps `m=.5`. Trace
evidence (seed 6, t=23): cue onset, `inc[state=pos0]=0.54` vs `0.66/0.64` for
pos2/pos1 and 0.88 for unsupported codes; the cue bonus is already lower at the
state that pays off than at the state that cannot. Seed 0 shows the opposite
ordering at the same step index and pays at t=3.

Alternatives still live: (a) the bonus is zeroed by `u*=~known` when any parent
is admitted at that arrival code; (b) the first-cue choice is an exact tie, so
which seed succeeds is pure tie-break luck and nothing is learned from the
failing tie. Next probe: dump `onset[:,cue]`, `onset_reward[:,cue]`,
`type_count[pos0]`, the admitted parents for pos0, and the resulting `inc`
ordering at several checkpoints for seeds 0 and 6.

## AR-6 research note 2: the binding term is the successor-state novelty, and two fixes were falsified

Two candidate repairs were built on the falsifier-counted denominator of the
onset optimism and both are rejected by exposed data (8 dev seeds per row):

* `v50` (denominator = feature-present arrivals that paid or were denied):
  `canc` 1/8 episodes reach a payoff, `canc-noise3` 2/8. Worse than v48.
* `v51` (Beta rate on the feature's own decided trials, same optimism law):
  same picture, `canc` 1/8.

Why they fail, from the planner trace (`TRACE` hook in
`models/factor_rollout.py`): the first-term value of each action is
`Σ_s P(s|a) [inc(s) + gain]`. `inc(s)` is the arrival state's optimistic
bound `max_f bound(s,f)`. Counting only decided trials makes a *neutral*
arrival state (the middle, where nothing pays and nothing is denied) keep its
untested prior at 0.89 while the payoff state falls to 0.4–0.55 after its own
payoff/denial trials. `canc` seed 6, t=23: `inc = [0.86, 0.63, 0.51, 0.86,
0.76, …]` for arrival codes (pos2, pos0, pos1) — the state that pays is ranked
last, so the agent moves right/stays and never presents the cue at the left
position. Forcing the very first action (0/1/2, 32 episodes each) does not
change the aggregate (`0.536/0.539/0.541` vs free `0.543`), so the lock is not
a single early commitment; it is the persistent ranking above.

Consequence for the model: the unsupported-onset optimism is attached to
*hypothetical successor states*, so what it rewards is arriving somewhere
unfamiliar, not answering the question posed by the cue that is already on
screen. On `canc` the cue's own optimism is identical for all three actions
(it is a property of the current observation), so it contributes no directional
drive at all; the direction is set by position novelty and by which state lost
its `known` veto first. That is the mathematical defect to repair.

## AR-6 research note 3: the gate-indexing defect is real but does not pay

The diagnosis is confirmed by trace and the candidate repair is rejected by
data.

**Confirmed defect.** `models/factor_rollout.py::provisional_values` evaluates
the optimistic onset law at the *successor*: the only cell that predicts the
`canc` payoff is `pos0 ∧ cue` (the cue is a one-step pulse, and the payoff is
collected on the step after the cue appears, when the agent must already be at
the left position). At a cue step the successor never carries the cue, so that
cell never enters `first`; what decides the action is which *successor position*
is least `known`. Since the arrival code of any position acquires a `known`
parent the moment one payoff or shock is seen there, the position that first
pays is the one whose optimism closes first. Trace: `canc` seed 6, t=23,
`first = [0.648, 0.702, 0.648]` with `inc[code4] = 0.708` while
`inc[paying code 0] = 0.861`, but `trans[a0]` puts only `.01` on that code and
`.77` on the middle code.

**Candidate (v54).** `agents/v54.py` + one hook in `provisional_values`:
`cont = max(cont, max_f gate_f · bound(state, f))` with `gate` the onset phase
of the decision observation. One line of law, same statistics.

**Result, ordered sweep, 15 dev rows x seeds 0-3, 600 steps** (`/tmp/rows2.json`):
mean tail v48 0.5587, v54 0.5593 (+0.00061). Row deltas: `canc-noise3`
+0.0137 (`s3` +0.055, 3 → 53 payoffs), `aut-fooddelay3-noise1` +0.0075,
`canc` −0.0144 (`s3` −0.060, 51 → 2 payoffs), everything else within ±0.002.
On 8 seeds with the gate hook confirmed live (vs the `v48` run of the same
instrument): `canc` mean +0.6364 vs +0.6477, `canc-noise3` +0.8580 vs +0.4261.

A **real defect in the same family remains**: the left action's learned
transition lands on the middle code (`0.77`, not the left code) because the
controlled successor was inferred late, so the incumbent optimism of the paying
state is only partially reachable at the decision that matters.

**Two further fixes falsified** (8 seeds each, paired against v48):
`v50` falsifier-counted denominator −0.58 / −0.39; `v51` Beta rate on decided
trials −0.59 / −0.25. Both make the *neutral* middle arrival look more
promising than the arrival that actually pays, because deciding a neutral
arrival by denial is not possible on `canc` (the middle always returns `.5`).

**Guards with the hook in place:** `tests.test_factor_control` +
`tests.test_factor_reward` 15/15 OK (7.3 s); act+learn / plan ratio 1.475
(v48: 1.494), so 1.475 ≤ 2.50. `eval/scripts/parity_3b.py` reports
`max abs 0.521` **both with and without the rollout hook** (stash check), i.e.
that fixture was already failing on this clone and is not caused by this work.

Given the net-wash result the candidate is **not frozen**; `STATE.md`,
`factor_rollout.py` hook and `agents/v5{0,1,2,3,4}.py` are working-tree material.
