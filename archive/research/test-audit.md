# Test audit (2026-11-17, after slice 2)

Review of every evaluation instrument (T1 predictive worlds, T2 phenomena, T3 DEV, T0 unit
tests) for **logic**, **bugs**, **feasibility** (can an ideal learner of the stated generative
model pass it at all?) and **discriminativeness** (does passing it say anything about the
hypothesis, or does a trivial learner pass too?). No test was changed by this audit; the
proposals at the end need agreement. Numbers come from cached results
(`results/t2_v13.json`, `t2_v11.json`, `dev16_v11.json`, `scr_v13.json`, `scr_v12e_b.json`).

Verdict labels: **OK** · **BUG** (the stream or readout does not implement the stated
procedure) · **TRIVIAL** (passes for reasons unrelated to the hypothesis) · **FRAGILE**
(margin small compared with ceiling or noise) · **FEASIBILITY UNKNOWN** (nobody has checked
what an ideal learner does on this exact stream).

---

## T2 — conditioning phenomena (`envs/phenomena.py`)

Mean probe values, 16 seeds:

| row | RW | v11 | v12 | v13 |
|---|---|---|---|---|
| blocking: control B / blocked B | .50 / .00 | .72 / .02 | .62 / .04 | .59 / .04 |
| inhibition: BY / BX | 1.00 / .10 | 1.00 / .85 | .71 / .28 | .71 / .28 |
| contingency: contingent A, ctx / degraded A, ctx | .48, .00 / .14, .01 | .50, .00 / .50, .32 | .50, .00 / .50, .33 | same as v12 |
| renewal ABA: ABA / ABB | .00 / .00 | 1.00 / .00 | .98 / .01 | .98 / .01 |
| renewal ABC: ABC / ABB | – | – | .66 / .01 | .66 / .01 |
| spont. recovery: immediate / delayed | .00 / .00 | .11 / .11 | .13 / .13 | .05 / .05 |
| spont. recovery home: immediate / delayed | – | – | .13 / .14 | .05 / .11 |
| savings: acq2 / reacq2 | .36 / .36 | .94 / .18 | .79 / .19 | .79 / .18 |

### 1. Blocking — OK
Kamin's design with a control group (X+ then AB+) that guarantees B is learnable. It
discriminates "learns nothing about compounds" from blocking. RW passes, so it is a guard,
not evidence for the inference hypothesis. Near-complete blocking (.04) is stronger than in
animals but not a defect.

### 2. Conditioned inhibition — OK
Summation test with a novel-cue control (BY), which removes generalisation decrement. RW
passes, so this row is a guard as well. Rescorla's two-test strategy also asks for a
retardation test. That is optional.

### 3. Contingency degradation — BUG (minor) + TRIVIAL
* **Bug.** The 15-step context-only gap before the probe contains no free USs. In the
  degraded group that gap is a short extinction of the context right before the ctx probe:
  RW's ctx probe falls to .01 (true rate .5) and v12's to .33. The measured effect therefore
  mixes the contingency effect with gap extinction. The verdict is unlikely to flip, but the
  magnitude is not what the row claims to measure. Fix: the gap keeps the group's `p_free`.
* **Trivial.** The readout P(US|A) − P(US|ctx) (a suppression-ratio analogue) passes for any
  agent that tracks the context's US rate. v12/v13 still predict P(US|A) = .5 in the degraded
  group, which is correct as a prediction. This row is a guard only.

### 4. Renewal ABA — TRIVIAL
A configural learner treats (A, ctx1) and (A, ctx2) as unrelated situations and gives
complete renewal (v11 1.00, v12 .98). In animals renewal is partial, and the diagnostic fact
is the asymmetry: acquisition transfers across contexts, extinction does not (Bouton). The row
cannot distinguish latent-cause inference from "context is just another cue". It is a valid
guard but no evidence for slice 2.

### 5. Renewal ABC — BUG (translation)
"Novel context" is implemented as **no context channel at all**. That tests A alone (a
reduced configuration), not A in a novel context; an animal is always in some context. Fix:
a separate never-seen channel `ctx3`. v12 already passes through generalisation (.66). The
literature's ordering ABA > ABC is not checked.

### 6. Spontaneous recovery, v0 (interval in the conditioning context) — FEASIBLE in principle
I retract my slice-2 claim that "the normative model predicts no recovery here". If both
contexts predict the context-only steps equally (shared ITI statistics), the interval carries
no evidence and the belief drifts toward the stationary distribution, as in COIN's error-clamp
recovery. The size depends on the hazard:
* with h = .001 per step, about 25 % of the belief mixes in 285 steps, so recovery is ≈ .1;
* with a hazard learned from the stream (one change in ≈ 390 steps), mixing is ≈ 50 %, so
  recovery is ≈ .25.

These are rough estimates, not checked.

v13 fails this row because of its own mechanism: the current context has learned the
interval cells and the archived one has not.

Logic problem: the agent experiences time only as observation steps, so a long in-context
interval is also extensive context exposure. In animals the interval is spent in the home
cage, and the interval (days) is much longer than training (sessions). Here 285 steps is
shorter than training (≈ 390 steps).

### 7. Spontaneous recovery, home — BUG (translation) + contaminated
* "Home" is again **no context channel**, the same representation as ABC's "novel context".
  But the home cage is the most familiar context, not a novel one.
* The mechanism was tuned on this row after the prediction was written.

Effect sizes: v13 +.054 (6 % of the acquisition–extinction range), v12 +.017.

### 8. Savings — FRAGILE, FEASIBILITY UNKNOWN
* Baseline at ceiling: a fast learner reaches .79–.94 after two deterministic trials
  (Beta(½,½) gives .83), so savings must beat a near-ceiling naive acquisition after two
  trials. The margin is ≤ .2 for any agent.
* I retract "structurally unreachable". The ≤ 1.9 nats per trial came from v13's soft
  extinction context (v12's extinction floor is .13 after 30 unreinforced trials). For an
  ideal observer, the acquisition context predicts US ≈ .97 and a crisp extinction context
  ≈ .02, so one trial gives ≈ 4 nats and two trials exceed logit h. Retrieval is therefore
  plausible, and the effect would be ≈ +.05 to +.1. Unverified.
* Biology: rapid reacquisition is common but not universal; reacquisition can be slow after
  extensive extinction or when reinforced trials are scattered through extinction (Bouton and
  colleagues). A two-trial point readout is the weakest possible measurement of a learning
  rate.

### Cross-cutting T2 issues
1. **No feasibility witness.** Rows were translated from the literature without checking what
   an ideal observer of the claimed generative model (latent causes / contexts with a sticky
   prior) does on the exact stream. A row is useful only if (a) the witness passes it and
   (b) a trivial learner (RW, configural counts) does not. Today:
   * rows 1–5 fail (b) or are not checked against it;
   * rows 6–8 have never been checked against (a).
2. **Degenerate statistics.** The only randomness is the ITI jitter and trial order, so the
   95 % CIs are razor-thin ([+.575, +.580]). The verdict is effectively "mean ≥ .02", and .02
   is far below any biologically meaningful magnitude. Effects should be relative (fraction
   of the acquisition–extinction range) with a stated minimum, and reported next to the
   witness's value.
3. **Time scale.** One trial is ≈ 6.5 steps. Retention intervals are shorter than training.
   There is no session structure (sessions separated by home-cage intervals), which matters
   exactly for spontaneous recovery and savings.
4. **Gap before every probe** (15 context-only steps): harmless except in contingency (row 3).
5. **Missing diagnostic phenomena for contextual inference**, each well replicated and
   feasible for the witness:
   * reinstatement: unsignalled USs after extinction restore responding, and the effect is
     context-dependent;
   * the partial-reinforcement extinction effect: extinction is slower after partial
     reinforcement because the change is harder to detect.

## T1 — prediction vs the exact filter (`envs/family.py`, `eval/predictive.py`)

* **OK in logic.** Exact filter, random policy, second-half log loss, baselines markov1,
  markov4 and marginal.
* **GC = 1 is unreachable, and the reachable ceiling differs per world.** The filter knows
  the true parameters; a learner pays a parameter- and structure-learning cost that depends
  on the world. The mean GC averages unequal ceilings. A known-structure learner (an online
  HMM with the true K; v11_bma ≈ .81) would give "fraction of the *learnable* gap".
* **`count3`.** Every agent tested so far is at markov1's level (.41 vs .42). It is either a
  stretch goal or out of reach for this agent class. As it stands it adds a constant to every
  mean. Flag it or report it separately.
* **`two-hmm` is not a decomposition test.** The joint space has only 4 states, so a joint
  learner pays nothing. Slice 3 needs worlds with ≥ 3–4 processes and joint spaces of
  ≥ 50 states, where a joint learner measurably fails within the episode.
* **Early calibration.** "≤ markov1" is a strict comparison of 6-seed means with no
  tolerance, so 7/8 vs 8/8 can be noise. Use paired intervals.
* Action dependence is tested only in v0 (`control`). Acceptable for now.

## T3 — DEV behavioural benchmark (`envs/tasks.py`, `eval/benchmark.py`, `eval/screen.py`)

v11, 16 seeds:

| variant | mean | sd | oracle − stay per step | reading |
|---|---|---|---|---|
| canc, dist, dist-delay3 | .98–.99 | ≤ .06 | .05–.07 | saturated: regression guard only |
| aut / aut-half | .94 / .94 | .06 | .11 / .055 | aut-half is an **exact duplicate** for a scale-invariant agent; double weight |
| aut-noise4, aut-noise4-sticky, dist-noise2-sticky | .87–.89 | .07–.10 | | moderate |
| canc-noise3, dist-delay3-noise2, dist-noise2 | .59–.82 | .12–.21 | | informative |
| aut-serial300 | .74 | .16 | .11 | informative; biologically meaningful (serial-reversal learning set) |
| **dist-noise3-p4** | .32 | **.53** | .07 | **bimodal**: seeds at −.57…−.02 and .48….89 |
| **aut-fooddelay2** | .62 | .34 | .09 | **bimodal**: 4 seeds ≈ 0, 12 seeds .6–.94 |
| **aut-fooddelay3-noise1** | .34 | .32 | .08 | wide, partly bimodal |

* **Bimodality.** In three variants the outcome is mainly "did exploration discover the
  contingency before the last 200 steps". Negative scores are perseveration (worse than
  standing still). These three dominate the variance of the DEV mean. v12's "BETTER" was
  carried by one of them (dist-noise3-p4 +.62).
* **Small denominators.** oracle − stay is only .05–.11 reward per step, which amplifies
  noise in the normalised score.
* **Five of 15 variants are saturated or duplicated.**
* **Reference agent.** Decisions compare with the cached **v11**. Since v12 is the accepted
  agent, slice decisions need a v12 per-seed cache; today paired v13 − v12 exists only on
  the 3–4 screen seeds per variant (+.038 mean, noisy).
* **Small inconsistency.** The BlindGate docstring says food is "seen only there", but the
  food channel is on regardless of the agent's position. Intent should be confirmed; it
  changes how much is learnable without exploring.

## T0 — unit tests

44 pass. `tests/test_v13.py` checks only two properties (no birth on stationary data;
birth, normalised belief and retrieval). Adequate for a prototype.

---

## Proposals (need agreement; nothing changed yet)

| # | change | cost | effect |
|---|---|---|---|
| 1 | **Feasibility witness for T2**: an ideal latent-cause observer (sticky CRP over contexts, Beta-Bernoulli per configuration, particle filter), checked by exact enumeration on tiny streams; run on every row | ≈ 1 h of code, minutes of compute | every row gets "ideal value" and "trivial value"; impossible and trivial rows become visible |
| 2 | Fix definite bugs as T2 v1 (v0 kept for history): contingency gap keeps `p_free`; ABC uses a never-seen `ctx3`; home is a familiar `home` channel (present between sessions) | small | translations implement the stated procedures |
| 3 | Redesign spontaneous recovery and savings **after** the witness (session structure; reacquisition curve over several trials instead of a 2-trial point) | small | decided by feasibility, not by my agent |
| 4 | Criterion: relative effect (fraction of the range) with a pre-stated minimum, reported next to the witness; CI kept only as a formality | trivial | verdicts mean something |
| 5 | Optional additions: reinstatement, partial-reinforcement extinction effect | small | discriminative rows for contextual inference |
| 6 | DEV re-analysis from existing caches, no new runs: solved fraction plus score given solved for the bimodal variants; DEV mean without the duplicate aut-half; saturated variants as a separate regression check | 0 compute | separates exploration luck from model quality |
| 7 | T1: known-structure learner reference; slice-3 decomposition worlds (≥ 3–4 processes) | ≈ 2 min compute; design at slice 3 | GC relative to what is learnable |

---

## Status after W-0 / W-1 (same day)

* Witness built: `development/models/latent_cause.py`, checked against exact enumeration.
  The reference setting is `lc-f-E`: learned hazard, hierarchical prior on features, flat
  prior on contingencies, box regime unobserved while at home.
* T2 v1 implemented (`--set v1`), v0 frozen. Fixed: contingency gap, ABC `ctx3`, home as a
  familiar channel, reinstatement gap at home. Added: SR with sessions (long − short), PREE,
  reinstatement. Criterion .05.
* Witness results: every context row is passed jointly by `lc-f-E` except savings (+.048,
  below the bar). Rows that cfg also passes are labelled guards.
* DEV re-analysis (from caches; "solved" = per-seed score ≥ .3, fixed before computing):

| agent (seeds/variant) | dist-noise3-p4 solved | aut-fooddelay2 solved | aut-fooddelay3-noise1 solved | DEV mean | informative 10 | saturated 5 |
|---|---|---|---|---|---|---|
| v10 (16) | 9/16 | 14/16 | 10/16 | .729 | .629 | .927 |
| v11 (16) | 10/16 | 12/16 | 10/16 | .770 | .671 | .967 |
| v12c (16) | 11/16 | 16/16 | 6/16 | .715 | .628 | .888 |
| v12 (screen, 3–8) | 7/7 | 7/7 | 4/8 | — | — | — |
| v13 (screen, 3–4) | 4/4 | 3/3 | 2/3 | — | — | — |

  Paired on screen seeds versus v11, informative variants without the three bimodal ones:
  v12 +.052, v13 +.133 (3 seeds per variant; indicative only). Most of v12's DEV gain is the
  share of seeds that solve the bimodal variants (exploration), not better play once solved.
* Open:
  * the savings criterion (user decision);
  * the agents on the final v1 (registered predictions in W-1);
  * a v12 per-seed DEV cache for slice decisions;
  * the T1 known-structure reference (with slice 3).

## Status after the night run (2026-11-19/20)

* **Proposal 4 (relative criterion)** — done as a reading aid, verdicts unchanged:
  `eval/t2_relative.py` prints every T2 v1 effect as a fraction of the witness `lc-f-E` and
  the share of seeds with the predicted sign (e.g. v14's PREE: +.22× the witness, 69 % of
  seeds positive — bimodal).
* **Proposal 7 (T1 known-structure reference, slice-3 worlds)** — done with a caveat:
  `kfact` (online HMM per process, told the partition and K) and `kjoint` (told the joint
  size) in `eval/predictive.py`; slice-3 sets d1, d2, dscale, dcouple, d3 in `envs/family.py`.
  Caveat: the known-structure HMM learner is **not** a usable "learnable ceiling" on v1 — it
  predicts base rates on the cue → delayed-outcome, counter and context worlds (GC ≈ 0) where
  windows reach .4–.8. On d1 / d2 / d3 it is a good ceiling (.85–.99). GC against the exact
  filter stays the main number.
* `two-hmm` is confirmed as not a decomposition test (joint 4 states); d1 replaces it for
  slice 3 (kjoint loses .35 GC there on average).
* d2-5x2-weak has a small learnable gap (6–7 % of the marginal loss); the family test's bar
  for it is .95 (logged in decisions.md).
* The v12 per-seed DEV cache exists now (`results/dev16_v12.json`, mean .891); the earlier
  re-analysis row "v12c" is a different configuration.

## Proposal 8 (2026-11-20, not applied; T2 v1 stays frozen)

In T2 v1 the US is an anonymous channel and the value channel is constant 0, while every
latent-cause witness is told which channel is the US, which are cues and which is home. In
animals a US (food, shock) carries innate value; in this interface innate value is the value
channel. A future T2 version should make the US drive the value channel (food +, shock −) and
report the witnesses' role knowledge as a privilege. Changing it now would move a frozen
goalpost mid-slice; recorded for the user's next review.

## Observation for a future audit (AR-3, 2026-11-20 12:23; not a proposal yet, nothing changed)

**T2 v1 reinstatement: the home cage is novel.** The row sends the animal home for 15 steps
before the test (correct: context exposure after the USs would abolish reinstatement), but the
home channel never occurs earlier in that row, so a context-aware model treats home as a *new
place*. In rats the home cage is the most familiar place. Measured (D-31 diagnosis, 6 seeds,
outside the test set): F-12 −.077 as frozen vs +.130 with 20 home steps at the start; parent
+.109 vs +.045. Models without context emission are barely affected, so the row has so far been
fair for every model it judged. Before any change: check the other v1 rows that use HOME
(SR-sessions starts at home, so there home is familiar) and the biology of handling / transport
cues; decide together with Proposal 8.
Checked at 12:25: in v1 only reinstatement has the novel home (SR-sessions starts at home, so
home is familiar there); h2's reinstatement-context had the same property and was corrected
before any candidate run (D-30 amendment); v1 stays frozen.

## h2 / sensory-precond — validity note (AR-4, 13:16, after D-32; the test is NOT edited)
In the paired group the configuration "A without B" has never been seen, while in the control
group "A without X" also has never been seen. But only in the paired group does a configural
set that contains A ({A, B}) win the structure average, because B+ follows AB−. A model then
answers the probe from an unseen cell (its prior), and the scorer reads that ignorance as
transfer (D-32 diagnosis). The row is still valid for models without configural sets (rw passes
0, correctly), but a pass by a configural model requires a mechanism check before it counts.
A future revision (only for a *new* held-out set, never h2) could add a third group
AB− then X+ (B not conditioned) so that ignorance about "A without B" is equal across the
contrast.
