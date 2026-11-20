# Slice 3 — proposal (draft for the user, written in the night run 2026-11-19/20)

*Nothing of the slice-3 agent is started. The slice's hypothesis, translations and budget need
the user's agreement (goal.md §6). This file collects what the night run measured so the
decision can be made on data. Details and every prediction: `decisions.md` D-0 … D-8.*

## 1. Measurements

Yardsticks (T1: next-observation prediction vs the exact filter, random policy, GC on the
second half; all new sets are in `envs/family.py`):

| set | what | used for |
|---|---|---|
| `d1` | 5 worlds, 3–8 independent hidden processes, joint 27–256 | development of the witness (seed 0 of three worlds) |
| `d2` | 4 held-out structures (weak chains, 4-state chains, unequal groups, two delayed cues) | generalisation |
| `dscale` | P = 2, 4, 6, 7 three-state chains (joint up to 2187) | cost vs number of processes |
| `dcouple` | coupling through transitions | boundary (uninformative: coupling carries ≤ .08 GC) |
| `d3` | action-dependent lever; conjunctive channel (food iff lever AND light) | agent-relevant structure |

References (`eval/predictive.py`): `kfact` — online HMM per process, **told** the partition and
K (oracle structure); `kjoint` — one HMM told the joint size; `decomp` / `decompw` — the
feasibility witness (`models/decompose.py`) that **learns** the partition from anonymous
channels; `decompw` mixes two model families (HMM_K, window) inside each group.

| learner | v1 (8) | d1 (5) | d2 (4) | dscale | d3 (3) | cost |
|---|---|---|---|---|---|---|
| markov4 | .71 | .16 | −.10 | | .70 | ≈ 0 |
| v12 (predictive core) | .655 | ≈ .29 (3 worlds, 2000 st.) | — | | — | 175 s / 2000 steps on 13 ch. |
| kjoint | — | .58 | .51 | | .84 | |
| kfact (oracle partition) | .48 | .93 | .85 | .89–.95 | .97 | |
| decomp | .49 | .94 | .81 | | .98 | |
| **decompw** | **.77** | **.95** | **.90** | **.89–.95** | **.98** | ≈ 1/10 of v12; linear in P |

* Necessity: v12 pays the product of the processes' state spaces (≈ .29 where the oracle
  partition gives .93; ≈ 40× the oracle's cost).
* Feasibility: the partition is learnable online; `decompw` matches the oracle on d1, d2,
  dscale, d3 and beats v12's predictive core on the slice-1 set v1 (paired +.116
  [+.098, +.135]).
* Cost: linear in the number of processes (isolated timing; D-5).

## 2. Limits found (must travel with any claim)

1. **Merge latency vs fast association (D-8).** A merge needs ≈ 500 steps of evidence; a
   conditioning experiment is ≈ 60 trials. On T2 streams the witness never merges, so it cannot
   learn cue → US at all. Animals associate within a few trials.
2. **Conjunctions (D-7).** A channel that depends on two processes joins both of them in only
   5 / 12 runs; on that channel (the food, for an agent) this costs .07–.13 GC.
3. **Evidence is T1 only.** Prediction under a random policy; no value, no control, no T2, no
   DEV for the witness.
4. Weakly jointly-informative groups can stay split (v1 noisy-identity .43 vs v12 .74).
5. The FHP family cannot express emission-level coupling except through the new `links`.
6. **No cheap calibrated fast structure test (D-9, D-10).** Fast pairwise tests merge
   independent autocorrelated processes; calibration requires the slow test.

## 3. Proposed hypothesis

> The agent's predictive model is a set of channel groups found by Bayesian partition
> inference; each group predicts with a posterior mixture over model families and sizes
> (hidden-state HMMs, windows). Under it, a fast associative layer (today: v12's context
> tables) learns within the current structure from the first trials; the partition is revised
> slowly from accumulated evidence. Cost = sum over groups.

Why: one stated generative model at the structure level, every structural decision a
predictive-likelihood comparison; the witness reaches the oracle where v12 pays the product;
limit 1 says a fast layer is needed anyway.

## 4. Integration options

| option | idea | build cost | main risk |
|---|---|---|---|
| A | witness groups replace v12's tables for prediction | high | loses fast association (limit 1), T2 rows, DEV |
| B | v12 unchanged, but every context key restricted to one discovered group | medium | still a product of windows inside large groups |
| **C** (recommended) | groups (slow, Bayesian partition) + inside each group v12's tables (fast) and the HMM family, mixed by posterior weight | highest | complexity; must keep DEV (v12 .891 on the cache) |

Order I would propose (revised after D-9 / D-10): the cheap single-timescale "fast and
calibrated" merge test does **not** exist by the routes tried — pairwise count tests merge
independent sticky processes (Yule's spurious correlation; null misspecified while a channel's
own model is young). So: (1) option B as the cheap necessity check inside v12 (does restricting
v12's keys to slow-discovered groups keep DEV and T2?); (2) C, where fast association lives
*inside* the current partition (v12 tables, no structure decision needed to learn cue → US
once cue and US share a group — or a default "all-in-one fast group" that is split slowly);
the partition is revised only by the slow, fixed-horizon test. The fast variants `decompf` /
`decompr` stay in the code as the record of D-9 / D-10.

## 5. Criteria (proposal)

* T1 d1 + d2: GC within .1 of kfact on average; cost per step linear in the number of groups
  (isolated timing on dscale).
* T1 v1 not worse than v12 (paired, 6 seeds); d3 food channel ≥ .95 of the oracle.
* T2 v1 guards kept; DEV not worse than v12 (screen against `results/dev16_v12.json`).
* At most two expensive runs (goal.md §6).

## 6. Questions for the user

1. Approve d1 + d2 (+ dscale, d3) as the slice-3 yardstick with kfact / decompw as references?
2. Option order (B → C), or straight to C?
3. Slice 2 (not accepted; best partial v13, 2 / 4 context rows; v14 rejected): accept the
   anonymous-channel target proposed in global reflection 1 (rows passed by `lc-f-H`), keep the
   current target, or park slice 2 until the decomposition core exists (contexts may become a
   slow process of the same model)?
4. The decision about the v12 core (yours, after slice 3). Fact for it: DEV of the accepted v12
   is .891, paired vs v11 +.121 [+.093, +.150] on 16 seeds.

## 7. Self-answers to §6 (2026-11-20, at the user's request; reversible)

Method requested by the user: at a serious fork, spend a separate deliberation step on
proposals, questions and answers (goal, biology, mathematics, common sense) instead of handing
control back.

1. **Yardstick.** Yes, re-weighted: d2 (held-out structures) and d3 (action-controlled process,
   conjunctions) are primary — an animal's world is mostly loosely coupled causes, its own body /
   place is the main controlled process, and food is typically a conjunction. d1 is development
   only; dscale is for cost only; dcouple is dropped (uninformative, D-6). Main number: GC against
   the exact filter; kfact is a bound only where it works (not v1); decompw is an existence
   proof, not a target. T1 alone is not enough: T2 guards and DEV stay mandatory.
2. **Order.** Neither "B then C" nor full C. B is predictable from D-8 (keys limited to slowly
   found groups cannot learn cue → US early) → by P-1 not run first. Start with **C0**: v12
   unchanged + one HMM-family predictor per slowly found group, mixed per channel by posterior
   weight. Precedent: slice-1 A′ (v11 + one global HMM, per-channel BMA) reached GC .81 on v1
   (S1-2). A Bayesian mixture loses at most log(1/prior weight) nats per channel over the whole
   stream against its best member, so T1 ≥ max(v12, witness) up to a constant, and fast
   association stays v12's. Known gap of A′ to close in C0: planning must use the mixture's
   predictions. Key restriction to groups (B) comes afterwards, inside C, as the cost step.
3. **Slice 2.** Correction of my earlier proposal: do **not** lower the target to `lc-f-H`. The
   agent knows the value channel, so `lc-f-E`'s distinction (contexts share appearance, differ in
   contingencies) is available to it; renewal / SR / reinstatement / PREE are robust in animals.
   Keep the frozen T2 v1 and the `lc-f-E` target; **park** the v12 patches (v13 / v14 showed the
   patch route is piecemeal) and resume slice 2 as one more process family inside the
   decomposition core: a slow hidden process whose number of states grows (sticky CRP / HDP-HMM).
4. **v12 core.** Keep v12 as the fast associative layer and controller, not as the whole model.
   Formal decision at the decision point after C0, pre-set reading: if the v12 component gets
   negligible weight on most channels *and* DEV is unchanged without it, its tables become
   optional; otherwise v12 stays a permanent component.

Note on the user's remark about brain topology (a hypothesis, not a constraint): topographic
maps form by slow activity-dependent development, and learning is constrained by priors
(Garcia & Koelling 1966: taste associates with illness, light / sound with shock). Read here:
wiring stores the result of slow structure learning and makes fast association cheap and local —
the two-timescale design. Channels stay anonymous (no use of channel order); the analogue would
be a learned placement / prior over partitions.

## 8. Slice 3b — plan (2026-11-20, after F-4 and D-15 … D-18; agent's own decision, reversible)

**Hypothesis.** One generative model covers T1 and T2: factorised hidden causes (one sticky
hidden variable per group of channels, state count growing as in a sticky CRP / HDP-HMM) and,
for every channel, a Bayesian average over {parent sets of size ≤ 2 among bounded candidates,
one additive model}, conditioned on the hidden cause of the channel's group. v12's mechanisms
(context-tree mixing, VOI, timing windows, blocking / inhibition) are re-expressed on these
keys; the planner works on the factorised belief.

**Evidence so far (witnesses, not agents).** T1: `decompw` / `decompc` (groups with hidden
state; GC d2 .90, d3 .98). T2: `lc-sub-add` 10 / 10 rows of T2 v1 without role knowledge
(D-18). Neither covers the other's battery (F-7). Cost of the T2 predictor ∝ m² candidates.
Agents (session 2, seed 0 only, anecdotal): v17 (all channels as candidates) and v18 (4 slots)
show SR, savings, PREE, blocking, CI and ABA in the predicted direction; the 16-seed v17 run
(D-20) was stopped unfinished at its budget.

**Steps (each with its own pre-registration in `decisions.md`).**
1. 3b-1 *predictor*: `models/factor.py` — Decomposer's groups as the hidden causes (sticky,
   growing states), per-channel structure average over parents ≤ 2 from the candidate list
   (own group + groups linked by the structure evidence, ≤ 6 channels) + additive member,
   action in the key. Criteria: T1 d2 / d3 GC within .02 of `decompc`; T2 v1 rows as
   `lc-sub-add` (≥ 9 / 10 yes, savings by direction); inference ms/step linear in processes on
   dscale (ratio s-4x3 / s-2x3 ≤ 2.5).
2. 3b-2 *agent*: v12's planner and reward model reading the 3b-1 predictor (reward keyed on
   value's parents). Criteria: DEV paired vs the v12 cache, CI lower bound ≥ −.02; full step
   linear on dscale; T2 as in step 1.
3. Acceptance of slice 3 (and slice 2 within it) = step 2's criteria together with §5.

**Open risks.** Particles: `lc-sub-add` uses 1000; 3b needs ≤ 200 per group (to be measured
first). The additive member is a clipped Gaussian regression. SR and reinstatement are half the
target's sizes (allowed by T2 v1: verdicts, not sizes). T2's anonymous-US issue (§7 p. 3,
Proposal 8) stays open.

**Addendum (07:20 UTC, during D-20).** Cost of v17's cause predictors (inference only, 200
steps, measured while D-20 occupied both CPUs — ratios only): 28.5 ms/step on s-2x3 (7
channels, 8 candidates each) → 142 ms on s-4x3 (13 channels, 14 candidates): ×5 for ×1.9
channels, as expected from channels × sets (sets ∝ candidates²). DEV has 12 inputs and 3
actions (14 candidates per channel), so v17 on DEV-16 would take several hours — DEV for the
cause predictor waits for bounded candidates. Concrete 3b-1 design for the candidates:
own channel + action + the k = 4 channels with the largest pairwise sequential evidence
(lag-1 KT log Bayes factor of "i informs j" against j's own null, the Decomposer's `fast`
counters: cheap, O(n²) counts, fast enough for tens of trials); candidate lists may change
only when a channel's evidence overtakes the weakest member by a margin, and a newly admitted
set starts from the prior (no history replay). Expected cost per channel ≈ 3–4 ms at 200
particles (sets 37), i.e. linear in channels.

## 9. 3b predictor after AR-3 (design notes, not commitments)

**Evidence.** T2 v1 + h1 (16 rows): v12 10, witness 14, v18 14; v12 fails only on *time* rows,
the witness only on AAB and overexpectation (D-22 … D-26). Cost: v18 ≈ ×3.8 v12 on DEV.

**Why AAB needs more than back-off.** Inside the extinction cause even a context-tree mixture
predicts "A → nothing" for the unseen (A, ctx3): the parent node (A) has only extinction data.
v12 passes AAB because it has *no* extinction cause — its (A) node pools both phases. In
latent-cause accounts (Gershman et al. 2010) the novel context lowers the posterior of every
known cause, so a new cause (whose US prior is the pooled one, ≈ .5) gains weight. That needs
each cause to model the *context*, which F-9 showed must not be done with i.i.d. per-step
emissions of all channels (sparse cue onsets then always favour a fresh cause).

**Candidates, in order of principle (each needs its own pre-registration):**
1. Each cause models every channel as a Markov chain, P(x_t | x_{t−1}, cause), with the
   hierarchical (pooled) prior for new causes. Phasic cue onsets are then equally likely under
   known causes and under a new one (pooled prior), while a never-seen tonic state (ctx3 on,
   ctx1 off) is unlikely under every known cause → evidence for a new cause. No role knowledge:
   tonic vs phasic is just what the transition counts say. Development on v1 + h1, then a fresh
   held-out set h2 before any claim.
2. Shared causes across channels (one context variable, as in COIN): one filter instead of one
   per channel — the natural model of a context and the main cost lever (prior, sampling and
   resampling once per step; per-channel tables remain).
3. Overexpectation: an additive member whose error matters where configural cells fit — open;
   no candidate yet that is not a parameter.

**Tests to keep:** v1 + h1 (as development now), DEV regression check, T1 with the cause
member (never measured), dscale cost.

**Status after AR-3 (12:23).** Candidate 1 tried in two forms: F-11 (context emission inside the
cause; AAB yes, time rows halved — one variable for place and regime, diagnosis in decisions
12:15) and **F-12** (place factor × regime factor, regime redrawn only on the first visit of a
place; D-31: v1 9 / 10, h1 5 / 6, AAB +.42 and SR-sessions +.18 together; reinstatement lost
through the novel-home property of the v1 translation). F-12 is the context-model candidate for
the 3b predictor; it has not been judged on the frozen h2 yet.
