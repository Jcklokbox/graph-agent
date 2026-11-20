# 3b-1 — one predictive model `models/factor.py` (design note, AR-4 G3, 2026-11-20 13:25 UTC)

Status: design note, written before any code. It replaces the "mixtures on top of v12" line (v19
cancelled 13:07). The controller of v12 (planner and reward model) may later read this
predictor as an interface (3b-2), but v12's predictive tables are **not** mixed in. Whether 3b-2
needs them at all is decided by D-33 (prediction) and G2b (control).

## 1. What the model must explain (evidence, not taste)

| source | requirement |
|---|---|
| T1 d1–d3, dscale (`decompw` .90–.98) | hidden processes: groups of channels with their own hidden state and a growing number of states |
| T2 v1 (`lc-sub-add` 10/10) | latent causes (regimes) with a sticky CRP; inside a regime, a Bayesian average over parent sets of size ≤ 2 plus an additive member (blocking, CI, contingency, renewal, savings, PREE) |
| h1 AAB, SR-sessions (F-11 → F-12, D-31) | place and regime are **two** variables: a novel place makes other contingencies plausible, while a familiar place change does not reset the regime |
| D-32 reinstatement-context | the regime prior must be **conditioned on the place** (F-13): what was learned in ctx2 must not carry into ctx1 by mere persistence |
| D-32 sensory-precond / superconditioning | **an unseen configural cell must borrow from its sub-configurations** (hierarchical back-off), not take a flat .5 |
| DEV (v12 .891), T1 noisy-cue-delay3, T2 time rows | **delay windows are mandatory**: the candidates of a channel include lagged copies x_i(t−d), d = 1…D |
| cost (v18 ×3.8 v12; v17 quadratic) | one shared filter instead of one per channel; bounded candidates per channel; cost linear in channels |

## 2. Generative model (one particle = one joint hypothesis)

Per time step t, with action a_t:

1. **Place** y_t (sticky CRP, hazard h_p learned on a grid). Each place models every channel as a
   Markov chain P(x_{j,t} | x_{j,t−1}, y), with the prior mean pooled over the particle's places
   (F-12, unchanged).
2. **Regime** z_t, one per particle and shared by all channels (as in COIN). Its prior is a
   hierarchical CRP indexed by place (F-13):
   - if y_t = y_{t−1}: sticky, z_t = z_{t−1} with probability 1 − h_z, otherwise drawn as in
     the next case;
   - if y_t ≠ y_{t−1} (a place is entered), or a sticky switch occurs: z_t ~ P(z | y_t)
     ∝ n_{y,z} + β · G(z), where G(z) ∝ m_z (global popularity) plus α for a new regime, and
     n_{y,z} counts how many times regime z was entered in place y (an HDP "franchise"). A
     never-visited place has n = 0, so G alone applies: F-12's redraw by popularity becomes
     the special case.

   Consequence: free USs in ctx2 build regime counts *in ctx2*. Back in ctx1, the prior is
   ctx1's own history (acquisition, then extinction), so a ctx2-specific regime does not
   follow the animal (reinstatement-context). Returning home restores home's regime
   (familiar place, no reset). β is the only new constant (β = 1, not tuned).
3. **Processes** s_{g,t}: one hidden Markov variable per channel group g (the Decomposer's
   groups, growing states, online HMM). This factor is independent of the regime in this
   step, and for a channel it acts as one more member of its average (§3).
4. **Channels.** For every channel j:

       P(x_{j,t+1} = 1 | ·) = Σ_M w_{j,M}(z) · p_{j,M}(keys_t, z)

   The members M are:
   - **parent sets** S ⊆ C_j with |S| ≤ 2, including the empty set. C_j is j's candidate
     list: its own lags x_j(t−d), the action a_t, and k slots. A slot is a **(channel, lag)
     pair**, d ∈ {0 … D}, D = 4. A cell's Beta prior has mean equal to the average
     prediction of its sub-sets (S \ {i}) at the same input and concentration c = 2
     (back-off). The empty set backs off to the regime's base rate, which backs off to the
     pooled base rate.
   - **additive**: a Bayesian linear regression over C_j (clipped), as in `lc-sub-add`;
   - **process**: the predictive marginal of j's group HMM.

   The weights w_{j,M}(z) are the marginal likelihoods of the members on the data of regime
   z (Bayesian Occam). The table statistics are kept per regime and per particle; the
   process member is not regime-indexed.

**Slots** are filled by pairwise sequential KT evidence per (i, lag) → j: the `fast` counters
of the Decomposer and the v18 screen, extended to lags. A slot is swapped when an outsider
beats the weakest member by log 3. Swapped sets inherit the evidence of their subset and
start from the back-off prior, so they lose nothing.

## 3. Inference

- One SMC filter over (h_p, h_z, y, z, place and regime tables) with N = 200 particles,
  shared by all channels. Weight update: P(x_{t+1} | particle) = Π_j p_j, i.e. every channel
  is evidence about the regime. The risk is noted in §6 r1.
- Process HMMs and slot screening are deterministic: they live outside the particles, which
  makes them cheap.
- Cost per step ≈ N × Σ_j |sets_j| for the update, plus the place likelihood N × K × n. With
  k = 4 slots + action + 2 own lags, the candidates number ~8–9 → ~45 sets per channel. This
  is linear in channels.

## 4. Build steps, each with a mechanism check before any battery

| step | adds | mechanism check (1 seed, printed internals) | battery |
|---|---|---|---|
| F1 | shared filter: place × regime (HCRP), per-channel sets with back-off over **unlagged** candidates (as `lc-sub-add`'s keys) | AAB (novel place → regime mass on new), reinstatement-context (no ctx2 regime at the ctx1 test), superconditioning cell back-off (the cell (A = 0, X = 1) predicts ≈ 0 before XB+) | T2 v1 + h1, 16 seeds |
| F2 | lagged slots (D = 4) | trace conditioning with a 3-step gap: the US predicted at the right lag | T1 v1 noisy-cue-delay3 |
| F3 | process member | GC on d2 / d3 with seeds 0–1 against `decompw` | T1 v1, d2, d3 (6 seeds) |
| judge | — | — | T1 (v1, d2, d3) and T2 (v1, h1) against `decompw`, `lc-sub-add`, F-12, v18; cost on dscale |

Acceptance of 3b-1 stays as in slice3-plan §8: GC on d2 / d3 within .02 of `decompc`; T2 v1
≥ 9/10; cost ratio s-4x3 / s-2x3 ≤ 2.5. h2 is spent for F-12 and the parent, but **not** for
factor.py, so it can be run once as a held-out check after the judge (5 rows; the
sensory-precond row needs the mechanism check of `test-audit.md`).

## 5. What is deliberately left out

- Overexpectation (open, no non-parametric candidate).
- Cue emission by the regime (Courville's sensory preconditioning). Places emit the tonic
  state, which is enough for context. Phasic cue emission was F-9's failure mode.
- Reward and value: these belong to 3b-2.

## 6. Risks, each with how it would show

- r1. A shared regime driven by all channels: noisy irrelevant channels may split regimes. It
  would show as T2 rows with many births on h1 / v1 (probe_t2 counts births).
- r2. Back-off may weaken blocking or negative patterning (configural cells now start near the
  elemental prediction). It would show as NP or blocking failing on h1 / v1.
- r3. 200 shared particles over (place, regime) can degenerate. It would show as ESS collapse
  and high variance across seeds.

## 7. Build log (F1, 13:30–13:43)

- **Speed.** The first version took 17–20 ms/step. The (regime, channel, cell, particle)
  layout, float32 tables and one exp per call brought it to ~10 ms/step. The pooled tables
  are particle-independent (every particle sees the same data), so they are stored once.
- **Fix 1: one emission per observation.** The mechanism check on h1 AAB, seed 0, showed a new
  *place* born at the start of extinction (t ≈ 215, context unchanged). The place emitted every
  channel x_t, the US included, as a Markov chain, while the regime's members predicted the
  same x_{t+1}: every observation was generated twice. Now the place chain is a **member** of
  each channel's average, weighted like the others by its marginal likelihood in the regime.
  (place, regime) of each transition is sampled jointly from the exact posterior per particle
  (KP × K cells).
- **Fix 2: two-level back-off.** At the AAB test in ctx3, the new regime predicted A → 0,
  because its unseen A cell fell back to that regime's base rate. A regime's cell now backs off
  to the **same cell pooled over all data** (a new regime is expected to resemble the known
  ones, the hierarchical prior of F-11 applied to cells). A pooled cell never seen backs off to
  its sub-configurations (D-32). Both are statements about the generative model, and neither
  adds a tuned constant (C_REG = C_GLOB = 1 = 2a of the Beta(.5, .5) witnesses).
- Probes are *hypothetical* observations: the transition from the last learned observation to
  the probe is used as evidence about (place, regime) before predicting, as `lc-*` do with
  P(x | cause).
- Seed-0 look (development rows only): blocking .010 / .448, ABA .727 / .004, ABC .123 / .004,
  SR .379 / .100, savings acq2 .734 → reacq2 .899, NP A .998 AB .001, AAB .146 vs AAA .100,
  inhibitor-alone BX .329 vs BY .304.

## 8. F2 / F3 and r4 (14:02–14:10)

- **F2 (lags).** Slots are (channel, lag) pairs with lag 0–4, plus own lags 0–2 (5 slots, ~56
  sets per channel). On T1 v1 (2 seeds) the GC rose from the unlagged level to .85. The three
  delay worlds reach .84 / .89 / .95, above both markov4 and v12, so D-33's requirement is met.
- **F3 (processes).** The Decomposer (groups, windows) gives one more member per channel,
  weighted per regime by its marginal likelihood: two-hmm .85 → .95, mean .87.
- **r4, place identifiability.** As a member, the place chain competes with the regime's
  own-lag sets and earns no likelihood. The place then barely moves (novel home not
  detected, D-34 diagnosis). Fix (`place="view"`): the place's chains emit x_t as a
  *separate view* of the same input, as in F-12. They are not mixed into the channel
  predictions and act only through the regime prior.
  Biological reading: context is inferred by its own pathway (hippocampal), the
  contingencies by another, and both read the same senses. Formally the joint is two coupled
  models of one input. The channel predictions are not double-counted, but the particle
  weight contains the place's evidence for x_t in addition to the regime's for x_{t+1}. This is
  an approximation, stated as such.
- **Not solved.** v1 reinstatement: after a never-visited home, the return to ctx1 redraws the
  regime from ctx1's own history (acquisition and extinction ≈ 1 : 1), so both groups recover
  (≈ .36). F-12 failed this row for the same reason. h2's reinstatement-context (familiar home)
  is the cleaner test and remains held out.
- **3b-2 interface ready.** `agents/v20.py` is v12's controller reading only this predictor at
  depth 0, with the value channel taken from v12. Smoke test: 60 DEV steps run, 161 ms/step
  under full contention.

## 9. Open design targets after D-36–D-38 (15:30)

1. **Conditional screening (r5, deep form).** Candidates are now scored pairwise:
   log p(x_j' | x_j, x_i,t−d) − log p(x_j' | x_j). A cue that matters only *given* another
   cue (X in A+/AX−, the second cue of an AND) is scored on the marginal, so it scores near
   zero. Proposal: score a candidate against the model's *own* prediction q_j of the same
   transition, with counts in bins of q_j:
   s_i += log p(x_j' | bin(q_j), x_i) − log p(x_j' | bin(q_j)).
   This is the conditional mutual information given the current model, i.e. error-driven
   attention (Mackintosh, Pearce–Hall). Slotted candidates keep their pairwise score for the
   swap test, because given q they score ≈ 0 by construction. Cost: one extra table of
   (J, L, J, B, 2, 2). Predicted effect: CI no longer depends on late entry, and the
   lagged copies of A (timing that q already explains through A·0) stop winning slots. The
   `one_lag` restriction could then be removed, which should recover noisy-identity.
2. **Context × time (r4).** Reinstatement and SR-sessions are the rows still failing. Both
   need the place / regime prior to depend on elapsed time and on surprise at a familiar
   place. A member place loses to own-lag sets, and a view place costs savings. Candidate
   (not built): a place chain with its own weight that never competes with the channel
   members. Place evidence would enter only through the regime prior, and the place's
   emission would be fitted to the *context channels only*, not to all channels. In the
   view mode the place also emitted cues and US, so place splits tracked
   acquisition / extinction, and that is plausibly what cost savings.
