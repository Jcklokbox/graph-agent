# Insights

Durable lessons — about the problem, the model and the method — that should survive any
rewrite of the code. Each entry says where the evidence is. Entries are revised only when new
evidence contradicts them, and the revision is noted.

## About the model (v11)

1. **v11 is a strong delayed-association learner and nothing more.** On worlds whose hidden
   state is recoverable from a finite window of cues it closes ~.9 of the gap to the exact
   filter, even with iid noise channels (T1 v0: delay worlds .87–.94). Evidence: S0-2.
2. **v11 has no belief state.** On a noisy two-state HMM, where evidence must be integrated
   over time, it is exactly at the first-order Markov level (GC .74 vs .74). Evidence: S0-2.
3. **v11 overwrites, it does not infer contexts.** After acquisition and extinction it
   reacquires much *slower* than it first acquired (savings reversed, −.77), shows no
   spontaneous recovery, and its global change point is reset by predictable inter-trial
   steps. Evidence: S0-3.
4. **v11 learns a novel cue in about two trials** (P(US) = .94 after two pairings). Fast, but
   unlike animals and a source of the savings reversal. Evidence: S0-3 raw probes.
5. **Renewal in v11 is configural, not inferential**: explicit context channels keep the
   acquisition cell untouched. A renewal test only diagnoses latent-cause inference when the
   test context is novel (ABC). Evidence: S0-3.
6. **Only three Round-13 decisions are resolved by DEV**: the timing irrelevance test (−.12
   when replaced by attention), the large-sample BF tests (−.06) and the Kamin rule (−.06).
   The other differences are below the paired MDE (≈ .044 at 8 seeds). Evidence:
   `results.md`, R13; S0-1 outcome.

7. **Learned latent chains and anchored timers are complementary.** An online HMM integrates
   noisy evidence almost perfectly (GC .89–.99) but learns nothing on event worlds, because
   silent wait states are not anchored to any observation and online learning never breaks
   their symmetry with idle; v11's windows (anchored, deterministic) are the reverse.
   Plugging the HMM's state into v11 as hard one-hot channels keeps the floor but wastes
   most of its accuracy. Evidence: S1-1.

8. **v11's predictions are not calibrated likelihoods when data are few.** Early in learning
   it is worse than plain Laplace counts on every noisy world, and two trials give .94.
   Used as the likelihood for inference over latent causes it creates causes at once and
   chatters between them (≈ 1000 switches in 2000 steps). Inference over models needs a
   calibrated posterior predictive. Evidence: S1-3.
9. **Soft read-out needs soft training.** Marginalising predictions over a latent belief
   while training on the hard path queries contexts that were never trained, and is worse
   than the hard label. Evidence: S1-3.
10. **Per-channel model choice works.** Bayesian model averaging of v11 and an online HMM per
   channel beats both (T1 v1 .81, v0 .92): different channels need different model classes —
   a first, crude form of decomposition. Evidence: S1-2 A′.

## About the evaluation

11. **Held-out failures concentrate on long delays with noise** (`aut-fooddelay3-noise2-p4`
   −.14 / −.16 in HELD6; `fooddelay3-noise1` .35 on DEV). The behavioural deficit and the
   T1 deficit point in the same direction: hidden state not recoverable from the current
   input. Evidence: `results.md`.
12. **The DEV mean is precise; single variants are not.** Per-seed sd of the 15-variant mean
   is ≈ .05 (paired difference sd .053): 16 seeds resolve ≈ .04, 8 seeds ≈ .05. Single
   variants have CIs of ±.05 to ±.23 even at 16 seeds, so per-variant stories are mostly
   noise. Of the Round-13 decisions only three (timing irrelevance test, large-sample BF
   tests, Kamin rule) are resolved by DEV. Evidence: S0-1 outcome.
13. **Every fixed held-out set gets spent.** Six held-out sets were each run once; the next
   transfer test must come from a procedural generator. Evidence: R2–R13.
14. **A translation of a phenomenon can make it trivial** (renewal with explicit contexts;
    T2 effects with zero seed variance when seeds only jitter the ITI). Translations must be
    fixed in advance *and* stress-tested with a simple learner (Rescorla–Wagner) whose
    textbook pattern is known. Evidence: S0-3.
15. **Easy probe sets confirm what a model was built for.** T1 v0 was dominated by delay
    worlds and did not discriminate; probe sets should be built around the capability that
    is missing. Evidence: S0-2.

## About method

16. **Predictions written before runs are informative even when they hit.** In S0-3 all six
    verdicts were predicted, which shows the mechanism-level understanding of v11 is right;
    in S0-2 the miss (noise pooling better than expected) located a gap in that
    understanding.
17. **Trajectories are chaotic at the ulp level**: refactors are verified by a fingerprint of
    exact outputs (`eval/fingerprint.py`), not by benchmark scores.
18. **Cheap tiers first.** T1/T2 runs take minutes, DEV takes most of an hour on 16 seeds;
    iterate on T1/T2, confirm on T3, transfer on T4.

11. **Calibration first, then exploration from value of information.** v11's overconfident core
    (data counted at every context level) made its sd bonus *look* right (sd collapsed) while
    making every downstream inference wrong (latent causes born ~1000 times per episode). Once the
    predictive distribution is calibrated, the sd of fragmented contexts stays large and a bonus
    on sd over-explores; the bonus that works is the expected improvement of the decision
    (v12: DEV +.08 [+.005, +.15] vs v11). With the calibrated core, latent causes are born
    never instead of always: births need accumulated evidence, not a one-step likelihood race.

## Slice 2 inside 3b (session 2, 2026-11-20; D-16 … D-19)

19. **The context phenomena need no role knowledge.** An observer that treats every channel
    alike — sticky-CRP hidden causes with a learned hazard, the US predicted from the other
    channels — passes every context row of T2 v1; telling it which channels are contexts
    (emitting features) *hurt* PREE (−.47 REV → +.27). Evidence: D-16.
20. **Configural and elemental hypotheses belong in one Bayesian average.** Within each hidden
    cause: parent sets of size ≤ 2 (configural cells) give renewal, contingency, blocking (by
    Bayesian Occam); one additive member gives conditioned inhibition (a summation test is
    symmetric for every configural model). Together 10 / 10 rows of T2 v1 with polynomial
    statistics. Evidence: D-17, D-18; `tests/test_latent_cause_lin.py`.
21. **Fast association needs soft structure, slow structure may be discrete.** Animals learn a
    cue → outcome link in tens of trials; a learner that decides structure by discrete
    proposals after hundreds of steps (the slice-3 witness) cannot condition at all on T2.
    Keep all bounded parent sets in a soft average from step 1; reserve discrete decisions for
    the slow question which channels share a hidden cause. Evidence: F-7, D-19.
22. **Time and configuration are separate competences.** On held-out h1, v12's context-tree
    back-off (a never-seen configuration falls back to a shorter context) gives latent
    inhibition and AAB renewal; the latent-cause witness has regime memory (SR, savings,
    reinstatement, PREE on v1) but no back-off across causes, and loses AAB. The 3b predictor
    needs both: causes for time, back-off for configurations. Evidence: D-24 … D-26.
23. **Keep a held-out battery for the model family you are designing.** T2 v1 had become the
    design set of the `lc-*` family; a six-row held-out set found a real gap (AAB) and
    corrected an unfair judgement of v12 within one hour. Evidence: D-23 … D-26.
24. **Run a memoryless reference on every new translation before freezing it.** Two of five h2
    rows "passed" a configural table for reasons unrelated to the phenomenon — a control that
    showed the target cue alone (novelty), and a readout that subtracted a context probe (any
    model whose context predicts the US passes). The reference run took one second and caught
    both before any candidate was measured. Evidence: D-30.
25. **When adding a mechanism costs an old competence, ask whether one variable now does two
    jobs.** F-11 put place into the latent cause and lost regime recency at every home visit;
    factoring place and regime (F-12) kept AAB and restored the time rows. The segment-level
    MAP trace (one script, one seed) showed it faster than any parameter search. Evidence:
    F-11 diagnosis, D-31.
