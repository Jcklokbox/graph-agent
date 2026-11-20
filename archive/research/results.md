# Results

Measured behaviour of `development/agents/v11.py` (formerly `posterior_graph.py`) and its known failures. The model is
described in [model.md](model.md). The full round-by-round log of Rounds 1–12 (component
probes, trajectory diagnoses, every rejected idea with its numbers) is archived in
[`../archive/research/round-log-r1-r12.md`](../archive/research/round-log-r1-r12.md).

Reproduce (from `development/`, NumPy needed for the benchmark):

```
python -m unittest discover -s tests -t .                         # 36 tests, ~50 s
python -m eval.benchmark --dev                                    # DEV-15 x 8 seeds, ~15 min on 2 CPUs
python -m eval.fingerprint --check eval/fingerprint_v11.txt       # bit-exact behaviour check
python -m eval.benchmark --held6                                  # Round-13 held-out set (now used)
```

## Current agent of record: v18 (slice 2 accepted, AR-3, 2026-11-20)

`development/agents/v18.py` = v12 (accepted step 2) + a per-channel latent-cause predictor
(`models/latent_cause_lin.LatentCauseSubAdd`, sticky CRP × structure average over parent sets
≤ 2 + additive member, 4 candidate slots per channel, 200 particles), mixed per channel by
posterior weight. Decisions D-21 … D-29 in [decisions.md](decisions.md).

| tier | v12 | v18 | source |
|---|---|---|---|
| T1 v1, mean GC (8 hidden-state worlds × 6 seeds) | .66 | **.76**, paired +.103 [+.086, +.121] | D-29 |
| T2 v1 (10 rows, design set) | 5 (2 reversed) | **10** | D-22 |
| T2 h1 (6 rows, held out) | **5** | 4 (loses renewal AAB) | D-25, D-26 |
| DEV (15 variants), paired v18 − v12 | — | −.000 [−.044, +.044] (99 %), equivalent at δ .045 | D-27 |
| cost per DEV episode | 1 | ×3.8 | AR-3 C4 |

Open: AAB (context model F-11, D-28, costs time effects), overexpectation (no model passes),
absolute cost (shared causes across channels), held-out h2 before any generalisation claim.

The sections below describe v11 (Round 13) and are kept as the historical record.

**Score** per variant = (brain − stay)/(oracle − stay) on the episode tail (last 200 steps;
for serial reversal every step after the first phase); 0 = always "stay", 1 = oracle.
**Protocol.** Configurations are chosen on DEV only (15 variants × 8 seeds). A fresh held-out
split is written into `tasks.py` before any run and is run once on the final code; the
previous version is run on it afterwards as a control. The per-seed spread on the noisy
variants is about ±.4, so an 8-seed DEV mean of one variant has a standard error of ~.14 and
DEV means differing by less than ~.01 are treated as equal.

## Round 13: consolidation (v11)

Goal: one model instead of twelve rounds of layers — fewer mechanisms, one statistic, one
structure test, one combination rule — without losing quality. Every candidate removal was a
DEV ablation; a removal was kept when DEV stayed equal or improved, and the code of rejected
variants was deleted.

### What changed in the model

| change | DEV all / variants | decision |
|---|---|---|
| v10 (start) | .740 / .689 | |
| − pair (conjunction) births and their 2×2 BF | **.774 / .725** | removed: exact/attended contexts represent seen conjunctions; the pair prior hurt in unseen contexts (aut-noise4 .783 → .940, canc-noise3 .687 → .800) |
| … − Kamin blocking | .715 / .660 | kept |
| … − learned inhibition | .741 / .682 | kept (aut-fooddelay3-noise1 .369 → .001) |
| … − exact-context controllability (attended only) | .771 / .721 | removed (one controllability sum) |
| … timing context by attention instead of the irrelevance test | .658 / .630 | kept the test (dist .992 → .361) |
| … − protection of strong cue cells from cancellation credit | **.783 / .735** | removed |
| … (attended controllability) + CUSUM change point for event counts too | .757 / .708 | kept memoryless counts (delayed food breaks) |
| … + global discount of attended/reward cells too | .774 / .725 | adopted (no exceptions) |
| … action-pooled cells shrunk like the chain instead of mixed | .735 / .681 | kept plain mixing |
| … exact split BF > 0 for timing credibility and relevance | .708 / .649 | kept the large-sample tests |
| pairs, exact controllability, protection removed (combined) | .778 / .729 | base for the last batch |
| … − empirical-Bayes strength of the reduced timing level | .780 / .732 | removed (fixed `prior_strength`) |
| … − coarse (attended) reward level | .773 / .724 | kept (aut-noise4 −.10, dist-noise2 −.10) |
| **v11 = all adopted changes** | **.775 / .727** | final |

Rows marked "…" are single changes on top of the row they follow (the first block on top of
"− pairs", the second on top of "− pairs, attended controllability", the last on top of the
combined base).

### DEV per variant (8 seeds)

| variant | v10 | v11 |
|---|---|---|
| aut | .852 | .915 |
| canc | .995 | .995 |
| dist | .992 | .992 |
| aut-half | .852 | .915 |
| aut-noise4 | .783 | .871 |
| dist-delay3 | .972 | .972 |
| aut-serial300 | .706 | .714 |
| aut-noise4-sticky | .840 | .866 |
| canc-noise3 | .687 | .773 |
| dist-noise2-sticky | .876 | .881 |
| dist-noise2 | .723 | .704 |
| aut-fooddelay2 | .630 | .681 |
| dist-delay3-noise2 | .590 | .675 |
| dist-noise3-p4 | .226 | .324 |
| aut-fooddelay3-noise1 | .382 | .353 |
| **all / variants** | .740 / .689 | **.775 / .727** |

### HELD6 (fresh, 12 seeds, run once; v10 = control, run afterwards)

| variant | v10 (control) | **v11** | v11 wins vs stay |
|---|---|---|---|
| aut-noise7-p3 | .740 | **.819** | 12/12 |
| canc-noise5-sticky | .864 | **.919** | 12/12 |
| dist-delay4-noise1 | .872 | .875 | 12/12 |
| aut-fooddelay3-noise2-p4 | −.137 | −.160 | 3/12 |
| aut-serial325 | .699 | **.737** | 12/12 |
| dist-noise3-sticky | .390 | **.627** | 11/12 (v10 9/12) |
| aut-cube | .757 | .751 | 12/12 |
| canc-half | .981 | .981 | 12/12 |
| aut-dupaction-noise2-sticky | .759 | **.847** | 12/12 |
| dist-delay5-sticky1 | .779 | .771 | 12/12 |
| **mean** | .670 | **.717** | 110/120 (v10 108/120) |

**Reading.** The simpler model is not worse on fresh variations: +.047 on the mean, better on
5 variants by more than .03 (four noise variants, where the removed pair prior and the
fragmented controllability evidence hurt, and serial reversal), within ±.03 on the other five. HELD6 is easier
than HELD5 (.204): it has only one delayed-food-with-noise variant, which is again the one
clear failure (both versions below "stay": the food that follows the precursor 3 steps later
is not located among 2 iid noise channels with p = .4; late shock rate .135). The absolute level is therefore not comparable across held-out sets; the comparison
with the control on the same set is.

### Code and project

* `posterior_graph.py`: 1863 → 1243 lines. All tables of one regime are fields of one `Model`
  dataclass (archiving and reinstating a regime is `Model.copy()` / swap instead of a list of
  attribute names); one `Cell` type replaces the outcome posterior, hazard pairs and count
  lists; one `split_log_bf` serves attention and controllability; one `shrink` serves the
  chain. No experiment switches remain.
* Behaviour preservation was checked bit-exactly at each pure refactoring step
  (`trace_fingerprint.py`: hashes of every root Q value and action), e.g. the refactored code
  against v10 with the same removals applied through temporary flags: identical on all 15
  DEV episodes. The reference was re-saved for v11.
* The task suite (`tasks.py`) and the benchmark (`benchmark.py`) replace three scripts
  (`benchmark_envs.py`, `transfer_suite.py`, `run_posterior_scenarios.py`); the original
  `BlindGateGraphBrain` prototype, its environment, bounded algebra and tests, nine superseded
  research notes and the Round 1–12 log moved to `archive/`.
* Tests: 31 (pair tests removed; added: split Bayes factor, closed-loop equivariance under
  relabelling of sensors and actions; the timing-pooling test now checks the reduced count).

## Earlier rounds (summary; details in the archived log)

| round | version | main change | held-out (run once) | score (control) |
|---|---|---|---|---|
| 3–5 | — | surprise forgetting, τ-synchronised memory, observational inhibition, one optimism budget | base tasks, 12 fresh seeds | — |
| 7 | — | reference-point value coding, learned attention as a coarse level | HELD | .114 (no attention −.140) |
| 8 | — | trajectory-diagnosed fixes for dist, complex noise, delays > 2 | HELD2 | .354 |
| 9 | — | speed only (behaviour bit-identical) | — | — |
| 10 | v8 | noise fragmentation of timing, delayed food, change points and savings | HELD3 | .477 (v7 .315) |
| 11 | v9 | irrelevance instead of relevance, error-gated regime retrieval | HELD4 | .517 (v8 .517) |
| 12 | v10 | controllability on the attended context, earlier regime storage | HELD5 | .204 (v9 .155) |
| 13 | v11 | consolidation (above) | HELD6 | .717 (v10 .670) |

## Open problems

* **Deep exploration** in the distant-goal task with iid noise (dist-noise3-p4 ≈ .3): a test
  that needs two costly steps after a cue is attempted only when the immediate bonus is large.
* **Gain/loss asymmetry** of the value frame (dist-sqrt on HELD5: no exploration).
* **Delayed food with noise** (aut-fooddelay3-noise1 ≈ .35 on DEV) stays near "stay" on fresh
  variations.
* **Level 0** (noisy-OR of cue lifts, inhibition, blocking) is the least principled part of
  the model and the part that unseen contexts depend on; both of its hand-made mechanisms
  survived removal tests.
