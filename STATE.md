# STATE (SSOT) — read first after any compaction
Only .md in the active tree; history in archive/ (unchanged, excluded via .ignore, not importable).
Env: `pip install --target /tmp/graph-agent-deps numpy`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.
Trees (verified by tree_hash(development/)): `faba3f2bd496` = the pushed tree, 67 tests OK on it; `f0c6035b59c0` = the S0→S1 screen below. They differ by one line: the removal of `self._prevb`, which was assigned and never read (grep-verified) — no behaviour change, so the screen stands.

## goal
One generative agent (M_factor = `factor-L4-P-C`, models/factor.py CANONICAL) reproducing T2 and transferring on T1 at cost ≤ 2.5.
Gate 3b-1: A T2 v1 ≥ 9/10, 0 REV, cond-inhibition + reinstatement PASS; B T1 v1 ≥ .80, d2 ≥ .89, d3 ≥ .95; C cost ratio ≤ 2.5; D h1 ≥ 4/6, h2 ≥ 3/5 (one-shot).

## localized divergence terms ([0], deterministic, s = 0, `eval/trace3b.py`)
- d3 (T1, 2000 steps, N_p 200): the divergence is the member WEIGHT, not a member. On the independent-chain channels the weights converge to w_place = 1.000 / w_proc = 0.000 while the process member alone is better by +.0103 (ch2), +.0086 (ch3), +.0127 (ch4), +.0125 (ch5) nats/step on d3-ctrl (+.0068 ch0 on d3-and); the conjunctive ch6 is already better than decompw (−.0102). decompw's partition is the true one ([[0,1,6],[2,3],[4,5]]). Cause: the place member is scored at the SAMPLED place, which the outcome-dependent (y,z) draw selects to fit ⇒ regime/place-specific members beat the regime-independent proc member.
- reinstatement (T2, N_p 25): at the probe the regime posterior is .482 acquisition (US|A = .983) / .373 extinction (US|A = .000); the 4 unsignalled USs move it the wrong way (.482 → .468) because a US at A = 0 is a surprise under BOTH regimes; the US channel's dominant member is the set {(A,0)}, whose A = 1 cell is 30 reinforced / 30 extinct ⇒ .5.
- spont-recovery-sessions (T2, N_p 25): the interval hazard does relax the belief (place posterior long [.341,.170,.391,.099] vs short [.790,.179,.031,0]; sticky regime .996 → .911) but no live regime at the test predicts US|A > .491 (long [.101,.491,.110,.124]): the relaxation is applied to the CURRENT (home) place, so the re-drawn regime is a home regime — the unobserved box's acquisition regime keeps ~0 mass.

## decisions (SPEC-3b.2 [1], all in models/factor.py, one canonical config)
- Scale-separated place emission: g_j = P(tonic | the channel's own on→on counts), Beta-Binomial tonic Beta(9,1) vs phasic Beta(1,1) (no threshold, no channel named); the place emits j with prob g_j else .5 ⇒ lag/phasic channels leave the place prior ∪ likelihood (measured gate: ctx1 .90, home .85, A .0, US .0).
- Interval hazard in the regime prior only: h = 1 − (1 − h_z)^Δτ, Δτ = steps since the observation last changed (a deterministic f of the past) ⇒ gaps relax the regime towards the hierarchical CRP draw instead of spawning a cause.
- Hierarchical θ prior: a regime a particle has never visited starts at that particle's pooled θ (+1 on the diagonal), not at an uninformed prior ⇒ w_X < 0 survives a regime/context switch.
- Rejected: Δτ off (S0 n = 1: renewal-ABC −.053 z −5.71 FUTILE, srs −.097) ⇒ Δτ kept; two-log member evidence and np.add.at pooled counts (both replaced by exact equivalents, ε below).

## metrics
| item | value | basis (h_tree / n) | status |
|---|---|---|---|
| ε: vectorized vs de-vectorized (tests/data/factor_ref.py) | ≤ 1e-12 at N_p = 200 AND N_p = 25 (3 configs) | faba3f2bd496, tests 67 OK | ✓ |
| k_vec = T_ref(200)/T_vec(200) | 12.04/8.18 = 1.47 (was 1.38) | savings/one/0/v1, CPU s | ✗ < 1.80 |
| k_total(N_p' = 25) | 12.04/3.98 = 3.02 | same | ✗ < 3.50 |
| remaining hot path | Decomposer+OnlineHMM = 2.29 s = 28% of T_vec(200) (T_vec without −P = 5.89); 110386 `_lik` + 15700 `_learn_oldest` + 110372 `T()` calls per 450-step row | cProfile | open |
| S₀→S₁ screen, z (n = 4, N_p 25, c_s −3.733) | block 11.5, c-inh 0.78, cont 8.1, pree 1.8, rein **+1.60**, ABA 27.5, ABC 24.6, sav 5.2, srs **−5.11**, spont 2.8 | f0c6035b59c0 | FUTILE (srs) |
| same, baseline before [1] | rein −1.63, srs −3.20, c-inh 2.5, spont 0.8 | 98008dfd0fd6 | — |
| reinstatement defect (D3) | closed on S₁: mean −.028 → +.027, z −1.63 → +1.60 | n = 4 | ✓ |
| srs / cond-inhibition (D3) | srs −.110 → −.175 (z −5.11), c-inh 2.5 → 0.78 (< 1.5) | n = 4 | ✗ |
| A S₂ / B / C / D | not run: S₁ did not pass (freeze needs ∀m z_m > c_s) and T̂(A S₂) = 272 × 8.18/2 ≈ 1113 s > budget([2]) 800 s | — | open |
| C T_step(s-4x3)/T_step(s-2x3) | 1.64 (unchanged: [1] touches no timing path) | 1 seed × 600 steps, N200 | ✓ |

## open
- [0] k_vec/k_total: batching the ~35 concurrent equal-K HMMs of the Decomposer into one tensor is worth ≈ 1.5 s (⇒ k_vec ≈ 1.8, k_total ≈ 7); not attempted — it is a rewrite of decompose.py's group/candidate loops and W left no room to re-screen after it.
- [1] srs: the interval relaxation must be indexed by the place being RE-ENTERED (the unobserved box), not by the current place; cond-inhibition lost z (2.5 → 0.78) under the gate — the place factor now sees only tonic channels, so BX/BY differ less.
- [1].2 parent recruitment by Cov(x_i, y − ŷ_BMA) not implemented: the canonical −C screening is still the binned conditional form.
- D (h1/h2) kept unrun ⇒ still held-out.

## next
1. Place-indexed interval relaxation (per-place Δτ_y = t − last step y was active) → re-screen S₀→S₁ at N_p 25 before any S₂.
2. Batch the equal-K HMM population; ε-test against tests/data/factor_ref.py after it (both sides share the Decomposer, so ε stays meaningful).
3. Unbiased member evidence for the place member (expected log loss under q(y) instead of the sampled y) → d3 ≥ .95.
