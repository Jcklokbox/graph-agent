# STATE (SSOT) — read first after any compaction
Only .md in the active tree; history in archive/ (unchanged, excluded via .ignore, not importable).
Env: `pip install --target /tmp/graph-agent-deps numpy`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.

## goal
One generative agent (M_factor = `factor-L4-P-C`, models/factor.py CANONICAL) reproducing T2 and transferring on T1 at cost ≤ 2.5.
Gate 3b-1: A T2 v1 ≥ 9/10, 0 REV, cond-inhibition + reinstatement PASS; B T1 v1 ≥ .80, d2 ≥ .89, d3 ≥ .95; C cost ratio ≤ 2.5; D h1 ≥ 4/6, h2 ≥ 3/5 (one-shot).

## decisions
- Runs only via `harness.bg start [--tree H]` (immutable snapshot, nice 19, taskset, private cwd) + one `bg wait` per stage.
- Runner chunks every stage: prefix of tasks with Σ t̂/procs ≤ 150 s, exit 3 = PARTIAL (rerun resumes); >180 s ⇒ fatal abort. t̂ = q90 of logged `dt` per task, else per agent, else 1.6·(2 + .04 N_p).
- Journal ⟨(h_tree,seed,task), m, dt⟩, O_APPEND+fsync; resume only within the same h_tree (`--tree` reuses the snapshot) — a wrong-tree run cannot mix (6 records of a killed wrong-tree run are keyed apart).
- Gate: per-row Bonferroni floor c_s = −3.733 (t₁₅, β/20), native σ_m (factor-L4-P-O, 16 seeds); S0/S1 futility only (`--screen`), verdict only at S2, N_p = 200.
- Rejected branches deleted (view −V, one-lag −O, ctx-only −X: D-36/D-38/IA REV); unknown factor flags now raise (silent fallback was a real hazard).
- Exactness rule: optimizations keep ev layout and op order ⇒ bitwise equal; exp(ev) cached per regime, proc.predict deduplicated, OnlineHMM T/E cached, Sg in place.
- Dev fidelity: C = {factor, factor-L4-P-V, factor-L4-P-O} (16-seed logs) on discriminating rows only: τ_K^ceil(held-out seeds 4–9 vs 10–15) ≥ τ_disc = .5 ⇒ blocking, reinstatement, renewal-ABA (all 1.0); S1 seeds 0–3 untouched by selection.

## hypotheses open
- [0] k_vec 1.38 < 1.8, k_total 3.40 < 3.5: the rest is the Decomposer/online-HMM population (~30 %, N_p-independent, Python loops over small HMMs) → batch HMMs of equal K into one tensor (bitwise needs same reduction order).
- [0] τ(100) = .556 < τ(25) = τ(50) = .778: 3 configs × 3 rows ⇒ τ steps of 1/3 per row; one swap; needs |C| ≥ 5 with logs on discriminating rows.
- [1] not done (W): place/regime scale separation (lags out of place likelihood), hierarchical θ prior keeping w_X<0 across contexts, residual-covariance parent recruitment, d3 localization via trace (d3 .932 vs decompw .975).
- [2] not run (T̂ > W − t − reserve): A at S2 (T̂ ≈ 16 seeds × 17 tasks × 9.3 s / 2 ≈ 1270 s), B (T1 v1/d2/d3 at 6 seeds ≈ 2000 s+), D one-shot (kept unrun ⇒ still held-out).
- A risk: S1 reinstatement −.028 (z −1.63), spont-recovery-sessions −.110 (z −3.20) at N_p 25 → A likely < 9/10.

## metrics
| item | value | basis (h_tree / n) | status |
|---|---|---|---|
| ε: vectorized + pruned factor vs pre-Σ reference | max diff 0 (≤1e-12) | savings s0 N200 + unit test 3 configs | ✓ |
| k_vec = T_ref(200)/T_vec(200) | 12.89/9.32 = 1.38 | savings s0 CPU, 98008dfd0fd6 | ✗ |
| k_total(N_p' = 25) | 12.89/3.79 = 3.40 | same | ✗ |
| T_vec N_p 25/50/100/200 s | 3.79 / 4.07 / 5.79 / 9.32 | same | — |
| max T_stage / waits per stage / polls | 127 s / 1 / 0 (11 stages) | results/runs/{b2-*,gateA-*,suiteC,tests3b2,timing3b2} | ✓ |
| C_degenerate | 0 (3/3 rows τ^ceil_heldout = 1.0) | 3 configs × 16 seeds | ✓ |
| τ_K^ceil(S1) / τ* = min(.85, ceil − .05) | .778 / .728 | seeds {0,1} vs {2,3} | — |
| τ_K(dev, 200) N_p 25/50/100 | .778 / .778 / .556 | 3 configs × 3 rows × 4 seeds, d3b9dc578f7f | N_p' = 25 ✓, monotone ✗ |
| max per-row |Δy| N_p 25/50/100 | .102 / .084 / .110 | same | — |
| canonical config / rejected branches | factor-L4-P-C / ∅ (−V −O −X removed) | 98008dfd0fd6 (frozen) | ✓ |
| d3 discrepancy localized | no | — | ✗ |
| A screen S0→S1 (N_p 25, c_s −3.733) | pass; z: block 11.2, c-inh 2.5, cont 8.2, pree 3.3, rein −1.6, ABA 32.6, ABC 42.2, sav 4.5, srs −3.2, spont .8 | n = 4, 98008dfd0fd6 | screen ✓, verdict open |
| A S2 / B / D | not run | T̂ > budget | open |
| C T_step(s-4x3)/T_step(s-2x3) | 1.64 (45.9/27.9 s; ≤ 2.3 if the shared core is charged) | 1 seed × 600 steps, N200 | ✓ |
| unit tests from frozen snapshot | 67 OK | 98008dfd0fd6 | ✓ |

## next
1. A S2: `bg start gateA-s2 --tree 98008dfd0fd6 -- python3 -m harness.staged t2 --agents factor-L4-P-C --set v1 --journal j.jsonl` (~9 chunked stages).
2. Batch the Decomposer HMMs (k_vec, k_total), with the ε-test against tests/data/factor_ref.py.
3. [1] model items in a new tree; re-freeze before any S2 verdict; D only after that.
