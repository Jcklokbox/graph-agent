# STATE (SSOT) — read first after any compaction
Only .md in the active tree; history in archive/ (unchanged, excluded via .ignore, not importable).
Env: `pip install --target /tmp/graph-agent-deps numpy`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.

## goal
One generative agent reproducing T2 conditioning phenomena and transferring on T1 at cost ≤ 2.5.
Slice 3b-1 (`models/factor.py`) open: GC d2/d3 within .02 of decompc; T2 v1 ≥ 9/10; cost ≤ 2.5 (best: d3 −.043, T2 v1 7/10).

## decisions
- Runs only via `harness.bg start` (snapshot by h_tree, nice 19, taskset, private cwd) + one `bg wait` per stage (≤170 s, ≤2000 B).
- Journal JSONL ⟨k=(h_tree,seed,task), m⟩, O_APPEND+fsync per record; resume skips k — kill -9: 0 dup, 0 loss.
- Stages S0={0} → S1={0..3} → S2={0..15}; gates prune only; verdict on S2 (95 % t-CI > 0) — α not inflated.
- Gate = per-row Bonferroni floor c = t₁₅⁻¹(.02/20) = −3.733 on z = mean/(σ_m/√n), any row ⇒ prune (old conjunctive c gave β_FWER .574).
- σ_m native: factor-L4-P-O, 16 logged seeds (logs reproduced bitwise by this tree, harness/repro.py); harness/sigma.json.
- factor.py vectorized: flat `take` gather of q1/q0, preallocated P, one scatter for all channels; ev layout kept (sum order) ⇒ bitwise = reference.
- Dev fidelity for S0/S1 screening: agent suffix `-N25` (N_p' = 25); verdicts only at N_p = 200.
- C = 6 configs with 16-seed logs (factor, -V, -L4, -P, -L4-P-V, -L4-P-O) × 3 rows logged for all (cond-inhibition, savings, spont-recovery-sessions).

## hypotheses open
- k_vec 1.56 < 1.8: remaining cost is the Decomposer/online-HMM process member (~30 %, N_p-independent) and ev/exp in `_members`; next: cache exp(ev) per regime, batch OnlineHMM updates.
- τ_K^ceil = .20 on 3 rows × 2 seeds: C is near-tied on these rows; ranking C needs ≥ 8 seeds or all 10 rows.
- 3b-1 next (AR-5): judge factor-L4-P-C on S2 with the new gate; IA REV; d3; 3b-2.

## metrics
| item | value | basis | status |
|---|---|---|---|
| ε-test vectorized vs reference factor.py | max diff 0 (≤1e-12) | S0: 17 tasks T2 v1 + 3 unit configs | ✓ |
| k_vec = T_ref(200)/T_vec(200) | 1.56 (S0 T2 v1 wall 125→80 s); 1.40 (savings CPU 12.89→9.23 s) | factor-L4-P-C | ✗ (<1.8) |
| `_members` share of T_step | 38 % → 25 % (1.5×) | cProfile savings | ✗ (<2.5×) |
| unit tests from snapshot | 69 OK | unittest discover | ✓ |
| native σ_m (factor-L4-P-O) | block .075, c-inh .304, cont .118, pree .098, rein .034, ren-ABA .059, ren-ABC .009, sav .092, srs .068, spont .191 | 16 seeds | ✓ 0 borrowed |
| β_FWER (10 rows, all Δ=0) | .0189 Gauss+σ̂ error / .0020 σ known / .0010 bootstrap | 2·10⁵ sims, harness/fwer.json | ✓ ≤ .02 |
| α_final per row | .0250 (gates prune only) | 2·10⁵ sims | ✓ ≤ α_nom |
| τ_K^ceil (seeds {0,1} vs {2,3}, N_p 200) | .20 | C, 3 rows | — |
| N_p 25 / 50 / 100 / 200: T_vec s | 4.08 / 4.47 / 6.27 / 9.23 | savings CPU, seed 0 | — |
| k_total = T_ref(200)/T_vec(N_p) | 3.16 / 2.88 / 2.06 / 1.40 | T_ref 12.89 s | ✗ (<3.5 at N_p' = 25) |
| τ_K(rank C at N_p, at 200), S1 | .60 / .60 / .73 / 1 (monotone) | 6 configs × 4 seeds × 3 rows | ✓ |
| max per-row |Δy| vs 200 (S1 mean) | .39 / .45 / .30 / 0 | same | — |
| N_p' = min{τ ≥ min(.85, τ^ceil−.05) = .15, monotone} | 25 | — | ✓ locked |
| stage ≤180 s / waits per stage ≤1 | N100-b 368 s; 2 stages needed 2 waits | results/runs/s2-* | ✗ |

## next
1. Speed: cache exp(ev)/Z per regime; vectorize Decomposer/OnlineHMM across groups (ε-test against tests/data/factor_ref.py).
2. Raise τ^ceil: C on all 10 rows (factor-L4-P-* only) × 8 seeds, then re-select N_p'.
3. AR-5: `python3 -m harness.bg start ar5 -- python3 -m harness.staged t2 --agents factor-L4-P-C --set v1 --journal j.jsonl`.
