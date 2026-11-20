# STATE (SSOT) — read first after any compaction
Only .md in the active tree. History (decisions.md, AR reports, design notes): archive/research/ (unchanged, excluded from search via .ignore and from imports).
Env: `pip install --target /tmp/graph-agent-deps numpy`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.

## goal
Agent = one generative model that reproduces the conditioning phenomena (T2) and transfers on T1 at cost ≤ 2.5.
Open slice 3b-1 (`models/factor.py`) acceptance: GC d2/d3 within .02 of decompc; T2 v1 ≥ 9/10; cost ≤ 2.5. Not accepted (d3 −.043, T2 v1 7/10 best).

## decisions
- Trials run only via `harness.bg start` (snapshot by h_tree, nice 19, taskset, private cwd) + one `harness.bg wait` per stage (≤170 s, ≤2000 B) — no polling, output bounded.
- Journal = JSONL ⟨k=(h_tree,seed,task), m⟩, one O_APPEND write + fsync per record; resume skips k — kill -9 loses ≤ the in-flight task, never duplicates.
- Stages S0={0} → S1={0..3} → S2={0..15}, seed-major; S0/S1 futility only, verdict only on S2 (95 % t-CI > 0, no correction) — gates never declare success, so α is not inflated.
- c_s = (−2.0, −1.5) on z = mean/(σ_m/√n): β_loss(K=10 rows) = .042, null α̂ = .0252 vs .0252 ungated (harness/sim.py, 2·10⁵ sims).
- σ_m = median inter-seed sd per row (factor*, lc-sub*, results/t2*.json), floor .02 — zero-σ rows otherwise give infinite z.
- Snapshot covers .py/.json/.txt (sigma.json is read by the code); results/ never in the snapshot.
- Probe = s=0 records of the main journal (same k), never recomputed; `--trace` writes `trace()` per probe into JOURNAL.trace.
- Dev mode `factor…-N<k>` (particles); OnlineHMM.T cached until ntr changes — bitwise identical (ε = 0 test).
- 18 dead-end modules and 22 .md → archive/ (git mv, contents untouched); active tree = code + harness + tests + STATE.md.

## hypotheses open
- τ_K noise floor unknown: seed-0 effects swing (spont-recovery N100 +.71 vs N200 +.12); measure τ_K(N200 seed a, N200 seed b) before trusting the dev ranking.
- Non-particle cost dominates factor (T(N25)/T(N200) = .37): next speed lever is `_members` (33 % self-time: fancy-index q1/q0 (kk,J,S,N)), not N_p.
- σ_m for factor-L4-P-C itself is borrowed from sibling agents (sigma.json); re-estimate from its own S1 runs.
- 3b-1 candidates (AR-5): judge factor-L4-P-C on S2; IA REV; d3; 3b-2.

## metrics
| item | value | n / basis | status |
|---|---|---|---|
| T(S0), factor-L4-P-C, T2 v1, 2 CPU | 125 s | 10 rows × seed 0 | O(1) min ✓ |
| kill -9 + resume: dup / lost | 0 / 0, metrics bitwise = uninterrupted | tests/test_harness.py | ✓ |
| working copy mutated during run | metrics bitwise equal | tests/test_harness.py | ✓ |
| probe ≡ s=0 of main run | bitwise | tests/test_harness.py | ✓ |
| null α̂ (gated) / α_nom / ε | .0252 / .025 / .001 (3 se) | 2·10⁵ sims | ✓ |
| β_loss per row / per 10-row trial | .0043 / .042 at μ80 = .75σ | 2·10⁵ sims | ✓ |
| profile top-3 self-time share | 55 % (_members 26, learn 17, ufunc.reduce 12) | savings, seed 0 | — |
| N_p 25 / 50 / 100 / 200: T_task s | 4.7 / 5.0 / 7.8 / 12.9 | savings, seed 0 | — |
| τ_K(dev, full) | .73 / .73 / .56 / 1 | C = 10 T2 v1 rows, seed 0 | — |
| chosen front point (τ* = .7, k* = 2) | N_p' = 25, k = 2.73 | dev = futility/ranking only | ✓ |
| ε-test cached T vs reference | max diff 0 (ε = 0) | factor-L4-P-C-N25, savings | ✓ |
| active files / .md / .cache executables | 46 (was 165; N* = 50) / 1 / 0 | development/ excl. results, + STATE.md | ✓ |
| per-stage context ΔL_ctx | ≤ 2000 B (bg wait cap) | harness/bg.py | ✓ |
| polls | 0 | one wait per stage | ✓ |

## next
1. τ_K noise floor (2 seeds × N200 S0, one bg stage) → keep or raise N_p'.
2. AR-5: `python3 -m harness.bg start ar5 -- python3 -m harness.staged t2 --agents factor-L4-P-C --set v1 --journal j.jsonl` (S0→S2 gated), then pair against decompc on T1 d2/d3.
3. Vectorize `_members` gather with an ε-test like TestVectorizedEquivalence.
