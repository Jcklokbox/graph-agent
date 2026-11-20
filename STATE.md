# STATE (SSOT) — read first after any compaction
Only .md in the active tree (history in archive/, not importable); env: `pip install --target /tmp/graph-agent-deps numpy`, from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.

## goal
One generative agent (M_factor = `factor-L4-P-C`, models/factor.py CANONICAL) reproducing T2 and transferring on T1 at cost ≤ 2.5.
Gate 3b-1: A T2 v1 ≥ 9/10, 0 REV, cond-inhibition + reinstatement PASS; B T1 v1 ≥ .80, d2 ≥ .89, d3 ≥ .95; C ratio ≤ 2.5; D h1 ≥ 4/6, h2 ≥ 3/5 (one-shot).

## params — one canonical config, no flags, no tables
N 200 (S₀/S₁ N_p' = 25), K = KP = 10, k = 5, lags 0..4, own_lags 0..2, A 2, beta/alpha 1, noise .25, s0 1, C_REG/GLOB 1, SWAP log 3.
[1] scale separation: place prior / place chain / place evidence use no lagged copy; regime + θ condition on lags.
[1] hierarchical θ: a regime first entered inherits the particle's mass-weighted mean θ over its used regimes (parent), prior cov s0·I — learned w_X < 0 survives the switch.
[1] gap hazard: h_eff = 1 − (1 − h)^Δτ, Δτ = run length of identical observations (1 on an informative step: model unchanged there).
[1] −C recruitment: log BF = (n/2)·log 1/(1 − ρ²), ρ = corr(x_i(t−τ), r_j), r_j = x_j(t+1) − q_j, q_j = the model's own BMA prediction of channel j.

## localized divergence terms ([0], deterministic trace(state) at s = 0, N_p = 200, harness.localize)
- `reinstatement|reinst` A .4375 vs lc-sub-add .1925, decompw .0903; slots[6] (US) = (A, lag 3) + (US, lag 0): X never recruited; 4/5 slots of channels 1..8 are ctx1 lags 1–4; w_rest .90 on A, w_sets .98 elsewhere.
- `spont-recovery-sessions` Δτ 14 (short) / 300 (long) at test, h_eff .040 / .35 (hz .003 / .03) ⇒ the gap moves regime mass, θ inherited ⇒ z +1.74 (was −3.20).
- `cond-inhibition` same term as reinstatement (X missing from slots[6]) ⇒ B·Y − B·X only partly reproduced ⇒ z +0.59 (was +2.5).
- `d3-and` (500 steps, s0) factor .5473 vs decompw .5500 (ratio .995); Δ on channels 0,1 = +.019/+.016 nats, w_sets .9995, slots = the partner chain at lags 0–3 ⇒ the 2-bit conditional table cannot carry the witness's 4-step-window/Joint-HMM belief (resolution of |S| ≤ 2).
- Equation: Δloss = [resolution of |S| ≤ 2 on a 2-bit key, lags 0..4] + [θ inheritance at regime birth] + [gap-hazard regime prior]. Batched/vectorized decompose + online_hmm: ε = 0 bitwise (tests/data/ref_models), no cost gain (k unchanged).

## gate S₀→S₁ only (N_p' 25, c_s −3.733, n = 4, tree 730fed4a6fd8, seeds 0–3)
| row | Δ_m | σ_m | z_m | n | S₁ |
| blocking | +.320 | .0746 | +8.58 | 4 | ✓ | cond-inhibition | +.089 | .3044 | +0.59 | 4 | ✗(<1.5) |
| contingency | +.502 | .1176 | +8.54 | 4 | ✓ | pree | +.222 | .0981 | +4.54 | 4 | ✓ |
| reinstatement | +.053 | .0339 | +3.14 | 4 | ✓ | renewal-ABA | +.491 | .0592 | +16.59 | 4 | ✓ |
| renewal-ABC | −.014 | .0093 | −2.95 | 4 | ✓(floor) | savings | +.195 | .0915 | +4.26 | 4 | ✓ |
| spont-recovery-sessions | +.060 | .0684 | +1.74 | 4 | ✓ | spontaneous-recovery | +.168 | .1909 | +1.76 | 4 | ✓ |
S₁ verdict: ∀m z > c_s ✓ (min −2.95), reinstatement > 0 ✓, srs > 0 ✓, cond-inhibition > 1.5 ✗ ⇒ **S₁ not passed ⇒ h_tree not frozen, no S₂ run, Suite D untouched (still held-out).**

## metrics — AC | measured | status
| [0] ε vec vs ref, fixed seed / k_vec ≥ 1.80 / k_total ≥ 3.50 | 1e-12 / 1.80 / 3.50 | 0 bitwise / 13.60/10.03 = 1.36 (log 1.38) / 13.60/3.73 = 3.65 (log 3.40) | ✓ / ✗ / ~ |
| [0] divergence terms localized | required | 4 rows above | ✓ |
| [1] one default config, tests from snapshot | green | 68 OK (tree 679dc9fe0716) | ✓ |
| [1] D3 on S₀/S₁ | reinstatement > 0, srs > 0, c-inh > 1.5 | +3.14, +1.74, +0.59 | 2/3 ✗ |
| [2] S₂ A/B/D (verdicts) | 16 seeds, N 200 | T̂ = 16×17×9.3/2 ≈ 1265 s > 800 s budget([2]); none run | ✗ |
| C T_step ratio | ≤ 2.5 | 1.64 (600 steps, N 200) | ✓ |
| T_stage / waits / polls | ≤ 180 s / 1 | 75 s max (S0), 1 wait/stage, 0 polls | ✓ |

## decisions (+ rationale)
- θ inheritance at regime birth, not a reset: AAB/AB transfer needs w_X < 0 to persist across contexts (D1).
- gap hazard from Δτ (stasis run length), not from a task-specific clock: an unchanged observation carries no information (COIN session prior), and Δτ = 1 leaves every informative step bitwise unchanged.
- residual (−C) screen: `-C` stays canonical; the conditional-residual variant (r against the recruited slot cell) was measured and rejected at S0 (savings z −6.23) ⇒ not kept.
- ε reference re-frozen to the post-[1] model (the pre-[1] code is the frozen tree 98008dfd0fd6); the [0] batching keeps a frozen pre-[0] copy (tests/data/ref_models) at ε = 0.
- rejected branches: ∅ beyond −V/−O/−X (deleted); unknown factor flags raise.

## open / next
1. cond-inhibition z +0.59 < 1.5 blocks freeze: X must be recruited onto the US channel without lag-copy crowd-out (4 ctx1 lag copies occupy 4/5 slots); next: screen candidates by residual *after* removing the slow context from the key, or a third slot per channel — one config, then re-screen S₀→S₁.
2. k_vec 1.36 < 1.80: batching gave no gain (ε = 0, 0.97×) ⇒ the remaining N-proportional cost is the per-particle table gather (2 × (kk,J·S,N) takes/step); S₂ T̂ stays ≈ 1265 s > budget([2]) until k_vec ≥ 1.8.
3. renewal-ABC z −2.95 (floor-passed, REV at S₂): θ inheritance trades ABC renewal for transfer; quantify at S₂ only.
