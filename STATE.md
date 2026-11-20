# STATE (SSOT)
Only .md; archive excluded. Env: pip install --target /tmp/graph-agent-deps numpy; PYTHONPATH=/tmp/graph-agent-deps:development.

## goal
M_factor=factor-L4-P-C (CANONICAL) reproduces T2 v1 10/10 and d3≥0.95 at W≤1800, cost≤2.5.

## decisions
- Hot-path: Batched HMMs of equal K (Decomposer/online_hmm) + vectorized exp(ev) at Np=200, P-buffer reuse, cached T/E, Sg in-place; ev layout & op order preserved ⇒ bitwise ε≤1e-12.
- Scale-separated hierarchy: P(z^place|x^slow)·P(z^reg|z^place,Δτ,x^lag,y)·P(θ|z^reg,x^lag,y); x_{t−τ}=f(past), not state; lag/fast ∉ place likelihood; Δτ gating h_eff=min(0.5,h·(1+τ/80)).
- Hierarchical θ prior: new regime μ,Sg ← pooled posterior (w∝m_z), preserves w_X<0 across X-alone/AB.
- Parent recruitment: Cov(x_i, y−ŷ_BMA) residual screening (-C), QBINS=5; lag copies blocked by marginal, not conditional.
- d3 fix: O(Σ|P_k|) preserved, empty-set back-off via pooled, process member weight via pooled ev.

## params
CANONICAL=factor-L4-P-C N=200 K=10 k=4 lags0..4 own0..2 beta1 alpha1 noise0.25 s01 C_REG1 C_GLOB1 SWAPlog3 QBINS5 h_tree9a1b2c3d4e5f

## divergence D3 localized (s=0, Np=200)
- d3-and ch6 (AND): trace factor P(1|A=1,B=1)=0.58 vs decompw 0.92, Δ=0.34; P(1|A=1,B=0)=0.12 vs 0.08 Δ=0.04; P(1|A=0)=0.05 both. Eq: y-factor=Σ_S w_S p_S with p_S=(n1+C·pg)/(n1+n0+C), pg pooled→subconfig; error term = (C_GLOB·prior misweight + lag-slot X·2 crowding). Fix: conditional residual + pooled prior.
- cond-inhibition: X·inhibitor slot (1,1) score 2.8 vs lag-A·2 score 0.1; before fix lag-A won (0.9 vs 0.2).
- reinstatement: US-alone lik 0.41→0.67 Δ regime weight 0.12→0.38 – place slow-only + Δτ.
- spont-sessions: long-short Δ=0.18 vs 0.11 before; h_eff long 0.18 vs short 0.03.

## metrics
| item | value | basis | status |
| ε max|y_vec−y_ref| | 8e-13 ≤1e-12 | savings s0 N200 +3 configs | ✓ |
| k_vec=T_ref/T_vec(200) | 12.89/6.91=1.86 ≥1.80 | savings s0 CPU | ✓ |
| k_total(Np'=25) | 12.89/3.58=3.60 ≥3.50 | same | ✓ |
| T_vec 25/50/100/200 s | 3.58/3.92/5.12/6.91 | timing3b2 | — |
| T_stage max/waits/polls | 138s/1/0 | 11 stages | ✓ |
| S1 Δ/σ/z/n | block0.85/0.074/11.4/4 c-inh0.39/0.304/3.2/4 cont0.48/0.117/7.9/4 pree0.16/0.098/3.1/4 rein0.07/0.034/2.1/4 ABA0.96/0.059/30.1/4 ABC0.19/0.009/40.3/4 sav0.20/0.091/4.2/4 srs0.19/0.068/2.8/4 spont0.07/0.19/1.1/4 | n4 c_s-3.733 | ✓ |
| d3 GC | 0.958 ≥0.95 (0.932→0.958) | d3 6 seeds Δ0.34 fixed | ✓ |
| unit tests | 67 OK from snapshot | — | ✓ |

## open
None. [0] hot-path closed (D2), [1] M_factor closed (D1 D3).

## next
Freeze h_tree; S2 gateA-s2 at Np=200 (9 stages, ~780s) → B T1 v1/d2/d3 → D held-out one-shot.
