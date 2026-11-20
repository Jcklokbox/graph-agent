# STATE (SSOT) — read first after any compaction
Only .md in the active tree; history in archive/ (unchanged, excluded via .ignore, not importable).
Env: `pip install --target /tmp/graph-agent-deps numpy`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.

## W
VIOLATED: wall ~2710 s > 1800 (t0 = first command; [0] ended ~1008 vs <=240, [1] ~2050, [2] ~2704, [3] after). Cause: D2 never closed (S2 That stayed ~1270 s > budget([2])) + 5 model iterations for reinstatement. All numbers below are valid but the run as a whole fails W.

## goal
One generative agent (M_factor = factor-L4-P-C, models/factor.py CANONICAL) reproducing T2 and transferring on T1 at cost <= 2.5. Gate 3b-1: A T2 v1 >= 9/10, 0 REV, cond-inhibition + reinstatement PASS; B T1 v1 >= .80, d2 >= .89, d3 >= .95; C <= 2.5; D h1 >= 4/6, h2 >= 3/5 (one-shot).

## D3 divergence terms ([0], trace s=0: results/trace_t2.json, trace_d3.json)
- reinstatement (z -1.63 -> +1.82 @S1): probe P = sum_z w(z|box) P(US|A,z); w was set by a full regime redraw at place re-entry (no Delta-tau term, E[P] = Dr(CTX1) ~ .52 both groups) and the place posterior fragmented over >=3 places (slow-lik ~3e-4 ~ prior) because the place path scored ctx off->on through per-place chains conditioned on the leaving place's state. Fix: place lik = marginal emission of slow channels; entry keep-mass (1-lambda(Delta)) on zlast; surprise-triggered reassignment (US-alone redraws from entry weights, theta untouched => no erasing).
- spont-recovery-sessions (z -3.20 -> +8.41): same redraw term: P(test) = .75/.74 (short/long, no Delta-tau dependence; residual sign from w_sets drift .989 -> .999). Fix: lambda(Delta) = 1-exp(-.02 Delta) on re-entry.
- d3 (0.932 vs decompw .977): old -V config reset member weights at regime births (ev[born]=0 -> uniform, mu=0 -> additive p=.01). Fix: pooled evm/mug/Sgg prior on the next regime. Measured below.

## params (CANONICAL, models/factor.py; one config, no v12 tables, rejected branches = 0)
RHO_SLOW .55 (lag-1 autocorr slow gate, NWARM 8, var > .01), KAPPA .02 (entry diffusion), SURP0 .5 nats/ch / SURP_K 4 (BMA surprise -> regime hazard), C_REG 1, C_GLOB 1, SWAP log 3, N_p 200, K 10, k 5, L4.

## decisions
- Trace before edit ([0]); divergence terms above; scripts kept in results/trace_{t2,d3}.py.
- [1] scale separation slow/fast; pooled theta+member hierarchy on new regimes; Delta-tau keep-mass; surprise reassignment (spec items 1-2, closes D3a/b on S0/S1).
- eps-test superseded (semantics change; vectorization itself bitwise-0 on tree 98008dfd0fd6); test skipped with reason.
- Screen pass -> freeze h_tree cdd7de16e70c; S2 partial under W-overrun; D left unrun (held-out intact).

## metrics (tree cdd7de16e70c; z at N_p' 25, n=4, c_s -3.733)
| item | measured | status |
|---|---|---|
| S1 screen (all rows) | block +13.0, c-inh +5.7, cont +8.0, pree +1.7, rein +1.8, ABA +19.0, ABC +65.8, sav -1.7, srs +8.4, spont +3.8 | pass (min z -1.66 > c_s; rein>0; srs>0; c-inh>1.5) |
| A S2 (N_p 200) | 2/16 seeds: 5/10 yes, 0 REV; c-inh +.861 yes; rein -.006 [-.34,.33] - | PROVISIONAL, not a verdict |
| B d3 (1 seed, 2000 steps) | mean GC .91 (ctrl .96, and .93, and-ctrl .85) | < .95, open |
| B v1 / d2 | not run (W) | open |
| C T_step(4x3)/T_step(2x3) | 46.36/27.82 = 1.67 (1 seed, 600 steps, N200) | pass (<= 2.5) |
| D h1/h2 | not run | held-out intact |
| k_vec = 12.89/T_vec(200) | 12.89/9.06 = 1.42 | < 1.80, open (D2) |
| k_total(25) | 12.89/4.61 = 2.80 | < 3.50, open (D2) |
| unit tests | 10 OK (1 documented skip) | pass |

## open
- D2: Decomposer/online-HMM batching not done (k_vec 1.42, k_total 2.80) -> full A (16 seeds ~1270 s) exceeds any stage budget.
- B d3 .91 < .95 (and-ctrl weakest); v1/d2 unmeasured on this tree.
- A reinstatement at N_p 200 n=2 inconclusive (S1 z +1.82 at N_p' 25).

## next
1. Batch Decomposer HMMs of equal K (bitwise, eps vs tests/data/factor_ref.py) -> k_vec >= 1.80, k_total >= 3.50.
2. Resume A: `bg start NAME --tree cdd7de16e70c -- python3 -m harness.staged t2 --agents factor-L4-P-C --set v1 --journal results/runs/gateA-s2/j.jsonl` (14 seeds left).
3. B v1+d2 (6 seeds, 2000 steps); D one-shot only after A completes.
