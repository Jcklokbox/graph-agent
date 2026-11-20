# STATE (SSOT) — read first after any compaction
Only .md in the active tree; history in archive/ (unchanged, excluded via .ignore, not importable).
Env: `pip install --target /tmp/graph-agent-deps numpy pytest`; run from development/ with `PYTHONPATH=/tmp/graph-agent-deps:.`.
## goal
One generative agent (M_factor = `factor-L4-P-C`, models/factor.py CANONICAL) reproducing T2 and transferring on T1 at cost <= 2.5.
Gate 3b-1: A T2 v1 >= 9/10, 0 REV, cond-inhibition + reinstatement PASS; B T1 v1 >= .80, d2 >= .89, d3 >= .95; C cost ratio <= 2.5; D h1 >= 4/6, h2 >= 3/5 (one-shot).

## params (canonical, one config; rejected branches = empty set)
factor-L4-P-C: lags 0..4, own_lags 0..2, k=5 slots, N_p=200, K=KP=10, C_REG=C_GLOB=1, beta=alpha=1, noise=.25, s0=1, SWAP=log 3, QBINS=5.
Hierarchy P(y|x^slow) . P(z|y,Dtau,x^lag,y) . P(theta|z,x^lag,y); place = sticky CRP over SLOW channels only (>=5 onsets, P(on->on)>.7, data-driven); theta is place-free (no double generation).

## decisions (+1-line rationale)
- Scale separation (place view over slow channels; place not a channel member): lags/fast channels fragmented the place (r4); theta|z only removes double generation of x.
- Regime = sticky CRP indexed by place (F-13): home/box share causes, so a gap cannot reset theta; place emits nothing into the channel average.
- Runs only via `harness.bg start --tree H` (immutable snapshot, nice 19, taskset, private cwd) + one `bg wait` per stage; runner chunks each stage (prefix, sum t_hat/procs <= 150 s, exit 3 = PARTIAL, >180 s fatal).
- Journal <(h_tree,seed,task), m, dt>, O_APPEND+fsync; resume only within the same h_tree (a wrong-tree run cannot mix).
- Gate: per-row Bonferroni floor c_s = -3.733 (t15, beta/20), native sigma_m; S0/S1 futility only, verdict at S2 N_p=200; PASS = mu - t.975*sd/sqrt(n) > 0 and mu >= .05.
- Exactness: optimizations keep ev layout/op order => bitwise equal; exp(ev) cached per regime; proc.predict deduplicated; OnlineHMM T/E cached; Sg in place; flat take/scatter for q1/q0 counts.
- Rejected: view -V, one-lag -O, ctx-only -X (D-36/D-38/IA REV); unknown factor flags raise; franchise regime prior kappa>0 and global (place-free) regime prior both tested and rejected (savings -0.58 / reinstatement -0.074).

## D3 divergence terms (localized, N=25 trace, seed 0/2)
- reinstatement: at the ctx1 test place = [.267,.008,.722] (reinst) vs [.962,.034,.005] (control) -> d(place mass on home) = .695; control never re-identifies ctx1. Home gap births regime 278->336 (58 spurious causes in 15 steps).
- spont-recovery-sessions: at the test regime = [0,.48,.52] (short) vs [0,.14,.85] (long) -> d(regime-2 mass) = .33 with reversed sign; npz[:,box,:] is frozen during the home phase, so Dr[box] has no Dtau drift term (the box's extinction cause is never unconfirmed).
- d3: per-channel LL gap vs the exact filter is uniform (.01-.045 on d3-and/d3-and-ctrl), no single-channel defect; process member already takes w=1 on 6/7 channels; residual is mixture/regime fragmentation (used=6 regimes, 323 births).

## metrics
| item | value | basis (h_tree / n) | status |
|---|---|---|---|
| eps: vectorized vs pre-S reference | max diff 0 (<=1e-12) | savings s0 N200 + unit test | OK |
| unit tests from snapshot | 67 OK | h_tree 98008dfd0fd6 | OK |
| k_vec = T_ref(200)/T_vec(200) | 12.59/7.60 = 1.66 | savings s0 CPU | X (<1.80) |
| k_total(N_p'=25) | 12.89/3.59 = 3.59 | same | OK (>=3.50) |
| T_vec N_p 25/50/100/200 s | 3.59 / 4.02 / 5.07 / 7.60 | same | -- |
| A S2 (N_p 200, 16 seeds) | 8/10 PASS, 0 REV; srs -.022 [-.181,+.137], rein -.016 [-.068,+.037] | journal /tmp/work/screen1.jsonl | X (<9/10) |
| A screen S1 (N_p'=25, n=4) | block .396, c-inh .476, cont .469, pree .187, ABA .928, ABC .246, sav .170, spont .200; rein -.004 (z -.24), srs -.097 (z -2.82) | screen25 journal | gate X (z>0 miss) |
| B T1 v1 GC / d2 GC / d3 GC | .87 / .89 / .90 | 2 seeds x 2000 steps | v1 OK, d2 OK, d3 X (<.95) |
| C T_step(s-4x3)/T_step(s-2x3) | 33.39/11.70 = 2.85 | 1 seed x 600 steps, N200, process_time | X (>2.5) |
| D h1/h2 | not run (freeze gate not passed) | -- | open |
| max T_stage / waits / polls | <=150 s / 1 / 0 | all staged runs | OK |

## open
- A/D3: reinstatement and srs not > 0 at N_p'=25; both need a Dtau term in Dr[box] (regime unconfirmation while away) without diluting US-alone evidence at a familiar place.
- B/d3: .90 < .95; try slot/pruning or member gating; decompw = .975 reference.
- C: 2.85 > 2.5; cost is _resample (3.4 s/300 steps) + _reset_pos/ascontiguousarray (3.1 s), not the place view.
- [0] k_vec 1.66 < 1.80: Decomposer/online_hmm still ~30% of T_step.

## next
1. Add the Dtau drift term to the place-conditioned regime prior (unconfirmation rate), re-screen S1 at N_p'=25.
2. Cut _resample/_reset_pos cost (lazy q1/q0 compaction; reset only on slot change at that position).
3. Re-run A S2, B, C in one tree; freeze; then D one-shot.
