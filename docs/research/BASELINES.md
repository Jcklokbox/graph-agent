# Historical baselines and measurement inventory

Pre-AR4 snapshot; do not read unrun entries as failures. See [roadmap](ROADMAP.md) for live rules.

| T1 set | Units / purpose |
| --- | --- |
| v0 | delay2, delay2-noise2, delay3-sticky, delay4-p.8, two-delays, hmm-noisy, context-reversal, control; easy regression floor |
| v1 | hmm-noisy, hmm-1ch, hmm3, noisy-cue-delay3, count3, noisy-identity, context-hidden, two-hmm; hidden state beyond observation window |
| d1 | d-3x3, d-6x2, d-4x3, d-4x2-noise4, d-mixed; development |
| d2 | d2-5x2-weak, d2-3x4, d2-unequal, d2-two-cues; novel structure |
| d3 | d3-ctrl, d3-and, d3-and-ctrl; controlled process and linked emissions |
| d3h / dscale / dcouple | OR/XOR/two (spent D13/14); s-2x3…s-7x3 (cost); c-shared/c-driver (uninformative gap) |

| GC | v0 | v1 | d1 | d2 | d3 | dscale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| markov4 | .72 | .71 | .16 | −.10 | .70 | −.10 |
| kfact | — | .48 | .93 | .85 | .97 | .92 |
| v11 | .89 | .66 | — | — | — | — |
| v12 | .89 | .66 | .29 (3 seeds) | — | — | .28 (1 seed) |
| A′/v11_bma | .92 | .81 | — | — | — | — |
| decompw | — | .77 | .95 | .90 | .98 | .92 |
| v15 / v18 | — | .83 / .76 | — | — | — | — |

T2 observations: A, B, X, Y; ctx1–3, home, US, value. Probe P(US | cue+context) after 15 context-only steps. `v0`: blocking, CI, contingency, ABA, ABC, SR, SR-home, savings; `v1` replaces SR-home with sessions and adds reinstatement, PREE; `h1`: overshadow/LI/negative patterning/overexpectation/AAB/inhibitor alone; `h2`: contextual LI, sensory preconditioning, US pre-exposure, superconditioning, contextual reinstatement. At this snapshot v1/h1 yes counts: RW 3/2 (one REV each); configural 3/3 (two v1 REV); `lc-f-E` v1 7/10; v12 5/5 (two v1 REV); v13/v14/v15 v1 7/6/5; anonymous `lc-sub-add` N200 10/4; F11/F12 9/5; v18 10/4. RW/configural h2 2/1 prior to its candidate exposure. Overexpectation then passed only RW; D65 later changes model status. AAB not by the same mechanism in v12 and F12. Do not infer a procedure from the aggregate alone.

Historical time (hardware/context dependent): v12 DEV aut/dist 13.5/9.8 s vs v18 44.5/46.8 s (roughly ×3.8 overall); v12 DEV-16 of order one hour and v18 forecast 3.5–4 h; D27 resolved screening in ~49 min. Per T1 v1 episode CPU: v12/v15/v18 6.0/12.9/18.5 s; full timed 48 episodes of order 3/10/10 min (different process usage). T2 v1/16: ~4/18/51 min; witnesses N200/F11/F12 151/139/141 s. h1 v12 460 s (one process), v18 1427 s, witnesses 68–69 s. Inference only from s-2x3→s-4x3: v12 2.4→14.7 ms, v15 ×3.5, v18 local emitter 17.0→35.7 ms, decompw ×3 on ×3.5 processes. Distinct from accepted N200 whole-agent cost gates.

Old version disposition: v10/v11 baseline and fingerprints; hard/soft/context/cause v11 witnesses GC .48–.69 vs markov4 .71; A′ retained diagnostic mixture; v12a/b/c/e explored (v12c DEV .715, v12e=v12); v12d/v16 rejected local optimizations; v13/v14 patch failures; v17 quadratic superseded; F9 per-cause emission dead end, F11 intermediate, F12 separate place and regime. `lc-anon`/`lc-sub` antecedents retained as witnesses, not interchangeable with agents. Unresolved then: Proposition 8 anonymous reinforcement, v1 novel-home confound, DEV width and deep-exploration/asymmetric outcomes, need blind procedural benchmark.
