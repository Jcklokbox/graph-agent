# Development workspace

The executable agent, the environments, the evaluation harnesses and the tests. The model is
described in [`../research/model.md`](../research/model.md); the goal, environment family and
evidence tiers in [`../research/goal.md`](../research/goal.md); every decision with its
pre-registered prediction and outcome in [`../research/decisions.md`](../research/decisions.md).

## Layout

```
agents/   the agents; agents.load(name) -> module
  v11.py          PosteriorGraph (v11), standard library only
  v11_latent.py   slice-1 prototype: v11 reading an online HMM's state as one-hot channels
                  (_soft, _ctx: soft read-out variants)
  v11_bma.py      slice-1 A': per-channel Bayesian model averaging of v11 and the HMM
  v11_cause.py    slice-1 B: latent causes with v11 as emission model (rejected, S1-3)
  v12.py          step 2: calibrated core (context-tree mixture over depths, learned
                  generalisation from the parent) + value-of-information exploration
                  (S2-0..S2-5); prototypes v12a-e removed, bit-exact with v12e
  v13.py          slice 2: v12 + one inference over contexts (Page-CUSUM births with a
                  hierarchical prior, belief with a sticky prior, mixture predictions)
  v14.py          slice 2 attempt N-2 (learned hazard, spike-and-slab challenger); rejected
  v15.py          slice 3 C0: v12 + the decomposition witness, per-channel posterior mixture,
                  planner on the mixture (D-11; T1 v1 .83, T2 / DEV = v12, cost x2)
  v16.py          v15 + pruning of v12's context to the witness group after 1000 steps
                  (D-12; mechanism check negative, kept as record)
  v12d.py         v12 with an event-driven attention test (D-15; -5 % only, kept as record)
  v17.py          slice 3b-0: v12 + anonymous latent-cause predictor per channel
                  (LatentCauseSubAdd, 200 particles), per-channel posterior mixture as in v15
                  (D-20; cost quadratic in channels — interim)
  v18.py          slice 3b-1 candidates: v17 with 4 candidate slots per channel chosen by
                  pairwise sequential evidence (D-21 prep; cost linear in channels)
envs/     environments
  tasks.py        behavioural tasks (blind gate, cancellation, distant gate), variation
                  wrappers, oracles, VARIANTS and SPLITS
  family.py       FHP family: factored hidden-process worlds with an exact Bayes filter;
                  T1 probe sets t1_worlds() (v0, floor) and t1_worlds_v1() (v1, hidden state);
                  slice-3 sets (drafts, D-0..D-7): d1 (decomposition), d2 (held-out
                  structures), dscale (2-7 processes), dcouple (coupled transitions),
                  d3 (action-dependent lever, shared-emission "AND" channels via World.links),
                  d3h (held-out conjunctions: OR, XOR, two links; D-13)
  phenomena.py    T2: conditioning phenomena translated into channel streams; set v0 (8 rows,
                  frozen) and v1 (10 rows, audit fixes + reinstatement, PREE; W-1);
                  held-out set h1 (6 rows: overshadowing, latent inhibition, negative
                  patterning, overexpectation, renewal AAB, inhibitor-alone; D-23, frozen;
                  used for development since D-28); held-out set h2 (5 rows: LI-context,
                  sensory preconditioning, US pre-exposure, superconditioning,
                  reinstatement-context; D-30, frozen, no candidate run on it yet)
models/   component models studied in isolation
  online_hmm.py   online HMM over binary channels (conjugate counts, fixed-lag smoothing);
                  T1 name hmm<K>-L<lag>[-d<decay>]
  context_tree.py context-tree weighting along one path (exact mixture over depths)
  latent_cause.py T2 feasibility witness: ideal observer of latent contexts (sticky CRP,
                  particle filter, exact enumeration for tests); options learned hazard,
                  hierarchical feature prior, home steps not emitted (W-0, W-1)
  latent_cause_lin.py  anonymous T2 witnesses (D-16..D-18): LatentCauseLin (additive model per
                  cause), LatentCauseSub (sticky-CRP causes x Bayesian average over parent
                  sets <= 2), LatentCauseSubAdd (+ the additive model as a member; T2 v1 10/10);
                  T2 names lc-anon, lc-lin[-x], lc-sub, lc-sub-add[-n<particles>], lc-sub-add-M (F-11, Markov context per cause), lc-sub-add-P (F-12, place x regime)
                  (-e / -E: emission per cause, flat / hierarchical prior; F-9, not adopted)
  decompose.py    slice-3 feasibility witness: online Bayesian partition of the channels into
                  independent groups, each a posterior mixture over model families (online
                  HMMs with K in {2,3,4,6}, windows); T1 names decomp / decompw / decompf /
                  decompr (fast merge tests, D-9/D-10, not calibrated) / decompc (conjunctive
                  parent pairs per channel, D-13/D-14)
eval/     harnesses
  benchmark.py    T3: behavioural transfer benchmark (score = (brain-stay)/(oracle-stay))
  predictive.py   T1: prediction vs the exact filter (gap closed, GC); structure oracles
                  kfact (true partition and K) and kjoint (one HMM with the joint size)
  phenomena.py    T2: phenomena (--set v0|v1); references rw, cfg (configuration counts),
                  lc* (latent-cause witness), decomp* (slice-3 witness adapter)
  t2_relative.py  T2 effects as a fraction of the witness lc-f-E, share of seeds with the sign
  screen.py       T3 sequential paired screen vs cached reference (stops when the answer
                  is known; --budget-min; --replay re-applies the rules to a finished run)
  early.py        T1 calibration with few data: excess loss in the first 1/8 of episodes
  diagnose.py     T3 loss decomposition (model vs control) on delayed-food variants
  stats.py        seed-level statistics: t-intervals, paired/unpaired, MDE, n needed
  fingerprint.py  bit-exact behavioural fingerprint; fingerprint_v11.txt is the reference
tests/    test_v11.py (31), test_family.py (7, filter vs brute force, slice-3 sets and
          oracles), test_context_tree.py (1, recursion vs brute-force mixture),
          test_screen.py (3, stopping rules on synthetic replays), test_v12.py (2),
          test_v13.py (2), test_latent_cause.py (1, filter vs exact enumeration),
          test_decompose.py (1, merges one process, keeps a distractor apart),
          test_v15.py (2, v15 = v12 when the witness has no weight; conjunctive candidates),
          test_latent_cause_lin.py (5, blocking, inhibition needs the additive member, savings, F-11 novel context renews, F-12 novel place renews),
          test_v17.py (2, v17 = v12 when the cause predictor has no weight; probabilities),
          test_v18.py (1, valid slots; screening ranks the true partner first),
          test_phenomena_h1.py (3, h1 / h2 translations checked with rw and cfg)
results/  JSON outputs of the runs logged in decisions.md
```

### Agent interface

An agent module exposes a class `Agent` (or `PosteriorGraph`) constructed as
`Agent(n_inputs, n_actions, value_index, seed=...)` with `step(obs) -> action` followed by
`learn(next_obs)`. T1 and T2 additionally need `predict_next(obs, action)` → per-channel
probabilities. `--agent` / `--agents` accept a registered name (`v11`) or a path to a `.py`
file (e.g. `../archive/development/posterior_graph_v10.py`).

## Running

From this directory; NumPy is needed by everything except `agents/v11.py`.

```bash
python -m unittest discover -s tests -t .                     # 61 tests, ~70 s
python -m eval.fingerprint --check eval/fingerprint_v11.txt   # ~5 min on 2 CPUs

python -m eval.predictive --agents markov1 v11 --out results/t1.json   # T1 v1, ~2 min
python -m eval.predictive --set v0 --agents v11 hmm8-L4                 # T1 v0 floor, HMM probe
python -m eval.phenomena --agents rw v11 --out results/t2.json         # T2 v0, ~4 min
python -m eval.phenomena --set v1 --agents cfg lc-f-E v12              # T2 v1 with references
python -m eval.phenomena --set h1 --agents rw cfg v12                   # held-out T2 set h1
python eval/scripts/probe_t2.py v18 blocking,savings 0 --set v1         # one-seed probes
python -m eval.t2_relative results/t2v1_v12_v13.json results/t2v1_v14.json   # vs the witness
python -m eval.predictive --set d1 --agents decompw kfact kjoint --steps 4000  # slice 3, ~10 min
python -m eval.screen --agent NAME --budget-min 15 --out results/scr.json  # T3 decision (default)
python -m eval.benchmark --dev --out results/dev.json                  # T3 fixed-length, ~15 min (8 seeds)
python -m eval.benchmark --dev --seeds 16 --agent PATH.py --out ...    # more seeds / other agent
python -m eval.stats results/a.json [results/b.json]                   # CIs, paired difference
```

Episodes run in a process pool (`PG_PROCS=<n>`; `PG_PROCS=1` is serial, output identical).
Benchmark options: `--set NAME=json` sets a module attribute of the agent, `--kw JSON`
passes constructor arguments; listing variant names restricts the run to them.

## Protocol

Experiment economy (decision P-1 in `research/decisions.md`):
* Mechanism first: check the quantity a change targets on 1-3 episodes before any statistics.
* Decisive test first, in its cheapest form: `eval.screen` against the cached reference;
  T1/T2 only to diagnose a screen result or for components that are not yet an agent.
* Before every run: an outcome -> action table; if two outcomes lead to the same action, the
  run is not made. The minute budget is passed as `--budget-min` and never extended.
* Sequential, not fixed-length; baselines from cache; budget in minutes per step (default 30).

* Every run that informs a decision is logged in `research/decisions.md` **before** it is
  made, with a quantitative prediction; the outcome is appended afterwards.
* Configurations are chosen on DEV / T1 / T2 only. The fixed held-out splits `--held` …
  `--held6` are all spent (each was run once); the next transfer test needs a procedural
  generator.
* Rejected experiments are removed from the code and recorded in `research/`, not kept
  behind flags.
* Refactors that must not change behaviour are verified with the fingerprint.
