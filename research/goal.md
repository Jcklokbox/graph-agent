# Goal, environment family and evidence tiers

*Draft of slice 0 (2026-11-19). Needs the user's agreement before slice 1 starts; edits after
agreement are logged in `decisions.md`.*

## 1. Long-term goal

A general adaptive agent that, without being told the task, learns online from anonymous
binary channels and one value channel what the world is like and how to act in it, and
transfers honestly to worlds it was never tuned on. The model should be one coherent
mathematical object (a learner whose every part can be read as inference in a stated
generative model) and biologically meaningful (its behaviour should reproduce, or knowingly
deviate from, classical findings of animal learning).

Long-term ladder (direction, not a plan): C0 associations and timing → C1 hidden state →
C2 independent processes (objects) → C3 concepts as latent types with instances → C4
hierarchy of types → C5 learning from other agents. Only C0–C2 are planned (slices 0–3).

## 2. Central question for slices 1–3

> How can an agent, online and without labels, learn from anonymous channels (a) a
> decomposition of the world into independent processes and (b) the hidden states of each
> process, so that its cost grows with the **number** of processes, not with the **product**
> of their state spaces?

Why this question: everything above C1 on the ladder (objects, concepts as types with
instances, hierarchy, other agents as processes) presupposes (a) and (b). v11 has neither: it
has no belief over hidden states (measured: GC on `hmm-noisy` = Markov level, `decisions.md`
S0-2) and one global model that is overwritten instead of contexts that are inferred
(measured: savings reversed, S0-3).

## 3. The environment family (FHP — factored hidden-process worlds)

Implemented in `development/envs/family.py`.

A world is a set of independent **processes**. Process *k* has a finite hidden state
*s_k*, a transition *P(s_k' | s_k, a)* that may depend on the action, and emits a fixed subset
of the binary channels through *P(x_j | s_k)*. Channels of different processes are disjoint;
one channel is the value channel (reward). The agent sees only the channel vector; it is not
told the number of processes, their state spaces, which channels belong together, or which
channel is value-relevant.

Properties that make the family a good yardstick:

* **Exact reference.** For small worlds (joint state space ≤ 5000) the Bayes filter on the
  joint state is exact (`Filter`), so prediction can be scored as the fraction of the gap
  between base rates and the optimum that the agent closes (GC, T1). The filter is checked
  against brute-force enumeration in `tests/test_family.py`.
* **Covers the existing suite.** Cue → delayed outcome, noise channels, sticky contexts,
  reversals and controllable lights are special cases (constructors `cue_delay`, `noise`,
  `sticky_context`, `hidden_hmm`, `controlled`, `constant`).
* **Makes the product problem explicit.** The joint state space is the product of the
  processes' spaces; a learner that models only the joint state pays that product.

What the family deliberately leaves out (for now): continuous channels, relational
structure between processes, processes that appear and disappear, other agents.
"Normative" in this project means *close to the exact filter / Bayes-optimal behaviour on
worlds from this family* — not optimal in any absolute sense.

## 4. Evidence tiers

| tier | what it measures | instrument | role |
|---|---|---|---|
| T0 | properties: invariances, exactness of formulas, no leakage of task knowledge | `tests/` | must always pass |
| T1 | inference: how close the agent's predictions are to the exact filter | `eval/predictive.py` on `t1_worlds()` | main criterion of slices 1 and 3 |
| T2 | adequacy: classical phenomena of animal learning, translations fixed in advance | `eval/phenomena.py` | main criterion of slice 2; guards biological meaning |
| T3 | usefulness: reward on the behavioural DEV split | `eval/benchmark.py --dev` | at most one or two runs per slice |
| T4 | transfer: held-out worlds, run exactly once on final code | procedural generator (to be built; every fixed held-out set is spent) | at the decision point after slice 3 |

T1 and T2 are cheap (minutes) and are where iteration happens; T3 and T4 are expensive and
are used to confirm, not to search.

## 5. Statistical rules

* The unit of observation is the seed. Effects are reported with 95 % t-intervals
  (`eval/stats.py`), paired by seed where the design allows.
* Before a comparison the minimal detectable effect at the planned number of seeds is
  stated. Measured in S0-1: per-seed sd of the DEV mean ≈ .05, sd of a paired difference
  ≈ .053, hence a paired DEV comparison resolves ≈ .05 on 8 seeds and ≈ .04 on 16. DEV
  comparisons are planned on 16 seeds; smaller differences are "not resolved" and do not
  justify keeping complexity.
* A configuration is chosen only on DEV / T1 / T2, never on held-out data.
* Every decision is logged in `decisions.md` before the run, with a quantitative prediction;
  the outcome is appended and never edited.

## 6. Slices

0. **Footing** (this slice): statistics, family v0 with exact reference, T2 v0, measurement of
   v11, these documents.
1. **Hidden state in one process.** Candidates: (a) v11 plus a latent state over its windows;
   (b) a clone-structured latent model learned online. Decision on T1 (hidden-state worlds)
   and the "delayed food with noise" probe; one T3 run.
2. **Contextual inference (COIN-like).** One inference over contexts replaces v11's change
   detector, regimes and reinstatement. Acceptance: T2 savings and spontaneous recovery
   reproduced without losing the others; DEV not worse than v11.
3. **Decomposition into processes (key slice).** Necessity probe: two processes plus noise.
   Candidates: grouping via attention; factorial models with Bayesian model reduction.
   Criterion: near-filter inference at cost linear in the number of processes.

Decision point after slice 3: the v12 core, and whether C2 is ready.

Per slice: hypothesis, quantitative prediction, translations and budget approved at the
start; derivations checked by brute force; at most two expensive runs; at the end the
result, the accuracy of the prediction and what was removed.
