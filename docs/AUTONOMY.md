# Active autonomous-run contract

**Do not start a run without authorization.** Then work continuously across tool updates and compaction. Stop only at the agreed wall/call limits, the agreed terminal goal, an explicit stop, or a genuine external blocker. An unresolved research question is work, not a blocker. Do not idle to fill the clock. Use English in active files, state, and responses. Historical records do not set rules.

## Recovery and durability

- Run **`./ar resume`** immediately after compaction. It prints the beacon, actual checkpoint summary, `STATE.md`, and [active index](INDEX.md); it never starts a run. Human-facing harness, research and evaluation CLIs emit the beacon on entry and exit (including errors) and every 75 s while running. The beacon reports elapsed wall / W, reported calls / cap, and reflection time. `./ar exec -- COMMAND` wraps other human-facing procedures. `GRAPH_AGENT_MACHINE=1` is only for internal workers and pure data streams; keep JSON files and audit comparisons beacon-free.
- This is an active **stdout** cue, not a platform hook. If compaction strips both stdout and its summary, the repository cannot inject instructions or trigger a new assistant turn. Put `./ar resume` in the run's initial instruction too. A hard platform termination still requires a new authorized turn.
- Keep `.run/checkpoint.json` for same-sandbox recovery. At initialization, each reflection, and before a long dispatch or valuable freeze, run `./ar publish` and commit/push `ops/recovery.json`, the short `STATE.md`, reproducible pipeline changes, and completed evidence to **the session branch**. Land durable milestones on `main` by numbered PR, not by pushing to another branch. Never commit volatile worker logs, caches, or secrets. Verify the remote checkpoint commit, not merely the local file.
- If the sandbox resets, `resume` restores from the published checkpoint using its original UTC epochs, counts downtime, and blocks dispatch in `reconcile`. Check Git and journals; missing terminal markers do not prove success or liveness. Charge any identifiable or conservatively bounded intervening calls via `--used N` and record what survived with `reconcile --evidence '…' --next '…'`. If the unknown gap cannot be bounded or the clock/calls expired, don't continue that run. Do not reinitialize to recover lost allowance. Only published data can be guaranteed to survive loss of the sandbox.

## Dispatch and calls

The observed previous cliff was **257 bash + 19 other = 276 calls**, not a platform guarantee. Use **220 working + 20 reserve = 240 hard**, aiming for **<150** total. Count all tool and parallel subcalls conservatively. `--used N` (before the subcommand) is calls since the last persisted update, including the current one; do not double-charge commands bundled within one platform call. The platform counter is not exposed. Reserve the final 10 min by default. Reserve is for reflection, audit, publishing and handoff, not new jobs.

Use one managed call per multistep inspection/test or immutable batch. Keep full logs locally; read narrow tails or journals, not repeated per-worker polls. `dispatch` takes a JSON list of up to eight jobs with disjoint pinned CPU affinities (`name`, `cpu`, `seconds`, `argv`); argv runs from scratch with a watchdog. Use absolute artifact destinations. `wait` returns one compact batch summary and ends by the next reflection or reserve. Prefer CPU time isolated for cost; estimate by warmed kernels and measure per-stage wall with headroom. Keep acceptance stages append-only, seed-major and keyed; a prefix can reject but cannot qualify. Use `checkpoint --close --next '…'` only after reconciling tracked jobs and outcomes, then publish the closure.

## Mandatory reflection and dynamic horizon

Every **30 min from the original start**, pause new dispatch. Existing bounded jobs may finish; wait ends at reflection. Before further batch work, write the three entries and one immediate next action:

```sh
./ar reflect --local 'Trace and causal contrast; alternative explanations' \
  --global 'Relevant L-* laws, biology, asymptotic work and roadmap implication' \
  --resolve 'Question, logic/math/micro-probe, answer or remaining uncertainty' \
  --next 'One falsifiable next decision'
```

Reflection updates the managed next-action block of `STATE.md`; review it and publish. A late reflection does not move the original cadence. Resolve uncertainties yourself with a targeted probe or deduction; if evidence is unfinished, state the uncertainty and make its discriminating probe the next action. Replan from current observations, confidence/coverage, remaining calls and measured throughput. No fixed minute-by-minute itinerary. Don't relax frozen comparisons, use a witness to claim an agent result, or count spent holdouts as fresh.

## Next authorized run: first decision only

On frozen `v48`, reproduce the exposed `canc` / `canc-noise3` split (D113 seed 6 both, clean seed 0 positive; D97 is a separate early-positive/poor-reuse witness). Freeze **paired, anonymous** fixtures: same exogenous seed, cue/timing schedule and action labels. One arm acts freely to test discovery; a second uses a scripted action prefix to generate a real payoff observation, then acts freely to test reuse. Yoke an adverse/null prefix with the same opportunity count and timing. Update the learner only through ordinary observations/actions in the diagnostic; no teacher flags, synthetic credit, or scripted actions in a qualification run. Log first maximum-payoff observation, eligible/reached interventions, and action choice/regret conditional on matched exposure. Keep transition discovery, reward credit and reuse separate; compare negative branch interventions too. No task labels, legacy controller, or unseen formal rows for tuning. Use the outcome to select the next probe, not a predetermined development schedule.
