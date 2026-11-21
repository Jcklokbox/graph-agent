# Research state

For run recovery: `./ar resume`; see the [documentation index](docs/INDEX.md).
Do not start another run without authorization.

<!-- managed-next -->
- **Run AR-7 (closed):** AR-7 PASS: v60 accepts 3b-2 on the fresh 300-pair gate plus exact parity, isolation and scaling. T4 not opened.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740 tuples, cost ratio 2.080340. [Verdict](development/results/ar5/gate3/validation.json).
- **3b-2 accepted:** v60 `d2c4031`, 600/600 episodes and 300/300 pairs. Paired difference +0.04972 vs v12, 99% CI [+0.00229, +0.09715]. D65 tape parity 640/640, isolation 15/15, full-step scale 1.861×. [Report](docs/research/AR7.md).
- **Current v60 checkout:** [shared forecast kernels](docs/research/V60_PATTERNS.md#exact-reuse-in-the-live-forecast) avoid rebuilding identical continuation branches per action or reading an identical birth row per particle. Short direct comparisons preserve Q and actions bitwise. The [action reward partition for unchanged contexts](docs/research/V60_PATTERNS.md#action-reward-with-an-unchanged-context) is a separate behavioral change. The accepted historical freeze and its evidence are unchanged.
- **Local research:** [reward partition selection](docs/research/REWARD_PARTITIONS.md) remains experimental. The [coverage audit](docs/research/V60_PATTERNS.md#a-mode-reset-does-not-necessarily-make-repeated-trials-wasteful) shows that post-reset trials can aid recovery. The [budget check](docs/research/V60_PATTERNS.md#a-hard-probe-budget-caps-interventions-not-their-reward-cost) bounds action overrides, but not their opportunity cost: two credits helped five of eight separate dev prefixes and harmed three, with mixed serial recovery. Coverage remains offline.
- **Next qualification decision:** blind T4 has not been opened. Serial reversals and delayed distal combinations remain priority stress cases.
- [Evidence replay and historical recovery](docs/EVIDENCE.md).
