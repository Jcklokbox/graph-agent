# Research state

For run recovery: `./ar resume`; see the [documentation index](docs/INDEX.md).
Do not start another run without authorization.

<!-- managed-next -->
- **Run AR-7 (closed):** AR-7 PASS: v60 accepts 3b-2 on the fresh 300-pair gate plus exact parity, isolation and scaling. T4 not opened.
<!-- managed-next -->

- **3b-1 accepted:** D65 `6f619568dc90`, 740/740 tuples, cost ratio 2.080340. [Verdict](development/results/ar5/gate3/validation.json).
- **3b-2 accepted:** v60 `d2c4031`, 600/600 episodes and 300/300 pairs. Paired difference +0.04972 vs v12, 99% CI [+0.00229, +0.09715]. D65 tape parity 640/640, isolation 15/15, full-step scale 1.861×. [Report](docs/research/AR7.md).
- **Next research decision:** preregister blind T4. Serial reversals and delayed distal combinations are priority stress cases. T4 has not been opened.
- [Evidence replay and historical recovery](docs/EVIDENCE.md).
