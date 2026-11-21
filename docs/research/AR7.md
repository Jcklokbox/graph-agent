# AR-7: deploy joint reward reuse; qualify only on a fresh paired draw

## Exposed selection

v60 extends v59's reward member, not the sensory predictor. Each actual
arrival compares pooled, unary and joint partitions on the same bounded
window. The joint family separates two observed moderator states at a real
onset and its outside branch; all three must be witnessed. Its log integrated
likelihood pays multiplicity cost before it can displace an admitted unary
rule. The selected predictive atoms retain the actual bounded payoff sums.
Optimistic and fixed two-step planning use the same projection; imagining
one atom changes no live statistics, mode, sensory state or RNG.

The candidate set has at most four sensory onset anchors, with a linear scan
of possible moderators and bounded lags. The real replay bound is `2*C*A`,
where `C<=16` is the bounded controlled state count. Count refresh and storage
are linear in channel count at fixed action count, replay bound and lags.
A rollout reads those counts only. The extra group identifier keeps the
outside branch distinct from both onsets in the virtual atom update. Actual
mode/scope changes rebuild from the current real replay, not from imagined or
stale old-mode counts. This is a batch implementation of the common-window
likelihood, not the prior streaming identity inserted into the production path.

- **AR7-02, `6ba39c831f35`:** all 54 fixed-value probes in 12 already-exposed
  intervention arms match the independent AR6 batch oracle exactly; 10 probes
  admit a compound. Seed 15 same-side real adverse and positive contact
  reproduces both signed decisions on both clean and noisy tapes. Unit
  checks cover exact real amounts, snapshot purity and missing support.
- **23/23 closed-loop exposed comparisons:** v60 improves noisy cancellation
  seed 12 (.72727 to .90909) and 13 (.73684 to 1.0), and ties 21. Mean noisy
  cancellation gain +.063568 across seven, no change across seven clean,
  three serial, three late-food and three distal guards. Serial remains a
  weakness; a tie here is not improvement. These are selection results only.
- v60 selected before fresh payoff, conditional on D65 tape parity, D111
  controller isolation, N200 full-step scale and complete funding. v59 is
  the exposed incumbent; the separately fetched replica is not a finalist.
- The trusted D65 and spent DEV-20 journals are unchanged. T4 not opened.

See [compute and preregistration](AR7-COMPUTE.md) and
[raw receipts](../../development/results/ar7/).

## Frozen qualification

- Candidate `d2c4031bf14b236114e46009cd73fadb7a26f56c`, tree `6ba39c831f35`. D65 fixed-tape
  parity: 640/640 byte-exact predictions and 10 pure planning checkpoints.
  Archived 740-tuple journal aggregation is byte-identical; its episodes
  were not regenerated. All 15 named D111 cases passed with v60 included in
  the actual-agent isolation cases. Warm full-step N200 CPU scaling:
  2.597444 s → 4.834566 s, ratio 1.861278 ≤ 2.50.
- Program SHA-256 `7f04badee1c825d7474ce20fe863ffd2f67243bf4b615a49b945d3c59d2838b3`. Fresh 20×15 blocks; 30 full,
  independently balanced stages per model. Both models run on CPUs 0,1 with
  2 workers and single-thread numerical libraries.

| Actual model on the finalist tree | Exposed stages / episodes | Mean / max stage s | Watchdog s |
| --- | ---: | ---: | ---: |
| v12 | 6 / 60 | 127.801 / 132.153 | 167 |
| v60 | 6 / 60 | 64.848 / 76.203 | 97 |

Fund all stages and grace: 8190 s, plus 600 s for audit. This
supersedes the proxy 11,580 s / latest start 1795265408. At exact freeze the
latest gate start is **2026-11-21T13:46:38+00:00** (`1795268798.582`); gate deadline
`1795276988.582`. The immutable freeze was published before any fresh
episode. No independent fresh reference cache, no shortened or retried stage.
