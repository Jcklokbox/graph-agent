# Active checkpoint

- **3b-1 accepted:** D65, freeze `6f619568dc90`; 740/740 keys, cost ratio 2.08034.
  Evidence: `development/results/ar5/gate3/validation.json`.
- **3b-2 not qualified:** frozen `v48` is a research candidate, not a replacement for `v12`.
  D112 DEV-16: 240/240 pairs; Δ −0.00330, 99% CI [−0.04336, +0.03677], required lower ≥ −0.02.
  Evidence: `development/results/ar5/gate-dev16-v48/{DEV,validation}.json`.
- **Next:** distinguish never discovering payoff (D113) from failing to reuse observed payoff
  (D97); test both without task labels. See [the three-hour plan](docs/AUTONOMY.md#next-three-hour-run).
- **Recovery:** [run protocol](docs/AUTONOMY.md); an active run's clock, call budget,
  job statuses and next action are in `.run/checkpoint.json`. AR-5 is closed, no live clock to resume.
- **Provenance:** [evidence index](archive/README.md); checkpoint [PR #16](https://github.com/Jcklokbox/graph-agent/pull/16)
  merged at `67ee305`. Archival merge is not 3b-2 acceptance. DEV-16 is now exposed regression evidence.
