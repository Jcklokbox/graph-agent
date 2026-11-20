# Posterior graph probe: initial results and scope

Run on 2026-11-17 with `python research/posterior_graph_probe.py` (stdlib-only, deterministic seed). It completed in under one second.

```
PASS test_calibration: Brier=0.201; nuisance pair-vs-main log evidence=-3.63
PASS test_structure: AND/XOR pair evidence beats pooled; redundant channel adds no evidence
PASS test_inhibition_memory: p(Y|A)=0.981, p(Y|A,X)=0.019; A-only posterior retained
PASS test_timing: discrete hazards=[0.077, 0.417, 0.625, 0.75]; one-step omission included in risk set
PASS test_value: Q(setup)=0.800; after reward revaluation=-0.055; blocked=-0.100
PASS test_equivariance: all 36 sensor/action permutation combinations map observations and unique winner exactly
RESULT 6/6 pass
```

These are **equation-level unit probes, not evidence that the architecture passes**. The first version exposed two test-construction errors: the structure contrast accidentally gave the baseline an extra predictor, and the timing test conflated modal lag with maximum conditional hazard. Both tests were corrected; no model parameter was tuned to a behavioral benchmark.

Important limitations of the probe:

- Calibration exercises a pooled Beta-Bernoulli estimate; nuisance structure evidence is a fixed generated table. It does not test sequential graph birth or posterior coverage.
- AND/XOR use integrated Beta-binomial evidence with the true pair offered in advance. There is no implemented residual-triggered proposal process, pair multiplicity correction, recursive composition, or held-out online structure selection. The result supports representation sufficiency, not the proposed structure learner.
- Inhibition is a direct conditional frequency table, not a learned inhibitor transferred by summation and retardation protocols. Memory is merely separate A-only counts.
- Timing computes discrete hazards with right censoring but does not test event consumption, overlapping pending events, pause handling through a live agent, or forecast calibration on held-out episodes.
- The two-step value result evaluates a supplied exact transition graph. It does not learn transitions or implement the proposed graph-based planning readout.
- The “equivariance” check permutes a hard-coded score vector and unique argmax. It does not train a learner on transformed data, test mirrored random tie-breaking, or test closed-loop interactions.

Consequently **the integrated candidate has not passed its falsifiers**. It has not been moved into `development/`, and the five-task suite / strict learner equivariance have not been run against it. This is an explicit no-go on migration, not a pass inferred from six disconnected arithmetic checks. The specific mathematical tightening made after the probe is the structure-selection odds equation and need for bounded total birth prior mass / multiplicity control; no experimental evidence here validates that remedy. The next minimum meaningful experiment must implement one shared transition-posterior object with actual structure proposals and compare its learned predictions on held-out data, rather than add more disconnected unit tests.
