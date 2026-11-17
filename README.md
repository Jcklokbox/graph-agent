# Blind Gate Graph — index-agnostic prototype

This revision replaces sensor-role-specific branches in `BlindGateGraphBrain` with symmetric processing over `sensor_nodes`, generic motor-conditioned transition predictions, EMA state-value tables, temporal vessels, and a two-step action evaluator.

## Run

```bash
python -m py_compile ops.py brain_p2.py blind_gate.py blind_gate_graph.py test_despecialized.py
python test_despecialized.py
```

The current five-test run is **partially passing**, not a release pass:

- Test 1 — pass
- Test 2 — fail (permuted autonomous play below reward target, shock rate too high)
- Test 3 — pass (conditional inhibition and delay 2–4)
- Test 4 — fail (distant two-step gate)
- Test 5 — fail (reversal learning)

A recorded full-suite run took 9.95 seconds. The JSON result is in `despecialized_test_results.json`.

`ops.py`, `brain_p2.py` (`DynamicQPool`), and `blind_gate.py` are unchanged. The constructor accepts legacy optional flags for compatibility, but the new tests use the default configuration.
