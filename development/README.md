# Development workspace

This directory contains the executable project and its current empirical/test artifacts. It is the place for implementation work on the agent and its environment; the broader theoretical workspace lives in [`../research/`](../research/README.md).

## Files

- `blind_gate.py` — three-position delayed-food environment, baseline policies, and benchmark helpers.
- `blind_gate_graph.py` — graph-based agent, transition learning, temporal relations, action evaluation, and query/snapshot APIs.
- `brain_p2.py` — lightweight `DynamicQPool` compatibility/statistics helper.
- `ops.py` — bounded activation/evidence operations.
- `test_*.py` — behavioral and structural test suites.
- `*_test.json` — recorded task or benchmark data used by the project.

## Running

Run commands from this directory so the local modules resolve:

```bash
python -m py_compile ops.py brain_p2.py blind_gate.py blind_gate_graph.py test_despecialized.py
python test_despecialized.py
python -m pytest -q
```

The prototype depends on NumPy; pytest is needed for the pytest runner. Test targets are exploratory and should not be read as a release guarantee. The previously committed README reported partial passes on a five-test run, including failures in autonomous generalization, a distant two-step gate, and reversal learning. Re-run tests in the current environment before relying on that historical status.

The implementation is one evolving realization of the project, not a complete specification of the research program. Changes to code and tests can proceed alongside independent theoretical development in `research/`.
