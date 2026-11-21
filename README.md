# Graph Agent

Graph Agent learns predictive structure and selects actions from anonymous
observations and bounded rewards. The accepted controller is `v60`; `v12`
remains the benchmark comparator.

- [Current state](STATE.md) and [documentation index](docs/INDEX.md)
- [Architecture](docs/research/ARCHITECTURE.md) and [roadmap](docs/research/ROADMAP.md)
- [Acceptance report](docs/research/AR7.md) and [evidence replay](docs/EVIDENCE.md)

Run the test suite from `development/` in a Python environment with the pinned
requirements installed:

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m unittest discover -s tests
```
