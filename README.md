# Graph Agent

Graph Agent learns predictive structure and selects actions from anonymous
observations and bounded rewards. The accepted controller is `v60`; `v12`
remains the benchmark comparator.

- [Current state](STATE.md) and [documentation index](docs/INDEX.md)
- [Architecture](docs/research/ARCHITECTURE.md) and [roadmap](docs/research/ROADMAP.md)
- [Acceptance report](docs/research/AR7.md) and [evidence replay](docs/EVIDENCE.md)

Run the same regression checks as CI (Python 3.11, from the repository root):

```sh
python -m pip install -r development/requirements.txt
(cd development && OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 FACTOR_ISOLATION_CANDIDATE=v60 python -m unittest discover -s tests)
bash tools/check_evidence.sh
```
