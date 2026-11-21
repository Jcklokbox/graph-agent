# Graph Agent

Start with [STATE.md](STATE.md) for the current research verdict and next decision.
After compaction run `./ar resume`; use the [active English index](docs/INDEX.md).

- `development/` — runnable models, benchmarks, tests, and keyed evidence.
- [Archive index](archive/README.md) — historical sources; open only the relevant section.

Quick suite (from `development/`, after installing `requirements.txt`):

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests
```
