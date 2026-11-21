# Graph Agent

Start with [STATE.md](STATE.md) for the current research verdict and next decision.
For autonomous runs, read [docs/AUTONOMY.md](docs/AUTONOMY.md) before dispatch.

- `development/` — runnable models, benchmarks, tests, and keyed evidence.
- [Archive index](archive/README.md) — historical sources; open only the relevant section.

Quick suite (from `development/`, after installing `requirements.txt`):

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests
```
