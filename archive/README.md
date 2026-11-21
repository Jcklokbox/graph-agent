# Historical evidence

Start at [the active index](../docs/INDEX.md). [Technical references](research/README.md)
and [archived code](development/README.md) remain available, not part of the active contract.
`development/results/ar5/gate3/` and `development/results/ar5/gate-dev16-v48/`
contain untouched numeric verdicts and paired journals.

**Two layers.** English modules preserve the decision-relevant interpretations, equations,
figures, contradictions and evidence limits. `docs/source-map.json` audits every pre-cleanup
Markdown file and names its exact source path, immutable commit, bytes, SHA-256 and section offsets.
This makes source preservation byte-lossless via Git; condensed text alone is not an invertible
translation. Raw non-English prose is not emitted by the excerpt tool. Do not load it into an
active run. Shallow checkouts may need `git fetch --unshallow origin` before historical lookup.

```sh
./ar exec python3 tools/doc_audit.py check
./ar exec python3 tools/doc_audit.py excerpt D-36 --part 2 --page 0
```
