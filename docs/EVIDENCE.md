# Evidence replay and history

These are read-only audits, not fresh evaluations. From `development/` with
`requirements.txt` installed:

```sh
GRAPH_AGENT_MACHINE=1 python -m research.ar5_validate results/ar5/gate3
python -m research.ar7_gate audit results/ar7/dev20-v60
```

Compare each command's JSON with its accepted report:

- [D65 / 3b-1 verdict](../development/results/ar5/gate3/validation.json)
- [v60 / 3b-2 verdict](../development/results/ar7/dev20-v60/DEV.json)
- [v60 freeze](../development/results/ar7/dev20-v60/freeze.json) and
  [audit hashes and checks](../development/results/ar7/final-audit.json)

The recorded content identities refer to the original code, not this cleaned
checkout. To reconstruct a frozen code tree, export `development/` from its
registered commit into a scratch directory with `git archive`:

| Freeze | Commit | Content tree |
| --- | --- | --- |
| D65 | `ac37d503f3d0f655bbfca616f0766b9154d9a3b5` | `6f619568dc90` |
| v60 | `d2c4031bf14b236114e46009cd73fadb7a26f56c` | `6ba39c831f35` |

For retired experiments, reports, and documents, inspect the pre-cleanup Git
revision rather than restoring a bulk archive to the working tree:

```sh
git ls-tree -r --name-only a17d4d26 -- archive development/results docs
git show a17d4d26:<path>
```

Fetch repository history first if the checkout is shallow and the requested
commit is missing. Historical seeds and partially completed gates remain spent.
