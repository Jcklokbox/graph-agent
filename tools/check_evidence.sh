#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../development"
out=$(mktemp -d)
trap 'rm -rf -- "$out"' EXIT

GRAPH_AGENT_MACHINE=1 python -m research.ar5_validate results/ar5/gate3 > "$out/d65.json"
cmp "$out/d65.json" results/ar5/gate3/validation.json
printf '%s\n' 'D65 / 3b-1 verdict matches byte for byte.'

python -m research.ar7_gate audit results/ar7/dev20-v60 > "$out/ar7.json"
cmp "$out/ar7.json" results/ar7/dev20-v60/DEV.json
printf '%s\n' 'AR-7 / 3b-2 verdict matches byte for byte.'
