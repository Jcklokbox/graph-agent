set -eu
out=$1
model=$2
mkdir -p "$out"
# Oracles and deployed code execute in separate processes and imports.
PYTHONPATH="$3:${PYTHONPATH_DEPS:-/tmp/ar6-deps}" python "$PG_SNAP/eval/scripts/ar7_d65_parity.py" "$out/d65" d65
python -m eval.scripts.ar7_d65_parity "$out/candidate" "$model"
for tape in sticky canc-noise3; do cmp "$out/d65/$tape.npy" "$out/candidate/$tape.npy"; done
python - "$out" "$PG_TREE" "$model" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'d65/tapes.json').read_text());a=json.loads((p/'candidate/tapes.json').read_text())
for k in d:
 assert {x:y for x,y in d[k].items() if x!='pure_plans'}=={x:y for x,y in a[k].items() if x!='pure_plans'}
 assert a[k]['pure_plans']==5
(p/'parity.json').write_text(json.dumps(dict(d65='6f619568dc90',tree=sys.argv[2],model=sys.argv[3],bit_exact=True,tapes=a),indent=2,sort_keys=True)+'\n')
PY
FACTOR_ISOLATION_CANDIDATE="$model" python -m unittest -v tests.test_factor_control tests.test_factor_reward > "$out/isolation.log" 2>&1
python -m eval.scripts.ar6_assurance "$out/assurance.json" --model "$model" --scale
