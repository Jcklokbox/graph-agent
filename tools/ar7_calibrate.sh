# Executed by the immutable runner. Both agents use the same affinity and cache.
set -eu
model=$1
out=$2
workers=$3
python -m eval.scripts.ar7_cost "$out/cal-v12" v12 "$workers"
python -m eval.scripts.ar7_cost "$out/cal-$model" "$model" "$workers"
