cd development
PG_PROCS=2 /tmp/venv/bin/python -m eval.benchmark --dev --seeds 16 --agent v52 --out results/ar6/dev16_v52.json > results/ar6/dev16_v52.log 2>&1
echo "exit=$?" > results/ar6/dev16_v52.done
