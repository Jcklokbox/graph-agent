set -eu
root=$1
m=$2
w=$3
# Compile the bounded count kernel and verify the independent arithmetic and
# actual frozen replay fixture before drawing any closed-loop conclusions.
if [ "$m" = v60 ]; then
  python -m unittest -v tests.test_factor_joint_reward > "$root/joint-unit.log" 2>&1
  python -m eval.scripts.ar7_joint_fixture "$root/joint-fixture.json"
fi
python -m eval.scripts.ar5_controller_run "$root/$m-cancellation.jsonl" --model "$m" --workers "$w" --seed 2 6 7 8 12 13 15 --variants canc canc-noise3
python -m eval.scripts.ar5_controller_run "$root/$m-guards.jsonl" --model "$m" --workers "$w" --seed 6 13 2 --variants aut-serial300 aut-fooddelay3-noise1 dist-delay3-noise2
