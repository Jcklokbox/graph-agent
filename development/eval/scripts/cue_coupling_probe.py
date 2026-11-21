"""Check the v60 arrival/cue interface on four already exposed tapes.

Run from development/ with ``python -m eval.scripts.cue_coupling_probe``.
Uses forced prefixes and fixed Q, not the optimistic policy or qualification.
"""
import json
from unittest.mock import patch

from agents.v60 import Agent
from envs.tasks import VARIANTS
from eval.scripts import ar6_adverse
from eval.scripts.reward_partition_probe import CASES
from research.cue_coupling import compare_cues


def current_v60(n, actions, value, *, seed, draws, **_):
    return Agent(n, actions, value, seed, draws=draws)


def main():
    with patch.object(ar6_adverse, 'FactorController', current_v60):
        for name, seed, arm in CASES:
            v = VARIANTS[name]
            table = ar6_adverse.build(v, seed, 4, 16, 5)
            rows = ar6_adverse.run(v, seed, arm, table, 5, lambda f, _, x, u: compare_cues(f, x, u))
            for row in rows['results']:
                print(json.dumps(dict(case=(name, seed, arm), tick=row['t'], analysis=row['structure'])))


if __name__ == '__main__':
    main()
