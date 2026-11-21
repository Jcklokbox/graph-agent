"""Closed-loop audit of finite coverage credits on exposed dev prefixes.

Screen: compare with unrestricted active coverage, including mode forks.
Check: compare the preselected two-credit arm with v60 on distinct task/seeds.
No qualification splits or runner are used.
"""
import argparse
import json

from eval.scripts.distal_contact_probe import CLEAN, NOISY
from eval.scripts.mode_coverage_probe import advance, continuation, warmup
from eval.scripts.recode_coverage_probe import CASES


# Unchanged before opening either arm. Pair seeds across task families.
CHECK_CASES = tuple((task, seed) for task in (CLEAN, NOISY, 'aut-noise4-sticky', 'canc-noise3')
                    for seed in (26, 91))


def compare(checkpoint, *, task, seed, start, steps, limits):
    for label, limit in limits:
        result = continuation(checkpoint, start, steps, policy='off' if label == 'off' else 'active',
                              limit=limit)
        print(json.dumps(dict(task=task, seed=seed, start=start, steps=steps,
                              arm=label, limit=limit, **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', choices=('screen', 'check', 'modes'))
    panel = parser.parse_args().panel
    limits = (('off', None), ('one', 1), ('two', 2), ('unlimited', None))
    if panel == 'modes':
        task = 'aut-serial300'
        for seed in (6, 1822923777):
            checkpoint = warmup(task, seed, 300)
            compare(checkpoint, task=task, seed=seed, start=300, steps=80, limits=limits)
            if seed == 6:
                checkpoint = advance(checkpoint, 300)
                compare(checkpoint, task=task, seed=seed, start=600, steps=80, limits=limits)
        return
    if panel == 'check':
        limits = (('off', None), ('two', 2))
    for task, seed in CHECK_CASES if panel == 'check' else CASES:
        compare(warmup(task, seed, 0), task=task, seed=seed, start=0, steps=160, limits=limits)


if __name__ == '__main__':
    main()
