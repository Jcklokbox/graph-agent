"""Check coverage costs near genuine control-code revisions in short dev prefixes."""
import json

from eval.scripts.distal_contact_probe import CLEAN, NOISY, SEED
from eval.scripts.mode_coverage_probe import continuation, warmup


# Include the diagnosed failure, previously working controls and two additional
# dev variants, fixed before observing the revision-gate results.
CASES = ((CLEAN, SEED), (NOISY, SEED), (CLEAN, 1774893116), ('canc-noise3', 15),
         ('aut', 6), ('aut-noise4-sticky', 6), ('dist-noise2-sticky', 6))


def main():
    for task, seed in CASES:
        checkpoint = warmup(task, seed, 0)
        for policy in ('off', 'active', 'revisited'):
            measures = continuation(checkpoint, 0, 160, policy=policy)
            print(json.dumps(dict(task=task, seed=seed, policy=policy, **measures)), flush=True)


if __name__ == '__main__':
    main()
