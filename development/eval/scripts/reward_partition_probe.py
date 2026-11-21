"""Four fixed, already exposed intervention tapes; no qualification or new seeds.

Run from development/ with ``python -m eval.scripts.reward_partition_probe``.
This diagnostic prints fixed action margins and does not write evidence.
"""
from collections import deque
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from envs.tasks import VARIANTS
from eval.scripts.ar6_adverse import build, run
from models.factor_event_reward import SignedSelection
from research.reward_partitions import fit_arrivals, fixed_values, refit_arrivals


CASES = (('canc-noise3', 15, 'same'), ('canc-noise3', 15, 'opposite'),
         ('canc-noise3', 15, 'neutral'), ('canc', 6, 'same'))


def probe(name, seed, arm):
    v = VARIANTS[name]
    longer = None
    original = SignedSelection.learn_joint

    def capture(head, b, hist, action, reward, arrival):
        nonlocal longer
        if longer is None:
            longer = deque(maxlen=2 * head.replay.maxlen)
        old_slots = head.slots.copy()
        old_mode = head.switches
        original(head, b, hist, action, reward, arrival)
        if old_mode != head.switches or not np.array_equal(old_slots, head.slots):
            longer.clear()
            longer.extend(head.replay)
            return
        longer.append(head.replay[-1])

    def compare(factor, arrival, x, uniforms):
        result = {}
        head = factor.reward
        replay = fit_arrivals(head)
        extended = fit_arrivals(head, longer)
        comparisons = (
            ('short_short', replay, head.replay),
            ('short_long', refit_arrivals(head, replay, longer), longer),
            ('long_short', refit_arrivals(head, extended, head.replay), head.replay),
            ('long_long', extended, longer),
            ('context_time', fit_arrivals(head, longer, allow_change=True), longer),
        )
        for label, laws, rows in comparisons:
            law = laws[arrival]
            p = law.partition
            q = fixed_values(factor, x, uniforms, laws)
            result[label] = dict(groups=p.groups, cue=int(p.cue), moderator=int(p.moderator),
                                 margin=float(q[0] - q[1]), rows=len(rows))
            if label == 'context_time':
                result[label]['changes'] = {c: other.changes for c, other in enumerate(laws)
                                            if other.changes}
        return result

    with patch.object(SignedSelection, 'learn_joint', capture):
        return run(v, seed, arm, build(v, seed, 4, 16, 5), 5, compare)['results']


def main():
    source = Path(__file__).resolve().parents[2] / 'results/ar7/joint-fixture.json'
    saved = {(row['variant'], row['seed'], row['arm']): row['probes'] for row in json.loads(source.read_text())}
    for case in CASES:
        for row, ref in zip(probe(*case), saved[case], strict=True):
            if row['cancelled'] != ref['cancelled']:
                raise ValueError('wrong probe alignment')
            print(json.dumps(dict(case=case, t=row['t'], cancelled=row['cancelled'],
                                  v60=ref['delta'], **row['structure']), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
