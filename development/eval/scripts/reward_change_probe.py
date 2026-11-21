"""Closed-loop diagnostic for local reward cuts, not a v60 configuration.

Both policies get the same contexts, paired payoff noise, bootstrap and
optional exploration budget. Actions have no effect on the next context.
"""
import json

import numpy as np

from research.reward_partitions import fit


STEPS, SWITCH, RARE_PERIOD = 320, 160, 8
SCENARIOS = ('stable', 'reversal', 'unseen_improvement')


def episode(scenario, seed, *, cuts, probes, noisy=False):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    noise = np.random.default_rng(seed).random((STEPS, 2))
    rows = [[], []]
    observed = np.zeros(STEPS)
    picked = np.zeros(STEPS, dtype=int)
    admitted = np.zeros(STEPS, dtype=bool)
    rare = np.arange(STEPS) % RARE_PERIOD == 0
    initial = seed % 2
    for t, context in enumerate(rare.astype(int)):
        history = rows[context]
        visits = len(history)
        # Matched bootstrap: both actions are tried twice in each context.
        if visits < 4:
            a = (initial + visits) % 2
        else:
            actions, rewards = zip(*history)
            bits = np.asarray(actions, dtype=bool)[:, None]
            law = fit(bits, bits, rewards, [0], [], allow_change=cuts)
            lookup = np.arange(2) if law.partition.groups == 2 else np.zeros(2, dtype=int)
            p, atoms, _ = law.atoms(lookup)
            values = (p * atoms).sum(-1)
            a = initial if values[0] == values[1] else int(values.argmax())
            admitted[t] = bool(law.changes)
            if context and probes and visits % 4 == 3:
                a = 1 - a
        favored = .6 if scenario == 'unseen_improvement' and context else .9
        amount = favored if a == initial else .1
        if context and t >= SWITCH:
            if scenario == 'reversal':
                amount = .1 if a == initial else .9
            elif scenario == 'unseen_improvement' and a != initial:
                amount = .9
        reward = float(noise[t, a] < amount) if noisy else amount
        history.append((a, reward))
        observed[t], picked[t] = reward, a
    post = np.flatnonzero(rare & (np.arange(STEPS) >= SWITCH))
    return dict(rare_post=float(observed[post].mean()), rare_first_six=float(observed[post[:6]].mean()),
                common=float(observed[~rare].mean()), rare_post_cuts=int(admitted[post].sum()),
                post_alternative=int((picked[post] != initial).sum()))


def main():
    for scenario in SCENARIOS:
        for probes, noisy in ((False, False), (True, False), (True, True)):
            for seed in (0, 1, 2):
                print(json.dumps(dict(scenario=scenario, probes=probes, noisy=noisy, seed=seed,
                                      control=episode(scenario, seed, cuts=False, probes=probes, noisy=noisy),
                                      temporal=episode(scenario, seed, cuts=True, probes=probes, noisy=noisy))))


if __name__ == '__main__':
    main()
