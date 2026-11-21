"""AR-6 exact-replay decision dump: replicate the v48 controller step by step.

The controller of every agent v21..v49 is
    u = rng.random((draws, J+3+control)); theta = sample(rng) if posterior
    q = f.two_step_values(x, u, theta); a = argmax-tie-break
so this module can replay it bit-exactly while also reading the reward head's
internal terms. It never writes to the model.
"""
from __future__ import annotations

import numpy as np


def act(brain, x):
    f = brain.factor
    u = brain.rng.random((brain.draws, f.J + 3 + int(f.reward_control)))
    theta = f.reward.sample(brain.rng) if brain.posterior else None
    q = f.two_step_values(x, u, theta)
    a = int(brain.rng.choice(np.flatnonzero(q == q.max())))
    f.predict(x, a)
    brain.pending = (x, a)
    return a, q


def terms(head, f, x):
    """Read-only decomposition of the initial (pre-branch) reward terms."""
    b = f._bits(x)
    rule = head.rule()
    input0 = head.input(b, f.hist)[1:]
    r, p, _, _ = head.likelihood(rule, input0[None, :])
    splits = head.signed_rules(rule) if hasattr(head, 'signed_rules') else []
    seen = head.phase(b, f.hist)
    base = rule[5]
    bound, _ = head.provisional(rule)
    inc = np.maximum(r[0], (bound * seen).max(1))
    out = dict(rule_idx=rule[0].tolist(), rule_w=rule[1].astype(int).tolist(),
               r=np.round(r[0], 4).tolist(), base=np.round(base, 4).tolist(),
               inc=np.round(inc, 4).tolist(), seen=[int(i) for i in np.flatnonzero(seen)],
               bnd=np.round((bound * seen).max(1), 4).tolist(),
               n_onset=head.onset_total.round(0).tolist(),
               slots=head.slots.tolist(), splits=[[int(j), int(m), int(l), int(s.sum())]
                                                  for j, m, l, s, g, d in splits])
    return out
