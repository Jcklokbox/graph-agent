"""A fully specified two-step posterior decision fixture (analysis, not an agent)."""
from __future__ import annotations

from typing import NamedTuple


class Branch(NamedTuple):
    mass: float
    immediate: float
    continuation: tuple[float, ...]  # same actions on every branch of one root


def exact_q(branches, discount):
    """Return Bayes adaptive Q2 and its continuation's observation-information gain."""
    if abs(sum(b.mass for b in branches)-1.) > 1e-12:
        raise ValueError('branch probabilities must sum to one')
    if any(b.mass < 0 for b in branches):
        raise ValueError('negative branch')
    first = sum(b.mass*b.immediate for b in branches)
    adapted = sum(b.mass*max(b.continuation) for b in branches)
    precommitted = max(sum(b.mass*b.continuation[a] for b in branches)
                       for a in range(len(branches[0].continuation)))
    return first+discount*adapted, adapted-precommitted


def observe_then_choose(reliability=.90, stable_reward=.20, discount=.95):
    """Hidden fair bit; first action buys a noisy signal or takes a sure reward.

    At the second decision, either action has reward 1 iff it guesses the bit.
    The information action must win at default parameters even though it has
    strictly lower immediate reward. Adding EVSI a second time is incorrect.
    """
    signal = (Branch(.5,0.,(reliability,1-reliability)),
              Branch(.5,0.,(1-reliability,reliability)))
    idle = (Branch(1.,stable_reward,(.5,.5)),)
    return exact_q(signal,discount), exact_q(idle,discount)


def dirichlet_mean_moments(n, sum1, sum2, concentration=1., prior_m1=.5, prior_m2=1/3):
    """Mean uncertainty under a bounded random-measure prior; n counts observations."""
    mass=concentration+n
    mean=(concentration*prior_m1+sum1)/mass
    second=(concentration*prior_m2+sum2)/mass
    obs=max(0.,second-mean**2)
    return mean, obs/(mass+1.), obs
