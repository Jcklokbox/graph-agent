import unittest

import numpy as np

from models.latent_cause import LatentCauseSMC, exact_predictive


def tiny_stream(seed):
    """6 steps over channels 0 cue, 1 ctx, 2 US: acquisition, then a context switch."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(6):
        cue = float(t % 2 == 0)
        ctx = float(t < 3)
        us = cue * float(t < 3) if t < 5 else 0.
        if rng.random() < .2:
            ctx = 1. - ctx
        out.append(((cue, ctx, 0.), bool(us)))
    return out


class ExactVsFilter(unittest.TestCase):
    def test_predictive_matches_enumeration(self):
        for seed in range(3):
            for h in (.05, .3):
                s = tiny_stream(seed)
                exact = exact_predictive(s, emit=[0, 1], cues=[0], us=2, hazard=h)
                f = LatentCauseSMC(emit=[0, 1], cues=[0], us=2, hazard=h, n=40000, seed=seed)
                for x, u in s[:-1]:
                    f.observe(x, u)
                self.assertAlmostEqual(f.predict_us(s[-1][0]), exact, delta=.01,
                                       msg=f"seed {seed} h {h}")


if __name__ == "__main__":
    unittest.main()
