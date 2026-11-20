"""D-17 / D-18 witness: sticky-CRP causes x structure average (+ additive member).

Small synthetic designs with anonymous channels A, B, X, Y, ctx and the US. Each trial is one
step with the cues on followed by an empty step (inter-trial interval)."""
import unittest

import numpy as np

from models.latent_cause_lin import LatentCauseSub, LatentCauseSubAdd

A, B, X, Y, CTX, US = range(6)


def vec(*on):
    x = np.zeros(6)
    x[list(on)] = 1.
    return x


def run(model, trials):
    for cues, us in trials:
        model.observe(vec(CTX, *cues), us)
        model.observe(vec(CTX), False)
    return model


def make(cls):
    return cls(emit=[], cues=[A, B, X, Y, CTX], us=US, hazard=None, n=200, seed=0)


def p(model, *cues):
    return model.predict_us(vec(CTX, *cues))


class TestStructureAverage(unittest.TestCase):
    def test_blocking(self):
        """B gains less when A already predicts the US (Kamin blocking)."""
        blk = run(make(LatentCauseSub), [((A,), True)] * 30 + [((A, B), True)] * 20)
        ctl = run(make(LatentCauseSub), [((X,), False)] * 30 + [((A, B), True)] * 20)
        self.assertGreater(p(ctl, B) - p(blk, B), .2, (p(blk, B), p(ctl, B)))

    def test_conditioned_inhibition_needs_the_additive_member(self):
        """A+ / AX- / B+, summation test BX vs BY (Y novel): X inhibits only when the elemental
        hypothesis is a member of the average; with configural subsets BX and BY are equally
        novel compounds."""
        trials = [((A,), True), ((A, X), False), ((B,), True)] * 25
        add = run(make(LatentCauseSubAdd), trials)
        sub = run(make(LatentCauseSub), trials)
        self.assertGreater(p(add, B, Y) - p(add, B, X), .3)
        self.assertLess(abs(p(sub, B, Y) - p(sub, B, X)), .1)

    def test_savings(self):
        """After acquisition and extinction, re-acquisition is faster than acquisition: the
        acquisition cause is kept in memory and re-entered."""
        m = make(LatentCauseSubAdd)
        run(m, [((A,), True)] * 3)
        first = p(m, A)
        run(m, [((A,), True)] * 30 + [((A,), False)] * 40 + [((A,), True)] * 3)
        self.assertGreater(p(m, A), first)


if __name__ == "__main__":
    unittest.main()


class TestMarkovContext(unittest.TestCase):
    """F-11 (D-28): a never-seen tonic context is evidence for a new cause (AAB-like renewal),
    which the parent without context emission cannot show."""

    def test_novel_context_renews(self):
        import numpy as np
        from models.latent_cause_lin import LatentCauseSubAdd, LatentCauseSubAddM

        def x(a, c1, c3):
            return np.array([a, c1, c3, 0.])

        def run(cls, emit, seed):
            lc = cls(emit=emit, cues=[0, 1, 2], us=-1, hazard=None, n=200, seed=seed)
            for t in range(60):
                lc.observe(x(t % 3 == 0, 1, 0), t % 3 == 0)
            for t in range(60):
                lc.observe(x(t % 3 == 0, 1, 0), False)
            lc.observe(x(0, 0, 1), False)
            return lc.predict_us(x(1, 0, 1))

        for seed in (0, 1):
            self.assertGreater(run(LatentCauseSubAddM, [1, 2], seed),
                               run(LatentCauseSubAdd, [], seed) + .05)


class TestPlaceRegime(unittest.TestCase):
    """F-12 (D-31): a never-visited place redraws the regime (AAB-like renewal)."""

    def test_novel_place_renews(self):
        import numpy as np
        from models.latent_cause_lin import LatentCauseSubAdd, LatentCauseSubAddP

        def x(a, c1, c3):
            return np.array([a, c1, c3, 0.])

        def run(cls, emit, seed):
            lc = cls(emit=emit, cues=[0, 1, 2], us=-1, hazard=None, n=200, seed=seed)
            for t in range(60):
                lc.observe(x(t % 3 == 0, 1, 0), t % 3 == 0)
            for t in range(60):
                lc.observe(x(t % 3 == 0, 1, 0), False)
            lc.observe(x(0, 0, 1), False)
            return lc.predict_us(x(1, 0, 1))

        for seed in (0, 1):
            self.assertGreater(run(LatentCauseSubAddP, [0, 1, 2], seed),
                               run(LatentCauseSubAdd, [], seed) + .05)
