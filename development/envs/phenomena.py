"""Tier T2: conditioning phenomena translated into the agent interface.

The translations are fixed here before any agent is measured: six in S0-2, ABC renewal in
S3-0, spontaneous recovery with the retention interval outside the context in S3-1 (the v0
rows are kept unchanged).

Channels: 0 A, 1 B, 2 X, 3 Y (cues), 4 ctx1, 5 ctx2 (tonic contexts), 6 US,
7 value (constant 0). Two actions with no effect on the world.

A trial is: an inter-trial interval of 3-6 steps with only the context on, one
step with the CS (cues + context), then one step with the context and the US if
reinforced. Free USs (contingency degradation) occur in ITI steps.

Readout ("probe"): the agent's predicted P(US on the next step | CS + context
now), averaged over the two actions, read without learning, after 15 steps of
context alone so that no expectation window is open. Effects are differences of
probes, with a predicted sign from the animal literature; a phenomenon is
reproduced when the 95 % CI of the effect over seeds lies above 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A, B, X, Y, CTX1, CTX2, US, VALUE = range(8)
CTX3, HOME = 8, 9             # set v1 only
CUES = (A, B, X, Y)
N_CH, N_ACT = 8, 2
N_CH_V1 = 10


class Schedule:
    def __init__(self, seed: int, n_ch: int = N_CH):
        self.n_ch = n_ch
        self.rng = np.random.default_rng(seed)
        self.obs = []                 # observation stream
        self.probes = []              # (index of the last observation seen, name, probe obs)

    def _o(self, cues=(), ctx=CTX1, us=False):
        x = [0.] * self.n_ch
        for c in cues:
            x[c] = 1.
        if ctx is not None:
            x[ctx] = 1.
        if us:
            x[US] = 1.
        self.obs.append(tuple(x))

    def iti(self, ctx=CTX1, p_free=0.):
        for _ in range(int(self.rng.integers(3, 7))):
            self._o(ctx=ctx, us=bool(self.rng.random() < p_free))

    def trial(self, cues, p_us, ctx=CTX1, p_free=0.):
        self.iti(ctx, p_free)
        self._o(cues, ctx)
        self._o(ctx=ctx, us=bool(self.rng.random() < p_us))

    def trials(self, cues, p_us, n, ctx=CTX1, p_free=0.):
        for _ in range(n):
            self.trial(cues, p_us, ctx, p_free)

    def mixed(self, kinds, n_each, ctx=CTX1):
        order = [k for k in range(len(kinds)) for _ in range(n_each)]
        self.rng.shuffle(order)
        for k in order:
            self.trial(kinds[k][0], kinds[k][1], ctx)

    def gap(self, n, ctx=CTX1, p_free=0.):
        for _ in range(n):
            self._o(ctx=ctx, us=bool(p_free) and bool(self.rng.random() < p_free))

    def probe(self, name, cues, ctx=CTX1):
        x = [0.] * self.n_ch
        for c in cues:
            x[c] = 1.
        if ctx is not None:
            x[ctx] = 1.
        self.probes.append((len(self.obs) - 1, name, tuple(x)))


@dataclass
class Phenomenon:
    name: str
    groups: tuple            # group names
    build: object            # (group, seed) -> Schedule
    effect: object           # {group: {probe: value}} -> effect
    expected: str            # what the literature reports
    normative: str           # a normative account that predicts it
    min_effect: float | None = None   # row-specific bar (None: the set's)


def _blocking(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.trials([A] if group == "blocking" else [X], 1., 30)
    s.trials([A, B], 1., 20)
    s.gap(15)
    s.probe("B", [B])
    return s


def _inhibition(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.mixed([([A], 1.), ([A, X], 0.), ([B], 1.)], 30)
    s.gap(15)
    s.probe("BX", [B, X])
    s.probe("BY", [B, Y])           # Y: a novel cue (controls for novelty)
    return s


def _contingency(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.trials([A], .5, 60, p_free=.5 if group == "degraded" else 0.)
    s.gap(15)
    s.probe("A", [A])
    s.probe("ctx", [])
    return s


def _renewal(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.trials([A], 1., 30, ctx=CTX1)
    s.trials([A], 0., 30, ctx=CTX2)
    test = CTX1 if group == "ABA" else CTX2
    s.gap(15, ctx=test)
    s.probe("A", [A], ctx=test)
    return s


def _renewal_abc(group, seed, n=N_CH):
    """S3-0: test in a novel context (no context channel) vs the extinction context."""
    s = Schedule(seed, n)
    s.trials([A], 1., 30, ctx=CTX1)
    s.trials([A], 0., 30, ctx=CTX2)
    test = None if group == "ABC" else CTX2
    s.gap(15, ctx=test)
    s.probe("A", [A], ctx=test)
    return s


def _spontaneous(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.trials([A], 1., 30)
    s.trials([A], 0., 30)
    s.gap(15)
    s.probe("immediate", [A])
    s.gap(285)
    s.probe("delayed", [A])
    return s


def _spontaneous_home(group, seed, n=N_CH):
    """S3-1: the retention interval is spent outside the conditioning context (Pavlov's
    animals were returned to the home cage): no context channel during the long gap."""
    s = Schedule(seed, n)
    s.trials([A], 1., 30)
    s.trials([A], 0., 30)
    s.gap(15)
    s.probe("immediate", [A])
    s.gap(270, ctx=None)
    s.gap(15)
    s.probe("delayed", [A])
    return s


def _savings(group, seed, n=N_CH):
    s = Schedule(seed, n)
    s.trials([A], 1., 2)
    s.gap(15)
    s.probe("acq2", [A])
    s.trials([A], 1., 28)
    s.trials([A], 0., 30)
    s.trials([A], 1., 2)
    s.gap(15)
    s.probe("reacq2", [A])
    return s


# ------------------------------------------------------------------ set v1 (W-1)
def _contingency_v1(group, seed, n=N_CH_V1):
    """The pre-probe gap keeps the group's free-US rate (v0's gap extinguished the context)."""
    p = .5 if group == "degraded" else 0.
    s = Schedule(seed, n)
    s.trials([A], .5, 60, p_free=p)
    s.gap(15, p_free=p)
    s.probe("A", [A])
    s.probe("ctx", [])
    return s


def _renewal_abc_v1(group, seed, n=N_CH_V1):
    """Test in a never-seen context channel (ABC) vs the extinction context (ABB)."""
    s = Schedule(seed, n)
    s.trials([A], 1., 30, ctx=CTX1)
    s.trials([A], 0., 30, ctx=CTX2)
    test = CTX3 if group == "ABC" else CTX2
    s.gap(15, ctx=test)
    s.probe("A", [A], ctx=test)
    return s


def _sr_sessions(group, seed, n=N_CH_V1):
    """Sessions of 10 trials, each preceded by 20 steps at home; the retention interval at
    home is 20 (short) or 300 (long) steps; both groups return to the box for 15 steps."""
    s = Schedule(seed, n)
    for p in (1., 0.):
        for _ in range(3):
            s.gap(20, ctx=HOME)
            s.trials([A], p, 10)
    s.gap(15)
    s.probe("end", [A])
    s.gap(20 if group == "short" else 300, ctx=HOME)
    s.gap(15)
    s.probe("test", [A])
    return s


def _reinstatement(group, seed, n=N_CH_V1):
    """After extinction, 4 unsignalled USs in the context (or none) over the same ITIs; the
    interval before the test is spent at home (context exposure after the USs abolishes
    reinstatement in rats, Bouton & Bolles 1979, so it must not be part of the procedure)."""
    s = Schedule(seed, n)
    s.trials([A], 1., 30)
    s.trials([A], 0., 30)
    for _ in range(4):
        s.iti()
        s._o(ctx=CTX1, us=group == "reinst")
    s.gap(15, ctx=HOME)
    s.probe("A", [A])
    return s


def _pree(group, seed, n=N_CH_V1):
    """40 acquisition trials, continuous (p = 1) or partial (p = .5), then 6 unreinforced."""
    s = Schedule(seed, n)
    s.trials([A], 1. if group == "continuous" else .5, 40)
    s.trials([A], 0., 6)
    s.gap(15)
    s.probe("A", [A])
    return s


PHENOMENA = {p.name: p for p in [
    Phenomenon("blocking", ("blocking", "control"), _blocking,
               lambda r: r["control"]["B"] - r["blocking"]["B"],
               "pretraining A blocks learning about B in AB+ (Kamin 1969)",
               "explaining away in Bayesian cue combination (Courville et al. 2006)"),
    Phenomenon("cond-inhibition", ("one",), _inhibition,
               lambda r: r["one"]["BY"] - r["one"]["BX"],
               "X from A+/AX- reduces responding to another excitor B (summation test)",
               "negative causal power / inhibitor in the generative model"),
    Phenomenon("contingency", ("contingent", "degraded"), _contingency,
               lambda r: (r["contingent"]["A"] - r["contingent"]["ctx"])
               - (r["degraded"]["A"] - r["degraded"]["ctx"]),
               "free USs at the CS rate abolish conditioned responding (Rescorla 1968)",
               "Delta-P / causal power (Cheng 1997); readout is P(US|A) - P(US|ctx)"),
    Phenomenon("renewal-ABA", ("ABA", "ABB"), _renewal,
               lambda r: r["ABA"]["A"] - r["ABB"]["A"],
               "extinguished responding returns in the acquisition context (Bouton)",
               "latent-cause inference (Gershman, Blei & Niv 2010)"),
    Phenomenon("renewal-ABC", ("ABC", "ABB"), _renewal_abc,
               lambda r: r["ABC"]["A"] - r["ABB"]["A"],
               "extinguished responding returns in a novel context, less than in ABA (Bouton)",
               "latent-cause inference: a novel context is not the extinction cause"),
    Phenomenon("spontaneous-recovery", ("one",), _spontaneous,
               lambda r: r["one"]["delayed"] - r["one"]["immediate"],
               "extinguished responding partly returns with time (Pavlov)",
               "latent causes with a temporal prior; contextual inference (COIN)"),
    Phenomenon("spont-recovery-home", ("one",), _spontaneous_home,
               lambda r: r["one"]["delayed"] - r["one"]["immediate"],
               "extinguished responding partly returns with time (Pavlov; interval in the home cage)",
               "contextual inference: the belief drifts to the stationary context distribution"),
    Phenomenon("savings", ("one",), _savings,
               lambda r: r["one"]["reacq2"] - r["one"]["acq2"],
               "reacquisition after extinction is faster than acquisition",
               "latent-cause / contextual inference (COIN, Heald et al. 2021)"),
]}


def _v1(fn):
    return lambda group, seed: fn(group, seed, N_CH_V1)


_V0 = PHENOMENA
PHENOMENA_V1 = {p.name: p for p in [
    Phenomenon("blocking", ("blocking", "control"), _v1(_blocking), _V0["blocking"].effect,
               _V0["blocking"].expected, _V0["blocking"].normative),
    Phenomenon("cond-inhibition", ("one",), _v1(_inhibition), _V0["cond-inhibition"].effect,
               _V0["cond-inhibition"].expected, _V0["cond-inhibition"].normative),
    Phenomenon("contingency", ("contingent", "degraded"), _contingency_v1,
               _V0["contingency"].effect, _V0["contingency"].expected,
               _V0["contingency"].normative),
    Phenomenon("renewal-ABA", ("ABA", "ABB"), _v1(_renewal), _V0["renewal-ABA"].effect,
               _V0["renewal-ABA"].expected, _V0["renewal-ABA"].normative),
    Phenomenon("renewal-ABC", ("ABC", "ABB"), _renewal_abc_v1, _V0["renewal-ABC"].effect,
               _V0["renewal-ABC"].expected, _V0["renewal-ABC"].normative),
    Phenomenon("spontaneous-recovery", ("one",), _v1(_spontaneous),
               _V0["spontaneous-recovery"].effect, _V0["spontaneous-recovery"].expected,
               _V0["spontaneous-recovery"].normative),
    Phenomenon("spont-recovery-sessions", ("short", "long"), _sr_sessions,
               lambda r: r["long"]["test"] - r["short"]["test"],
               "recovery grows with the retention interval (Pavlov; Rescorla 2004)",
               "the unobserved regime of the box keeps evolving while the animal is away"),
    Phenomenon("savings", ("one",), _v1(_savings), _V0["savings"].effect,
               _V0["savings"].expected, _V0["savings"].normative,
               min_effect=0.),   # direction only: the reference reaches +.048 (W-1, user: (a))
    Phenomenon("reinstatement", ("reinst", "control"), _reinstatement,
               lambda r: r["reinst"]["A"] - r["control"]["A"],
               "unsignalled USs after extinction restore responding (Rescorla & Heth 1975)",
               "the USs are evidence for the acquisition context / a reinforcing cause"),
    Phenomenon("pree", ("partial", "continuous"), _pree,
               lambda r: r["partial"]["A"] - r["continuous"]["A"],
               "extinction is slower after partial reinforcement (Humphreys 1939)",
               "a change from p = .5 to 0 is harder to detect than from 1 to 0"),
]}
# ------------------------------------------------------------------ set h1 (D-23, AR-3)
# Held-out phenomena for models designed on v1 (the lc-* family and agents built on it). Fixed
# here before any agent or latent-cause witness is run on them; only the references rw and cfg
# are run as a stress test of the translations. Same channels, trial structure and readout as v1.

def _overshadowing(group, seed, n=N_CH_V1):
    """B trained in compound with A (AB+) vs B trained alone (B+), same number of trials."""
    s = Schedule(seed, n)
    s.trials([A, B] if group == "overshadow" else [B], 1., 20)
    s.gap(15)
    s.probe("B", [B])
    return s


def _latent_inhibition(group, seed, n=N_CH_V1):
    """30 non-reinforced pre-exposures of A (or of another cue X), then 3 A+ trials."""
    s = Schedule(seed, n)
    s.trials([A] if group == "preexposed" else [X], 0., 30)
    s.trials([A], 1., 3)
    s.gap(15)
    s.probe("A", [A])
    return s


def _negative_patterning(group, seed, n=N_CH_V1):
    """A+, B+, AB- intermixed (30 each)."""
    s = Schedule(seed, n)
    s.mixed([([A], 1.), ([B], 1.), ([A, B], 0.)], 30)
    s.gap(15)
    s.probe("A", [A])
    s.probe("B", [B])
    s.probe("AB", [A, B])
    return s


def _overexpectation(group, seed, n=N_CH_V1):
    """A+ and B+ separately (20 each), then AB+ (20) vs 10 more of each element alone."""
    s = Schedule(seed, n)
    s.mixed([([A], 1.), ([B], 1.)], 20)
    if group == "overexp":
        s.trials([A, B], 1., 20)
    else:
        s.mixed([([A], 1.), ([B], 1.)], 10)
    s.gap(15)
    s.probe("A", [A])
    return s


def _renewal_aab(group, seed, n=N_CH_V1):
    """Acquisition and extinction in the same context; test in a never-seen context (AAB) vs
    the training context (AAA)."""
    s = Schedule(seed, n)
    s.trials([A], 1., 30, ctx=CTX1)
    s.trials([A], 0., 30, ctx=CTX1)
    test = CTX3 if group == "AAB" else CTX1
    s.gap(15, ctx=test)
    s.probe("A", [A], ctx=test)
    return s


def _inhibitor_alone(group, seed, n=N_CH_V1):
    """A+ / AX- / B+ (30 each), then 30 presentations of X alone without the US; summation
    test BX vs BY."""
    s = Schedule(seed, n)
    s.mixed([([A], 1.), ([A, X], 0.), ([B], 1.)], 30)
    s.trials([X], 0., 30)
    s.gap(15)
    s.probe("BX", [B, X])
    s.probe("BY", [B, Y])
    return s


PHENOMENA_H1 = {p.name: p for p in [
    Phenomenon("overshadowing", ("overshadow", "control"), _overshadowing,
               lambda r: r["control"]["B"] - r["overshadow"]["B"],
               "a cue conditioned in compound acquires less than alone (Pavlov 1927)",
               "credit assignment among cues (Rescorla-Wagner; Bayesian cue combination)"),
    Phenomenon("latent-inhibition", ("preexposed", "control"), _latent_inhibition,
               lambda r: r["control"]["A"] - r["preexposed"]["A"],
               "non-reinforced pre-exposure retards later conditioning (Lubow & Moore 1959)",
               "prior evidence that the cue predicts nothing (Gershman, Blei & Niv 2010)"),
    Phenomenon("negative-patterning", ("one",), _negative_patterning,
               lambda r: (r["one"]["A"] + r["one"]["B"]) / 2 - r["one"]["AB"],
               "A+, B+, AB- is solved: the compound is predicted less than its elements",
               "configural representation (Pearce 1987); structure learning (Courville 2006)"),
    Phenomenon("overexpectation", ("overexp", "control"), _overexpectation,
               lambda r: r["control"]["A"] - r["overexp"]["A"],
               "reinforcing a compound of two excitors reduces each (Kremer 1978)",
               "additive prediction error (Rescorla-Wagner; Kalman / Bayesian regression)"),
    Phenomenon("renewal-AAB", ("AAB", "AAA"), _renewal_aab,
               lambda r: r["AAB"]["A"] - r["AAA"]["A"],
               "extinguished responding returns in a novel context after same-context "
               "training and extinction (Bouton & Ricker 1994)",
               "extinction learning is context-specific; latent-cause inference"),
    Phenomenon("inhibitor-alone", ("one",), _inhibitor_alone,
               lambda r: r["one"]["BY"] - r["one"]["BX"],
               "non-reinforced X alone does not extinguish conditioned inhibition "
               "(Zimmer-Hart & Rescorla 1974)",
               "X- alone is no evidence against X as an inhibitor (no excitation to cancel)"),
]}

# ------------------------------------------------------------------ set h2 (D-30, AR-3)
# Second held-out set, fixed before any candidate is run on it (h1 was used to develop F-11).
# Rows probe context x time; only the references rw and cfg are run as a translation test.

def _li_context(group, seed, n=N_CH_V1):
    """30 non-reinforced pre-exposures of A in ctx2 (same) or ctx1 (switch), then 3 A+ in ctx2."""
    s = Schedule(seed, n)
    s.trials([A], 0., 30, ctx=CTX2 if group == "same" else CTX1)
    s.trials([A], 1., 3, ctx=CTX2)
    s.gap(15, ctx=CTX2)
    s.probe("A", [A], ctx=CTX2)
    return s


def _sensory_precond(group, seed, n=N_CH_V1):
    """AB- compound pre-exposure (paired) or AX- (control: A equally familiar, only in compound,
    with another partner; Rescorla & Cunningham 1978), then B+; test A. (An unpaired A-, B-
    control confounds the row with the novelty of A alone — D-30 fix.)"""
    s = Schedule(seed, n)
    s.trials([A, B] if group == "paired" else [A, X], 0., 20)
    s.trials([B], 1., 10)
    s.gap(15)
    s.probe("A", [A])
    return s


def _us_preexposure(group, seed, n=N_CH_V1):
    """100 context steps with free USs (p = .2) or without, then 5 A+ trials; A is tested in
    the neutral ctx2 so that the training context's own US association does not enter the
    readout. (Subtracting a context-alone probe lets any model whose context predicts the US
    pass without a CS effect — D-30 fix.)"""
    s = Schedule(seed, n)
    s.gap(100, p_free=.2 if group == "preexposed" else 0.)
    s.trials([A], 1., 5)
    s.gap(15, ctx=CTX2)
    s.probe("A", [A], ctx=CTX2)
    return s


def _superconditioning(group, seed, n=N_CH_V1):
    """A+ / AX- / Y- (30 each) makes X an inhibitor and Y a familiar neutral cue; then B is
    conditioned in compound with X (super) or Y (control)."""
    s = Schedule(seed, n)
    s.mixed([([A], 1.), ([A, X], 0.), ([Y], 0.)], 30)
    s.trials([X, B] if group == "super" else [Y, B], 1., 10)
    s.gap(15)
    s.probe("B", [B])
    return s


def _reinstatement_context(group, seed, n=N_CH_V1):
    """As v1 reinstatement, with the 4 free USs in the extinction context (same) or in ctx2.
    The animal starts from its home cage (20 home steps), so home is familiar at the test as in
    rats (D-30 amendment 12:25, before any candidate was run on h2)."""
    s = Schedule(seed, n)
    s.gap(20, ctx=HOME)
    s.trials([A], 1., 30)
    s.trials([A], 0., 30)
    where = CTX1 if group == "same" else CTX2
    for _ in range(4):
        s.iti(ctx=where)
        s._o(ctx=where, us=True)
    s.gap(15, ctx=HOME)
    s.probe("A", [A])
    return s


PHENOMENA_H2 = {p.name: p for p in [
    Phenomenon("LI-context", ("switch", "same"), _li_context,
               lambda r: r["switch"]["A"] - r["same"]["A"],
               "latent inhibition is attenuated when pre-exposure and conditioning contexts "
               "differ (Hall & Channell 1985)",
               "pre-exposure is evidence about the cue in its context / latent cause"),
    Phenomenon("sensory-precond", ("paired", "control"), _sensory_precond,
               lambda r: r["paired"]["A"] - r["control"]["A"],
               "after AB pre-exposure, conditioning B transfers to A (Brogden 1939; "
               "Rescorla & Cunningham 1978)",
               "A and B are inferred to share a cause (Courville et al. 2006)"),
    Phenomenon("US-preexposure", ("control", "preexposed"), _us_preexposure,
               lambda r: r["control"]["A"] - r["preexposed"]["A"],
               "US exposure in the context retards conditioning of a CS (Randich & LoLordo 1979)",
               "the context already predicts the US and blocks the CS (Rescorla-Wagner)"),
    Phenomenon("superconditioning", ("super", "control"), _superconditioning,
               lambda r: r["super"]["B"] - r["control"]["B"],
               "a CS conditioned with an inhibitor gains more than with a neutral cue "
               "(Rescorla 1971)",
               "additive prediction error: the inhibitor enlarges the error (Rescorla-Wagner)"),
    Phenomenon("reinstatement-context", ("same", "different"), _reinstatement_context,
               lambda r: r["same"]["A"] - r["different"]["A"],
               "reinstatement requires the USs in the test context (Bouton & Bolles 1979)",
               "the context's own US association summates with the CS"),
]}

SETS = {"v0": (PHENOMENA, N_CH, .02), "v1": (PHENOMENA_V1, N_CH_V1, .05),
        "h1": (PHENOMENA_H1, N_CH_V1, .05), "h2": (PHENOMENA_H2, N_CH_V1, .05)}
