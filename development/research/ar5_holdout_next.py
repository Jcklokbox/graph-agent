"""Second independent schedule registry; not a replacement or retest of D-40.

Build schedules from qualitative causal interventions, without model reads.
Only explicit schedule controls and noncandidate reference predictions can be
used for the design audit. Freeze this source and the seed range before any
candidate is run; abandon the registry if a design control is invalid.
"""
from __future__ import annotations

import numpy as np

from envs.phenomena import A, B, X, Y, CTX1, CTX2, HOME, N_CH_V1, Schedule, Phenomenon

SEEDS = tuple(range(83100, 83124))
FLOOR = .05
Z = N_CH_V1
N = N_CH_V1 + 1


def _base(seed):
    s = Schedule(seed, N)
    for place in (HOME, CTX1, CTX2):
        s.gap(64, place)
    return s


def _swap(seed, a, b):
    return (a, b) if seed % 2 == 0 else (b, a)


def compensated(group, seed):
    """With X still inhibitory, a new B needs larger positive credit than D."""
    s = _base(seed)
    b, d = _swap(seed, B, Z)
    inhib, neutral = _swap(seed // 2, X, Y)
    anchor = [([A], .80), ([A, inhib], .12), ([A, neutral], .80), ([neutral], .04)]
    s.mixed(anchor, 30)
    s.probe('AX-before', [A,inhib]); s.probe('AY-before', [A,neutral])
    for _ in range(4):
        s.mixed(anchor + [([b, inhib], .55), ([d, neutral], .55)], 7)
    s.gap(14)
    for name, cues in [('B',[b]), ('D',[d]), ('AX',[A,inhib]), ('AY',[A,neutral]),
                       ('BX',[b,inhib]), ('DY',[d,neutral])]:
        s.probe(name, cues)
    return s


def noisy_or(group, seed):
    """Different known tonic hazard, matched compound hazard, neutral cue test."""
    s = _base(seed)
    hi, lo = _swap(seed, CTX1, CTX2)
    ch, cl = _swap(seed // 2, A, B)
    # Reverse block order by arm; p_joint=.64, so implied cue risks .486/.629.
    per_block = [(hi, ch, .30), (lo, cl, .03)]
    for _ in range(3):
        for place, cue, bg in (per_block if group == 'hl' else per_block[::-1]):
            s.gap(80, place, bg)
            s.trials([cue], .64, 18, ctx=place, p_free=bg)
            s.gap(24, place, bg)
            s.probe('hi-bg' if cue==ch else 'lo-bg', [], place)
    s.gap(34, HOME)
    s.probe('hi', [ch], HOME); s.probe('lo', [cl], HOME); s.probe('base', [], HOME)
    return s


def interaction(group, seed):
    """The same two cues have opposite meanings in two equally familiar places."""
    s = _base(seed)
    a, b = _swap(seed, A, B)
    c, d = _swap(seed//2, CTX1, CTX2)
    for _ in range(3):
        s.mixed([([a], .85), ([b], .15)], 12, ctx=c)
        s.mixed([([b], .85), ([a], .15)], 12, ctx=d)
    s.gap(26, HOME)
    for name, place in [('left', c), ('right',d)]:
        s.probe(name+'-a', [a], place); s.probe(name+'-b', [b], place)
    return s


def place_intervention(group, seed):
    """Equal outcome exposure in crossed neutral backgrounds after extinguishing both cues."""
    s = _base(seed)
    a, b = _swap(seed, A, B)
    for cue, place in ((a,CTX1),(b,CTX2)):
        s.trials([cue], .9, 26, ctx=place)
        s.trials([cue], 0., 24, ctx=place)
    for place in (CTX1, CTX2):
        s.gap(80, place, .24 if group == ('first' if place==CTX1 else 'second') else 0.)
    s.gap(16, HOME)
    for label,cue,place in [('first',a,CTX1),('second',b,CTX2)]:
        s.probe(label, [cue], place); s.probe(label+'-base', [], place)
    return s


def temporal_associate(group, seed):
    """Forward sensory prediction with yoked exchanged identities."""
    s = _base(seed)
    a, b = _swap(seed, A, B)
    cue, other = _swap(seed//2, X, Y)
    rng = np.random.default_rng(seed ^ 0x39A4)
    paired = np.repeat([False, True], 32)
    rng.shuffle(paired)
    for i in paired:
        s.iti()
        s._o([a] if i else [b])
        s._o([cue] if bool(i) == (group=='aligned') else [other])
        s._o()
    s.trials([cue], .92, 24)
    s.gap(18)
    s.probe('a',[a]); s.probe('b',[b]); s.probe('train',[cue]); s.probe('other',[other])
    return s


def compensation(r):
    return r['paired']['B'] - r['paired']['D']


def background(r):
    return .5 * sum(r[g]['lo']-r[g]['hi'] for g in ('hl','lh'))


def place_effect(r):
    def surplus(g,p): return r[g][p]-r[g][p+'-base']
    return .5 * (surplus('first','first')-surplus('first','second') +
                 surplus('second','second')-surplus('second','first'))


ROWS = {p.name: p for p in (
    Phenomenon('compensated-inhibitor', ('paired',), compensated, compensation,
               'negative credit with maintained/no reversal control', 'signed extrapolation'),
    Phenomenon('competing-hazards', ('hl','lh'), noisy_or, background,
               'joint hazard matched below ceiling, tonic hazard crossed','tonic/cue credit'),
    Phenomenon('context-interaction', ('both',), interaction,
               lambda r:.5*(r['both']['left-a']-r['both']['left-b']+r['both']['right-b']-r['both']['right-a']),
               'opposite contingency in equal contexts','contextual retrieval'),
    Phenomenon('source-selective-return', ('first','second'), place_intervention, place_effect,
               'place-selective free outcome and subtract local baseline','regime activation'),
    Phenomenon('forward-associate', ('aligned','exchanged'), temporal_associate,
               lambda r:.5*(r['aligned']['a']-r['aligned']['b']+r['exchanged']['b']-r['exchanged']['a']),
               'forward sensory relation with exchanged identities','temporal transfer'),
)}


def controls(r):
    p = r['compensated-inhibitor']['paired']
    b = r['competing-hazards']
    return dict(inhib_before=p['AY-before']-p['AX-before'], inhib_after=p['AY']-p['AX'],
                baseline=.5 * sum(b[g]['hi-bg']-b[g]['lo-bg'] for g in ('hl','lh')),
                direct=.5 * sum(r['forward-associate'][g]['train']-r['forward-associate'][g]['other']
                                for g in ('aligned','exchanged')))
