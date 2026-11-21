"""Design-locked AR-5 contrasts. Do not execute a candidate before its freeze.

The designated reference agents may test schedule properties, separately from
candidate development. The effect summaries are paired by schedule seed. The
US is anonymous, as in v1; this test is prediction, not conditioned performance.
"""
from __future__ import annotations

from envs.phenomena import A, B, X, Y, CTX1, CTX2, HOME, Schedule, Phenomenon

SEEDS = tuple(range(52000, 52024))
FLOOR = .05


def _familiar(s):
    for c in (HOME, CTX1, CTX2):
        s.gap(50, ctx=c)


def mediated(group, seed):
    # Equal novelty at test. Both groups see the same A+B and Y+X compounds.
    # Only which associate is reinforced changes; partner alone is a check.
    s = Schedule(seed)
    _familiar(s)
    s.mixed([([A, B], 0.), ([Y, X], 0.)], 24)
    s.trials([B] if group == 'linked' else [X], 1., 12)
    s.gap(12)
    s.probe('A', [A]); s.probe('Y', [Y]); s.probe('B', [B]); s.probe('X', [X])
    return s


def background(group, seed):
    # Equal free-outcome history, cue training and neutral-place test. Only
    # whether background excitation is present during A training differs.
    s = Schedule(seed)
    _familiar(s)
    s.gap(120, CTX1 if group == 'same' else CTX2, .25)
    s.trials([A], 1., 8, ctx=CTX1)
    s.gap(30, HOME)
    s.probe('A', [A], HOME); s.probe('base', [], HOME)
    return s


def inhibition_transfer(group, seed):
    # A+ AX- makes X an inhibitor; AY+ makes Y familiar but can make it a weak
    # excitor. Also present Y alone unreinforced, and require a summation check
    # against neutral transfer before interpreting a B-credit contrast.
    s = Schedule(seed)
    _familiar(s)
    s.mixed([([A], 1.), ([A, X], 0.), ([Y], 0.), ([A, Y], 1.)], 28)
    s.trials([B, X if group == 'inhibitor' else Y], 1., 7)
    s.gap(12)
    s.probe('B', [B]); s.probe('BX', [B, X]); s.probe('BY', [B, Y])
    return s


def omission_vs_dwell(group, seed):
    s = Schedule(seed)
    _familiar(s)
    partial, omitted = group.split('-')
    s.trials([A], .5 if partial == 'partial' else 1., 36)
    # Place exposure and opportunity duration matched; only A omissions vary.
    for _ in range(8):
        s.iti()
        s._o([A] if omitted == 'omit' else [])
        s._o()
    s.gap(12)
    s.probe('A', [A])
    return s


def return_memory(group, seed):
    s = Schedule(seed)
    _familiar(s)
    s.trials([A], 1., 28, ctx=CTX1)
    s.trials([A], 0., 28, ctx=CTX2)
    s.gap(25, HOME)
    context = CTX1 if group == 'acquisition' else CTX2
    s.probe('A', [A], context); s.probe('base', [], context)
    return s


def _increment(r, g):
    return r[g]['A'] - r[g]['base']


def _drop(r, prefix):
    return r[prefix + '-dwell']['A'] - r[prefix + '-omit']['A']


ROWS = {p.name: p for p in (
    Phenomenon('associate-crossover', ('linked', 'alternate'), mediated,
               lambda r: ((r['linked']['A'] - r['linked']['Y']) -
                          (r['alternate']['A'] - r['alternate']['Y'])) / 2,
               'sensory preconditioning with reciprocal transfer control', 'mediated cue credit'),
    Phenomenon('background-credit', ('other', 'same'), background,
               lambda r: _increment(r, 'other') - _increment(r, 'same'),
               'context-US competition, neutral test of incremental CS', 'place-owned baseline'),
    Phenomenon('inhibitor-credit', ('inhibitor', 'control'), inhibition_transfer,
               lambda r: r['inhibitor']['B'] - r['control']['B'],
               'superconditioning with familiar inhibitor/control', 'signed residual credit'),
    Phenomenon('omission-by-history', ('partial-omit', 'partial-dwell', 'full-omit', 'full-dwell'),
               omission_vs_dwell, lambda r: _drop(r, 'full') - _drop(r, 'partial'),
               'PREE relative to matched cue-free exposure', 'contingency evidence, not silence'),
    Phenomenon('return-specificity', ('acquisition', 'extinction'), return_memory,
               lambda r: _increment(r, 'acquisition') - _increment(r, 'extinction'),
               'context-specific retrieval after reversal', 'place/contingency separation'),
)}
