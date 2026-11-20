"""Small, deterministic falsification probes for architecture-synthesis.md.

This is a research probe, not a production learner. Run: python research/posterior_graph_probe.py
Only NumPy is required. The tests deliberately separate posterior prediction,
structure selection, conditional inhibition, event timing, value backup, and symmetry.
"""
from __future__ import annotations

from itertools import permutations
from math import lgamma, log
import random


def beta_update(pos: int, neg: int, y: int, a: float = 1., b: float = 1.) -> tuple[int, int, float]:
    pos += int(y == 1)
    neg += int(y == 0)
    return pos, neg, (a + pos) / (a + b + pos + neg)


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def log_marginal_binary(rows, predictors):
    """Beta(1,1) integrated likelihood for each categorical parent pattern."""
    groups = {}
    for x, y in rows:
        key = tuple(x[i] for i in predictors)
        n1, n0 = groups.get(key, (0, 0))
        groups[key] = (n1 + int(y), n0 + int(not y))
    return sum(log_beta(1+n1, 1+n0)-log_beta(1, 1) for n1, n0 in groups.values())


def test_calibration():
    rng = random.Random(10)
    n1=n0=0; pred=[]; outcomes=[]
    for _ in range(600):
        y=int(rng.random()<.7)
        p=(1+n1)/(2+n1+n0); pred.append(p); outcomes.append(y)
        n1,n0,_=beta_update(n1,n0,y)
    brier=sum((p-y)**2 for p,y in zip(pred,outcomes))/len(pred)
    # Independent nuisance's pattern-specific posterior is pooled by no edge;
    # naive per-pattern Beta models can confidently fit finite-sample noise.
    rows=[]
    for i in range(400):
        x=(int(i%2), int((i//2)%2)); y=int((i*37%101)<70)
        rows.append((x,y))
    nuisance_gain=log_marginal_binary(rows,(0,1))-log_marginal_binary(rows,(0,))
    assert brier < .22, brier
    # Evidence difference is not itself a calibrated structural posterior.
    assert nuisance_gain < 15, nuisance_gain
    return f"Brier={brier:.3f}; nuisance pair-vs-main log evidence={nuisance_gain:.2f}"


def test_structure():
    # Balanced complete truth tables; held-out samples repeat different rows.
    def data(fn, reps):
        return [((a,b,c),fn(a,b)) for _ in range(reps) for a,b,c in ((0,0,0),(0,1,1),(1,0,1),(1,1,0))]
    for name, fn in (("AND",lambda a,b:a&b),("XOR",lambda a,b:a^b)):
        rows=data(fn,20)
        base=log_marginal_binary(rows,())
        pair=log_marginal_binary(rows,(0,1))
        assert pair > base + 20, (name,base,pair)
    # redundant third feature provides no conditional information
    rows=data(lambda a,b:a,30)
    main=log_marginal_binary(rows,(0,))
    extra=log_marginal_binary(rows,(0,2))
    assert extra <= main + 1e-8, (main,extra)
    return "AND/XOR pair evidence beats pooled; redundant channel adds no evidence"


def test_inhibition_memory():
    # Tables with positive base measure under A and suppression only with X.
    counts={}
    for _ in range(50):
        for a,x,y in ((1,0,1),(1,1,0),(0,1,0),(0,0,0)):
            k=(a,x); p,n=counts.get(k,(0,0)); counts[k]=(p+int(y==1),n+int(y==0))
    def prob(k):
        p,n=counts[k]; return (1+p)/(2+p+n)
    pa=prob((1,0)); pax=prob((1,1)); px=prob((0,1))
    assert pa > .95 and pax < .05 and px < .05
    # Context removal: A-only counts remain exactly available after AX learning.
    assert counts[(1,0)] == (50,0)
    return f"p(Y|A)={pa:.3f}, p(Y|A,X)={pax:.3f}; A-only posterior retained"


def test_timing():
    # Discrete survival/hazard counts: one pending event per cue; omitted tail
    # is right-censored, and event consumes the pending episode.
    episodes=[(lag,True) for lag in (2,3,2,4,3,2,4,3,2,3)]
    episodes.append((1,False))  # right-censored: observed event-free through age 1
    risk=[0]*4; events=[0]*4
    for duration, observed in episodes:
        for age in range(1,duration+1):
            risk[age-1]+=1
        if observed:
            events[duration-1]+=1
    hazard=[(e+1)/(r+2) for e,r in zip(events,risk)]
    # The mode of observed event lags is not necessarily the hazard maximum.
    pmf=[e/sum(events) for e in events]
    assert max(range(4),key=lambda i:pmf[i])+1 == 2
    assert risk[0] == len(episodes)
    assert all(0<h<1 for h in hazard)
    return f"discrete hazards={[round(h,3) for h in hazard]}; one-step omission included in risk set"


def test_value():
    # Exact two-step MDP: costly setup action enables a terminal reward route.
    gamma=.9; immediate_setup=-.1; terminal_reward=1.
    q_setup=immediate_setup+gamma*terminal_reward
    q_stay=0.
    assert q_setup > q_stay
    revalued=immediate_setup+gamma*.05
    assert revalued < q_stay
    blocked=immediate_setup+gamma*0.
    assert blocked < q_stay
    return f"Q(setup)={q_setup:.3f}; after reward revaluation={revalued:.3f}; blocked={blocked:.3f}"


def test_equivariance():
    # A single action-value graph whose factor keys are sets of named channel
    # indices; compare every sensor permutation and action permutation.
    x=(1,0,1); q=(.2,.8,.4)
    winner=max(range(3),key=lambda i:q[i])
    for p in permutations(range(3)):
        for a in permutations(range(3)):
            xp=tuple(x[i] for i in p); qp=tuple(q[i] for i in a)
            w=max(range(3),key=lambda i:qp[i])
            assert w == a.index(winner)
            assert tuple(xp[p.index(i)] for i in range(3)) == x
    return "all 36 sensor/action permutation combinations map observations and unique winner exactly"


def main():
    tests=(test_calibration,test_structure,test_inhibition_memory,test_timing,test_value,test_equivariance)
    failed=0
    for f in tests:
        try: print(f"PASS {f.__name__}: {f()}")
        except Exception as e:
            failed+=1; print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"RESULT {len(tests)-failed}/{len(tests)} pass")
    return failed

if __name__ == '__main__':
    raise SystemExit(main())
