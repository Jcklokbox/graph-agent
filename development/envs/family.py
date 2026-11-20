"""Environment family v0: factored hidden-process worlds (FHP).

A world is a set of independent hidden Markov processes. Process k has a hidden
state s_k in {0..S_k-1}, an initial distribution, a transition matrix that may
depend on the action (controllable process) and on the state of an optional
hidden *context* process (slow, sticky), and owns a disjoint set of binary
channels, each emitted as Bernoulli(E_k[s_k, i]). Several states may emit the
same pattern (aliasing): delays, counters and hidden regimes are hidden state.

    x_{t+1, i} ~ Bernoulli(E_k[s_{k,t+1}, i])       i in channels(k)
    s_{k,t+1}  ~ T_k[c_t, a_t, s_{k,t}, :]           c_t = context state

Because the generative model is known, the ideal predictor for a tiny world is
the exact Bayes filter on the joint state (`Filter`): the floor that any
learner's next-observation log loss is compared against (tier T1). The current
three tasks of `envs/tasks.py` are structurally members of the family (a cue
process with a delay, an action-gated reward, noise processes, reversals) but
are not re-implemented here.

Every world has a value channel (the agent interface needs one); in the T1
worlds it carries a process like any other or is constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Process:
    name: str
    init: np.ndarray        # (S,)
    trans: np.ndarray       # (C, A, S, S): context state, action, from, to
    emit: np.ndarray        # (S, m): P(channel on | state)
    channels: list          # global indices of the m channels

    @property
    def n_states(self) -> int:
        return len(self.init)


@dataclass
class World:
    name: str
    n_channels: int
    n_actions: int
    value_index: int
    processes: list
    context: Process | None = None      # its trans is (1, 1, C, C); may own channels
    note: str = ""
    links: list = field(default_factory=list)   # shared-emission channels (D-7):
                                                # (channel, k1, k2, table[S_k1, S_k2])
    _joint: tuple | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- joint
    def joint(self):
        """(init (N,), trans (A, N, N), emit (N, n_channels)) on the joint state,
        joint index = context-major, then processes in order."""
        if self._joint is not None:
            return self._joint
        A = self.n_actions
        ctx = self.context
        C = ctx.n_states if ctx is not None else 1
        Pc = ctx.trans[0, 0] if ctx is not None else np.ones((1, 1))
        ic = ctx.init if ctx is not None else np.ones(1)
        sizes = [p.n_states for p in self.processes]
        R = int(np.prod(sizes))
        N = C * R
        if N > 5000:
            raise ValueError(f"joint state too large for the exact filter: {N}")
        init = ic[:, None] * _kron_all([p.init for p in self.processes])[None, :]
        trans = np.zeros((A, N, N))
        for a in range(A):
            for c in range(C):
                Tr = _kron_all([p.trans[min(c, p.trans.shape[0] - 1), a] for p in self.processes])
                for c2 in range(C):
                    trans[a, c * R:(c + 1) * R, c2 * R:(c2 + 1) * R] = Pc[c, c2] * Tr
        emit = np.zeros((N, self.n_channels))
        states = np.array(np.unravel_index(np.arange(R), sizes)).T if sizes else np.zeros((1, 0), int)
        for c in range(C):
            for k, p in enumerate(self.processes):
                for j, ch in enumerate(p.channels):
                    emit[c * R:(c + 1) * R, ch] = p.emit[states[:, k], j]
            if ctx is not None:
                for j, ch in enumerate(ctx.channels):
                    emit[c * R:(c + 1) * R, ch] = ctx.emit[c, j]
            for ch, k1, k2, tab in self.links:
                emit[c * R:(c + 1) * R, ch] = np.asarray(tab)[states[:, k1], states[:, k2]]
        self._joint = (init.reshape(-1), trans, emit)
        return self._joint

    def check(self) -> None:
        owned = [ch for p in self.processes + ([self.context] if self.context else [])
                 for ch in p.channels] + [lk[0] for lk in self.links]
        assert len(owned) == len(set(owned)), "channels must be owned by one process"
        assert all(0 <= ch < self.n_channels for ch in owned)
        for p in self.processes + ([self.context] if self.context else []):
            assert np.allclose(p.trans.sum(-1), 1.) and np.isclose(p.init.sum(), 1.)
            assert p.emit.shape == (p.n_states, len(p.channels))


def _kron_all(mats):
    out = np.ones((1,) * mats[0].ndim) if mats else np.ones(1)
    for m in mats:
        out = np.kron(out, m)
    return out


# ===================================================================== sim
class WorldEnv:
    """Samples a world. step(action) -> next observation (tuple of 0/1 floats)."""

    def __init__(self, world: World, seed: int):
        world.check()
        self.w = world
        self.rng = np.random.default_rng(seed)

    def _emit(self) -> tuple:
        x = np.zeros(self.w.n_channels)
        for p, s in zip(self.w.processes, self.s):
            x[p.channels] = self.rng.random(len(p.channels)) < p.emit[s]
        if self.w.context is not None and self.w.context.channels:
            x[self.w.context.channels] = (self.rng.random(len(self.w.context.channels))
                                          < self.w.context.emit[self.c])
        for ch, k1, k2, tab in self.w.links:
            x[ch] = self.rng.random() < np.asarray(tab)[self.s[k1], self.s[k2]]
        return tuple(float(v) for v in x)

    def reset(self) -> tuple:
        w = self.w
        self.c = int(self.rng.choice(len(w.context.init), p=w.context.init)) if w.context else 0
        self.s = [int(self.rng.choice(p.n_states, p=p.init)) for p in w.processes]
        return self._emit()

    def step(self, action: int) -> tuple:
        w = self.w
        new = []
        for p, s in zip(w.processes, self.s):
            row = p.trans[min(self.c, p.trans.shape[0] - 1), action, s]
            new.append(int(self.rng.choice(p.n_states, p=row)))
        if w.context is not None:
            self.c = int(self.rng.choice(len(w.context.init), p=w.context.trans[0, 0, self.c]))
        self.s = new
        return self._emit()


class Filter:
    """Exact Bayes filter with the true model: the ideal next-observation predictor.

        f.observe(x0); loop: p = f.predict(a); f.update(a, x_next)"""

    def __init__(self, world: World):
        self.init, self.trans, self.emit = world.joint()
        self.b = None

    def _lik(self, x) -> np.ndarray:
        x = np.asarray(x) > .5
        return np.prod(np.where(x[None, :], self.emit, 1. - self.emit), axis=1)

    def _set(self, prior, x) -> None:
        b = prior * self._lik(x)
        z = b.sum()
        if z <= 0:
            raise ValueError("observation impossible under the true model")
        self.b = b / z

    def observe(self, x0) -> None:
        self._set(self.init, x0)

    def predict(self, action: int) -> np.ndarray:
        """P(x_{t+1,i} = 1 | history, action) for every channel."""
        return (self.b @ self.trans[action]) @ self.emit

    def update(self, action: int, x_next) -> None:
        self._set(self.b @ self.trans[action], x_next)


# ============================================================ constructors
def _static(stay: np.ndarray, A: int, C: int = 1) -> np.ndarray:
    return np.broadcast_to(stay, (C, A) + stay.shape).copy()


def cue_delay(name, ch_cue, ch_out, delay, p_cue, A, p_out=1., C=1, out_by_context=None):
    """idle -> cue (prob p_cue per step) -> delay-1 silent wait states -> outcome
    -> idle (or straight to a new cue). The outcome appears `delay` steps after
    the cue. With out_by_context = [ch_c0, ch_c1, ...] the outcome channel
    depends on the hidden context (one outcome state per context)."""
    outs = list(out_by_context or [ch_out])
    n_out, D = len(outs), delay
    S = D + 1 + n_out                    # idle 0, cue 1, waits 2..D, outcomes D+1..
    T = np.zeros((C, A, S, S))
    for c in range(C):
        for a in range(A):
            T[c, a, 0, 0], T[c, a, 0, 1] = 1. - p_cue, p_cue
            for st in range(1, D):
                T[c, a, st, st + 1] = 1.
            T[c, a, D, D + 1 + (c if out_by_context else 0)] = 1.
            for o in range(D + 1, S):
                T[c, a, o, 0], T[c, a, o, 1] = 1. - p_cue, p_cue
    E = np.zeros((S, 1 + n_out))
    E[1, 0] = 1.
    for i in range(n_out):
        E[D + 1 + i, 1 + i] = p_out
    init = np.zeros(S)
    init[0] = 1.
    return Process(name, init, T, E, [ch_cue] + outs)


def noise(name, ch, p=.5, stay=None, A=2):
    """Binary channel: iid Bernoulli(p), or sticky (keeps its state with prob stay)."""
    if stay is None:
        M = np.array([[1. - p, p], [1. - p, p]])
    else:
        M = np.array([[stay, 1. - stay], [1. - stay, stay]])
    return Process(name, np.array([1. - p, p]) if stay is None else np.array([.5, .5]),
                   _static(M, A), np.eye(2)[:, 1:], [ch])


def hidden_hmm(name, chans, stay, e_on, e_off, A=2):
    """Sticky hidden 2-state chain; every channel is on with prob e_on in state 1
    and e_off in state 0 (noisy evidence must be integrated over time)."""
    M = np.array([[stay, 1. - stay], [1. - stay, stay]])
    E = np.array([[e_off] * len(chans), [e_on] * len(chans)])
    return Process(name, np.array([.5, .5]), _static(M, A), E, list(chans))


def controlled(name, ch, p_on_by_action, A):
    """Next state 'on' with a probability that depends only on the action."""
    T = np.zeros((1, A, 2, 2))
    for a, p in enumerate(p_on_by_action):
        T[0, a, :, 1], T[0, a, :, 0] = p, 1. - p
    return Process(name, np.array([1., 0.]), T, np.eye(2)[:, 1:], [ch])


def constant(name, ch, A):
    return Process(name, np.ones(1), np.ones((1, A, 1, 1)), np.zeros((1, 1)), [ch])


def sticky_context(C, hazard, channels=(), emit=None):
    M = np.full((C, C), hazard / max(1, C - 1))
    np.fill_diagonal(M, 1. - hazard)
    E = np.zeros((C, len(channels))) if emit is None else np.asarray(emit, float)
    return Process("context", np.full(C, 1. / C), M[None, None], E, list(channels))


# ================================================================ T1 set v0
def t1_worlds() -> dict:
    """Tiny worlds of the T1 probe set v0 (fixed before any agent was run on
    them). Channel layout: listed per world; the last channel is the value
    channel (constant 0, excluded from the T1 score)."""
    A = 2
    W = {}

    def world(name, n, procs, ctx=None, note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], ctx, note)
        w.check()
        W[name] = w

    world("delay2", 3, [cue_delay("cue", 0, 1, 2, .15, A)],
          note="cue -> outcome 2 steps later")
    world("delay2-noise2", 5, [cue_delay("cue", 0, 1, 2, .15, A),
                               noise("n1", 2, .5, A=A), noise("n2", 3, .5, A=A)],
          note="+ 2 iid noise channels")
    world("delay3-sticky", 4, [cue_delay("cue", 0, 1, 3, .12, A),
                               noise("s1", 2, stay=.9, A=A)],
          note="delay 3 + a sticky noise channel")
    world("delay4-p.8", 3, [cue_delay("cue", 0, 1, 4, .1, A, p_out=.8)],
          note="delay 4, outcome only 80 % of the time")
    world("two-delays", 5, [cue_delay("cue1", 0, 1, 2, .12, A),
                            cue_delay("cue2", 2, 3, 3, .1, A)],
          note="two independent cue processes (factorisation)")
    world("hmm-noisy", 3, [hidden_hmm("h", [0, 1], .95, .85, .15, A)],
          note="sticky hidden state seen through two noisy channels")
    world("context-reversal", 4,
          [cue_delay("cue", 0, None, 1, .2, A, C=2, out_by_context=[1, 2])],
          ctx=sticky_context(2, .004),
          note="cue -> outcome channel 1 or 2 depending on a hidden sticky context")
    world("control", 4, [controlled("light", 0, [.9, .1], A), noise("n1", 1, .5, A=A),
                         noise("s1", 2, stay=.85, A=A)],
          note="action sets a light (controllable) + iid and sticky noise")
    return W


# ======================================================= T1 set v1 (slice 1)
def noisy_cue_delay(name, ch_cue, ch_out, delay, p_cue, A, miss=.2, fa=.08, p_out=.9):
    """cue_delay whose cue is seen through noise: the true cue lights the cue
    channel with prob 1 - miss, every other state with prob fa. Whether a cue
    really happened (and hence whether the outcome comes) is hidden state that
    must be inferred from the cue flash and the absence of later evidence."""
    p = cue_delay(name, ch_cue, ch_out, delay, p_cue, A, p_out=p_out)
    p.emit[:, 0] = fa
    p.emit[1, 0] = 1. - miss
    return p


def counter(name, ch_cue, ch_out, k, p_cue, A):
    """Cues arrive at random (prob p_cue per step); the step after every k-th cue
    since the last outcome the outcome channel fires. The count is hidden state
    that no fixed window of recent observations recovers when cues are sparse.
    States: silent I_c (c = 0..k-1), cue C_c (c = 1..k), outcome O."""
    S = 2 * k + 1
    I = lambda c: c                      # noqa: E731
    Cc = lambda c: k + c - 1             # noqa: E731  c = 1..k
    O = 2 * k
    T = np.zeros((1, A, S, S))
    for a in range(A):
        for c in range(k):
            for src in ([I(c)] + ([Cc(c)] if c >= 1 else []) + ([O] if c == 0 else [])):
                T[0, a, src, Cc(c + 1)] += p_cue
                T[0, a, src, I(c)] += 1. - p_cue
        T[0, a, Cc(k), O] = 1.
    E = np.zeros((S, 2))
    for c in range(1, k + 1):
        E[Cc(c), 0] = 1.
    E[O, 1] = 1.
    init = np.zeros(S)
    init[I(0)] = 1.
    return Process(name, init, T, E, [ch_cue, ch_out])


def noisy_identity(name, ch_cue, ch_id, ch_a, ch_b, delay, p_cue, A, e_id=.75):
    """A cue of hidden type a or b (prob 1/2 each) lights the cue channel; during
    the `delay` steps that follow the identity channel is on with prob e_id
    (type a) or 1 - e_id (type b); then outcome a or b fires on its own channel.
    Predicting the outcome requires integrating the identity evidence over the
    whole delay. States: idle, then per type: cue, waits 1..delay-1, outcome."""
    D = delay
    per = D + 1                          # cue, D-1 waits, outcome
    S = 1 + 2 * per
    base = lambda t: 1 + t * per          # noqa: E731
    T = np.zeros((1, A, S, S))
    for a in range(A):
        starts = [0] + [base(t) + D for t in range(2)]     # idle and outcomes
        for src in starts:
            T[0, a, src, 0] = 1. - p_cue
            for t in range(2):
                T[0, a, src, base(t)] += p_cue / 2.
        for t in range(2):
            for st in range(D):
                T[0, a, base(t) + st, base(t) + st + 1] = 1.
    E = np.zeros((S, 4))                 # cue, id, out a, out b
    for t in range(2):
        E[base(t), 0] = 1.
        for st in range(1, D):           # waits (identity evidence)
            E[base(t) + st, 1] = e_id if t == 0 else 1. - e_id
        E[base(t) + D, 2 + t] = 1.
    init = np.zeros(S)
    init[0] = 1.
    return Process(name, init, T, E, [ch_cue, ch_id, ch_a, ch_b])


def context_cue(name, ch_cue, ch_out, delay, p_cue, A, p_by_context):
    """A cue followed `delay` steps later by an outcome on ONE channel whose
    probability depends on a hidden sticky context (no channel shows the
    context): only the outcomes themselves are evidence about it."""
    C = len(p_by_context)
    D = delay
    S = D + 1 + C                        # idle, cue, waits, one outcome state per context
    T = np.zeros((C, A, S, S))
    for c in range(C):
        for a in range(A):
            T[c, a, 0, 0], T[c, a, 0, 1] = 1. - p_cue, p_cue
            for st in range(1, D):
                T[c, a, st, st + 1] = 1.
            T[c, a, D, D + 1 + c] = 1.
            for o in range(D + 1, S):
                T[c, a, o, 0], T[c, a, o, 1] = 1. - p_cue, p_cue
    E = np.zeros((S, 2))
    E[1, 0] = 1.
    for c, p in enumerate(p_by_context):
        E[D + 1 + c, 1] = p
    init = np.zeros(S)
    init[0] = 1.
    return Process(name, init, T, E, [ch_cue, ch_out])


def hmm_k(name, chans, stay, e_own, e_other, A=2):
    """Sticky hidden chain with len(chans) states; state i lights channel i with
    prob e_own and every other channel with prob e_other."""
    K = len(chans)
    M = np.full((K, K), (1. - stay) / (K - 1))
    np.fill_diagonal(M, stay)
    E = np.full((K, K), e_other)
    np.fill_diagonal(E, e_own)
    return Process(name, np.full(K, 1. / K), _static(M, A), E, list(chans))


def t1_worlds_v1() -> dict:
    """T1 probe set v1 (slice 1): hidden state that is NOT recoverable from a
    fixed window of recent observations. Fixed before any agent was run on it
    (decisions.md S1-0). The v0 worlds remain the regression floor."""
    A = 2
    W = {}

    def world(name, n, procs, ctx=None, note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], ctx, note)
        w.check()
        W[name] = w

    world("hmm-noisy", 3, [hidden_hmm("h", [0, 1], .95, .85, .15, A)],
          note="(from v0) sticky 2-state chain, two noisy channels")
    world("hmm-1ch", 2, [hidden_hmm("h", [0], .97, .8, .2, A)],
          note="sticky 2-state chain, ONE noisy channel: long integration")
    world("hmm3", 4, [hmm_k("h", [0, 1, 2], .95, .75, .1, A)],
          note="sticky 3-state chain, state i lights channel i (.75) or others (.1)")
    world("noisy-cue-delay3", 3, [noisy_cue_delay("cue", 0, 1, 3, .12, A)],
          note="'delayed food with noise': cue missed 20 %, false flashes 8 %, food 90 %")
    world("count3", 3, [counter("cnt", 0, 1, 3, .25, A)],
          note="outcome the step after every 3rd cue (hidden count)")
    world("noisy-identity", 5, [noisy_identity("id", 0, 1, 2, 3, 4, .12, A)],
          note="cue type shown only by noisy evidence (.75) during a 4-step delay")
    world("context-hidden", 3, [context_cue("cc", 0, 1, 2, .2, A, [.9, .1])],
          ctx=sticky_context(2, .01),
          note="cue -> outcome .9 or .1 by a hidden context seen only through outcomes")
    world("two-hmm", 5, [hidden_hmm("h1", [0, 1], .95, .85, .15, A),
                         hidden_hmm("h2", [2, 3], .9, .8, .2, A)],
          note="two independent noisy chains (decomposition; slice 3 preview)")
    return W


def t1_worlds_d1() -> dict:
    """T1 decomposition set d1 (DRAFT for slice 3; proposed 2026-11-19 night, not yet agreed
    with the user, no agent tuned on it). Worlds with >= 3 independent hidden processes and a
    joint state space of >= 27 (most >= 64), so that a learner of the joint state pays the
    product of the state spaces within the episode, while a learner that knows the partition
    pays their sum. References: `filter` (true model), `kfact` (online HMM per process with
    the true partition and K — learnable ceiling), `kjoint` (one online HMM with K = joint
    size — the product cost)."""
    A = 2
    W = {}

    def world(name, n, procs, note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], None, note)
        w.check()
        W[name] = w

    world("d-3x3", 10, [hmm_k(f"h{k}", [3 * k, 3 * k + 1, 3 * k + 2], .95, .75, .1, A)
                        for k in range(3)],
          note="3 sticky 3-state chains, 3 channels each; joint 27")
    world("d-6x2", 13, [hidden_hmm(f"h{k}", [2 * k, 2 * k + 1], .95, .85, .15, A)
                        for k in range(6)],
          note="6 sticky 2-state chains, 2 noisy channels each; joint 64")
    world("d-4x3", 13, [hmm_k(f"h{k}", [3 * k, 3 * k + 1, 3 * k + 2], .95, .75, .1, A)
                        for k in range(4)],
          note="4 sticky 3-state chains, 3 channels each; joint 81")
    world("d-4x2-noise4", 13, [hidden_hmm(f"h{k}", [2 * k, 2 * k + 1], .95, .85, .15, A)
                               for k in range(4)]
          + [noise(f"n{j}", 8 + j, p=.3, A=A) for j in range(4)],
          note="4 sticky 2-state chains + 4 iid distractor channels (p .3); joint 256")
    world("d-mixed", 9, [hmm_k("h3", [0, 1, 2], .95, .75, .1, A),
                         noisy_cue_delay("cue", 3, 4, 3, .12, A),
                         hidden_hmm("h2", [5, 6], .95, .85, .15, A),
                         noise("n", 7, p=.3, A=A)],
          note="3-state chain + noisy cue -> delayed food + 2-state chain + distractor")
    return W


def t1_worlds_d2() -> dict:
    """T1 decomposition set d2: held-out structures for the slice-3 witness (fixed
    2026-11-19 23:46 UTC before anything was run on it; decisions.md D-3). Parameters
    differ from d1: weaker / faster chains, 4-state chains, unequal group sizes, 1-channel
    hidden chains, sticky distractors (themselves 1-channel processes), two delayed cues."""
    A = 2
    W = {}

    def world(name, n, procs, note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], None, note)
        w.check()
        W[name] = w

    world("d2-5x2-weak", 11, [hidden_hmm(f"h{k}", [2 * k, 2 * k + 1], .9, .75, .25, A)
                              for k in range(5)],
          note="5 faster, weaker 2-state chains (stay .9, emission .75 / .25); joint 32")
    world("d2-3x4", 13, [hmm_k(f"h{k}", [4 * k + i for i in range(4)], .97, .7, .15, A)
                         for k in range(3)],
          note="3 sticky 4-state chains, 4 channels each; joint 64")
    world("d2-unequal", 10, [hmm_k("h3", [0, 1, 2], .95, .75, .1, A),
                             hidden_hmm("h1", [3], .97, .8, .2, A),
                             hidden_hmm("h2", [4, 5, 6], .95, .8, .2, A),
                             noise("s1", 7, stay=.9, A=A), noise("s2", 8, stay=.9, A=A)],
          note="groups of 3, 1, 3 channels + 2 sticky 1-channel distractors; joint 48")
    world("d2-two-cues", 8, [noisy_cue_delay("c2", 0, 1, 2, .15, A),
                             noisy_cue_delay("c3", 2, 3, 3, .12, A),
                             hidden_hmm("h", [4, 5], .95, .85, .15, A),
                             noise("n", 6, p=.3, A=A)],
          note="two noisy cue -> delayed outcome processes (delays 2, 3) + chain + noise")
    return W


def t1_worlds_dscale() -> dict:
    """Scaling set for slice 3 (cost vs number of processes; decisions.md D-5): P sticky
    3-state chains with 3 channels each (as in d-4x3), P = 2, 4, 6, 7 (joint 9 ... 2187)."""
    A = 2
    W = {}
    for P in (2, 4, 6, 7):
        n = 3 * P + 1
        procs = [hmm_k(f"h{k}", [3 * k, 3 * k + 1, 3 * k + 2], .95, .75, .1, A) for k in range(P)]
        w = World(f"s-{P}x3", n, A, n - 1, procs + [constant("value", n - 1, A)], None,
                  f"{P} sticky 3-state chains; joint {3 ** P}")
        w.check()
        W[w.name] = w
    return W


def ctx_chain(name, chans, stay, p1_by_ctx, e_on, e_off, A=2):
    """2-state chain whose resampling target depends on the context state c: keeps its state
    with prob `stay`, otherwise draws state 1 with prob p1_by_ctx[c]. Two such chains on the
    same context are dependent (coupled through c)."""
    C = len(p1_by_ctx)
    T = np.zeros((C, A, 2, 2))
    for c, p1 in enumerate(p1_by_ctx):
        M = (1. - stay) * np.array([[1. - p1, p1], [1. - p1, p1]]) + stay * np.eye(2)
        T[c, :] = M
    E = np.array([[e_off] * len(chans), [e_on] * len(chans)])
    return Process(name, np.array([.5, .5]), T, E, list(chans))


def t1_worlds_dcouple() -> dict:
    """Boundary probe for slice 3 (decisions.md D-6): processes that are NOT independent.
    c-shared: a hidden sticky context (no channels) sets where two chains drift; a third chain
    is independent. c-driver: chain A (the family's `context`, owning channels 0-1) drives
    chain B's drift; a third chain is independent. Correct grouping for prediction: {A, B}
    together, the third apart."""
    A = 2
    W = {}
    n = 7
    ctx = sticky_context(2, .02)
    W["c-shared"] = World("c-shared", n, A, n - 1,
                          [ctx_chain("a", [0, 1], .9, [.85, .15], .85, .15, A),
                           ctx_chain("b", [2, 3], .9, [.85, .15], .85, .15, A),
                           hidden_hmm("c", [4, 5], .95, .85, .15, A),
                           constant("value", n - 1, A)], ctx,
                          "two chains drifting by a shared hidden context + independent chain")
    drv = sticky_context(2, .05, channels=[0, 1], emit=[[.15, .15], [.85, .85]])
    W["c-driver"] = World("c-driver", n, A, n - 1,
                          [ctx_chain("b", [2, 3], .8, [.1, .9], .85, .15, A),
                           hidden_hmm("c", [4, 5], .95, .85, .15, A),
                           constant("value", n - 1, A)], drv,
                          "chain A (visible, noisy) drives chain B + independent chain")
    for w in W.values():
        w.check()
    return W


def lever(name, chans, p_on_by_action, stay, e_on=.9, e_off=.1, A=2):
    """Controllable 2-state process: from either state, with prob 1 - stay the next state is
    drawn as 'on' with p_on_by_action[a], else kept (the action matters only through the
    redraws). Seen through noisy channels."""
    T = np.zeros((1, A, 2, 2))
    for a, p in enumerate(p_on_by_action):
        T[0, a] = (1. - stay) * np.array([[1. - p, p], [1. - p, p]]) + stay * np.eye(2)
    E = np.array([[e_off] * len(chans), [e_on] * len(chans)])
    return Process(name, np.array([.5, .5]), T, E, list(chans))


def t1_worlds_d3() -> dict:
    """Agent-relevant structures for slice 3 (decisions.md D-7; fixed 2026-11-20 00:37 UTC):
    action-dependent processes and shared-emission ("conjunctive") channels, which the
    independence assumption of d1 / d2 does not cover."""
    A = 2
    W = {}

    def world(name, n, procs, links=(), note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], None, note,
                  links=list(links))
        w.check()
        W[name] = w

    AND = [[.05, .05], [.05, .9]]
    world("d3-ctrl", 7, [lever("lev", [0, 1], [.1, .9], .7),
                         hidden_hmm("h1", [2, 3], .95, .85, .15, A),
                         hidden_hmm("h2", [4, 5], .95, .85, .15, A)],
          note="a lever process driven by the action + 2 independent chains")
    world("d3-and", 8, [hidden_hmm("a", [0, 1], .95, .85, .15, A),
                        hidden_hmm("b", [2, 3], .95, .85, .15, A),
                        hidden_hmm("c", [4, 5], .95, .85, .15, A)], links=[(6, 0, 1, AND)],
          note="channel 6 on (.9) iff chain a AND chain b are in state 1; chain c independent")
    world("d3-and-ctrl", 8, [lever("lev", [0, 1], [.1, .9], .7),
                             hidden_hmm("light", [2, 3], .95, .85, .15, A),
                             hidden_hmm("c", [4, 5], .95, .85, .15, A)], links=[(6, 0, 1, AND)],
          note="food (6) iff lever on AND light state 1 (the canonical agent case); c independent")
    return W


def t1_worlds_d3h() -> dict:
    """Held-out conjunction worlds (decisions.md D-13; fixed 2026-11-20 05:40 UTC, before the
    conjunctive component was run on anything): other link tables (OR, XOR — XOR leaves no
    marginal correlation with either parent), other chain parameters, two conjunctive
    channels with different parent pairs."""
    A = 2
    W = {}

    def world(name, n, procs, links=(), note=""):
        w = World(name, n, A, n - 1, procs + [constant("value", n - 1, A)], None, note,
                  links=list(links))
        w.check()
        W[name] = w

    OR = [[.05, .85], [.85, .9]]
    XOR = [[.05, .9], [.9, .05]]
    AND = [[.1, .1], [.1, .85]]
    world("d3h-or", 8, [hidden_hmm("a", [0, 1], .92, .8, .2, A),
                        hidden_hmm("b", [2, 3], .92, .8, .2, A),
                        hidden_hmm("c", [4, 5], .92, .8, .2, A)], links=[(6, 0, 1, OR)],
          note="channel 6 on iff a OR b in state 1")
    world("d3h-xor", 8, [hidden_hmm("a", [0, 1], .95, .85, .15, A),
                         hidden_hmm("b", [2, 3], .95, .85, .15, A),
                         hidden_hmm("c", [4, 5], .95, .85, .15, A)], links=[(6, 0, 1, XOR)],
          note="channel 6 on iff exactly one of a, b in state 1 (no marginal correlation)")
    world("d3h-two", 10, [hidden_hmm("a", [0, 1], .93, .85, .15, A),
                          hidden_hmm("b", [2, 3], .93, .85, .15, A),
                          hidden_hmm("c", [4, 5], .93, .85, .15, A)],
          links=[(6, 0, 1, AND), (7, 1, 2, OR)],
          note="channel 6 = a AND b, channel 7 = b OR c; channel 8 unused (constant 0)")
    return W


WORLD_SETS = {"v0": t1_worlds, "v1": t1_worlds_v1, "d1": t1_worlds_d1, "d2": t1_worlds_d2,
              "dscale": t1_worlds_dscale, "dcouple": t1_worlds_dcouple,
              "d3": t1_worlds_d3, "d3h": t1_worlds_d3h}
