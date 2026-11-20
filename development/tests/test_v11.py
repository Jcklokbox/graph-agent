"""Focused regression tests of the v11 agent (one principle per test).

Run from development/:  python -m unittest discover -s tests -t ."""
from itertools import permutations
import random
import unittest

from agents.v11 import Cell, PosteriorGraph, Window, split_log_bf


def _probe(*args, **kw):
    """Brain for component probes that use the value channel as a generic
    [0, 1] target: its frame is fixed (reference .5, scale 1), so z = w."""
    b = PosteriorGraph(*args, **kw)
    b._see_value(.5)
    b._vlo, b._vhi, b._vcounts = 0., 1., {.5: 10 ** 9}
    b.attended = set(range(b.n_inputs))     # attention granted: probes test the layers below
    return b

class PosteriorGraphTests(unittest.TestCase):
    def test_fractional_determinism_is_not_bernoulli_noise(self):
        fractional, binary = Cell(), Cell()
        for i in range(40):
            fractional.update(.5)
            binary.update(float(i % 2))
        self.assertEqual(fractional.kind, "continuous")
        self.assertEqual(binary.kind, "binary")
        self.assertEqual(fractional.p, binary.p)
        self.assertEqual(fractional.m2, 0.)
        self.assertGreater(fractional.density(.5), 3.)     # a sharp Gaussian
        self.assertEqual(binary.density(1.), .5)

    def test_split_bayes_factor(self):
        self.assertLess(split_log_bf([(5, 10), (5, 10)]), 0.)        # same rate: Occam
        self.assertGreater(split_log_bf([(10, 10), (0, 10)]), 5.)    # different rates
        self.assertAlmostEqual(split_log_bf([(3, 7)]), 0.)

    def test_step_learn_online_cycle(self):
        brain = _probe(3, 2, 2, seed=8)
        for i in range(8):
            obs = (float(i % 2), 0., .5)
            action = brain.step(obs)
            next_obs = (obs[0], 0., float(action == 1))
            brain.learn(next_obs)
        self.assertIsNone(brain._pending)
        self.assertTrue(brain.m.fac)

    def test_tables_grow_with_observed_contexts_not_powerset(self):
        brain = _probe(4, 2, 3, seed=7)
        for i in range(240):
            x = (float(i % 2), float((i // 2) % 2), 0., .5)
            action = i % 2
            y = (x[0], x[1], 0., float(x[0] * x[1]))
            brain._pending = (x, action)
            brain.learn(y)
        # Factor cells: root + one per cue, per (action, target); context cells
        # only for the 4 observed active sets.
        self.assertLessEqual(len(brain.m.fac), 2 * 4 * 5)
        self.assertLessEqual(len({k[0] for k in brain.m.ctx}), 4)
        # The conjunction is known in its visited exact context.
        brain._invalidate()
        self.assertGreater(brain.predict((1., 1., 0., .5), 1, 3), .8)
        self.assertLess(brain.predict((1., 0., 0., .5), 1, 3), .2)

    def test_shared_inhibitor_transfer_and_excitor_memory(self):
        brain = _probe(4, 2, 3, seed=1)
        def train(obs, outcome, n=30):
            for _ in range(n):
                brain._pending = (tuple(map(float, obs)), 1)
                brain.learn((0., 0., 0., float(outcome)))
        train((1,0,0,0), 1)       # A+
        train((1,0,1,0), 0)       # AX-
        train((0,1,0,0), 1)       # independent B+
        p_a = brain.predict((1,0,0,0),1,3)
        p_ax = brain.predict((1,0,1,0),1,3)
        p_b = brain.predict((0,1,0,0),1,3)
        p_bx = brain.predict((0,1,1,0),1,3)
        self.assertGreater(p_a, .8)
        self.assertGreater(p_b, .8)
        self.assertLess(p_ax, .15)
        self.assertLess(p_bx, .15)

    def test_hazard_records_censoring_and_consumes_event(self):
        brain = _probe(4, 2, 3, seed=2)
        blank = (0.,0.,0.,.5)
        cue = (1.,0.,0.,.5)
        event = (0.,0.,1.,.5)
        # Rising edge of channel 0, one blank step, then channel 2 fires.
        for _ in range(12):
            brain._pending=(blank,0); brain.learn(cue)    # anchors pending (0,*)
            brain._pending=(cue,0); brain.learn(blank)    # age 1
            brain._pending=(blank,0); brain.learn(event)  # event at age 2
        key = (0, 2, 2, 0)
        self.assertIn(key, brain.m.hazard)
        cell = brain.m.hazard[key]
        self.assertGreater(cell.k / cell.n, .8)
        self.assertFalse(any(w.cue == 0 and w.target == 2 for w in brain.windows))

    def test_two_step_rollout_prefers_costly_setup(self):
        brain = _probe(4, 2, 3, seed=3)
        s0=(1.,0.,0.,.5); s1=(0.,1.,0.,.2)
        stay=(1.,0.,0.,.5); terminal=(0.,0.,1.,1.)
        for _ in range(40):
            brain._pending=(s0,0); brain.learn(s1)
            brain._pending=(s1,0); brain.learn(terminal)
            brain._pending=(s0,1); brain.learn(stay)
        self.assertGreater(brain._q(s0,0), brain._q(s0,1))
        self.assertEqual(brain.step(s0),0)

    def test_suppressive_baseline_cue_does_not_veto_enabling_cause(self):
        # Under action 1: P alone -> 0 (suppressive vs background), C alone -> 1,
        # and C with P -> 1. P's negative marginal only reflects C's absence.
        b = _probe(4, 2, 3, seed=0)
        data = [((1., 0., 0., 0.), 0.), ((0., 1., 0., 0.), 1.), ((1., 1., 0., 0.), 1.),
                ((0., 0., 1., 0.), .5)]
        for _ in range(25):
            for obs, y in data:
                b._pending = (obs, 1); b.learn((0., 0., 0., y))
        self.assertLess(b.predict((1., 0., 0., 0.), 1, 3), .2)
        self.assertGreater(b.predict((1., 1., 0., 0.), 1, 3), .8)

    def test_hazard_compares_event_rate_not_mean_on_continuous_channel(self):
        # A constant .5 value channel never has events; a transient cue's
        # hazard rate (0 events) must not drag the value prediction below .5.
        b = _probe(4, 2, 3, seed=0)
        for t in range(120):
            cue = 1. if t % 4 == 0 else 0.
            b.step((cue, 1., 0., .5)); b.learn((1. if (t + 1) % 4 == 0 else 0., 1., 0., .5))
        self.assertAlmostEqual(b.predict((0., 1., 0., .5), 1, 3), .5, delta=.05)

    def test_exact_context_prevents_correlated_cue_double_counting(self):
        # Cue C appears only together with P; P under action 1 is always .5.
        # Noisy-OR of marginal strengths would stack C on P; the visited
        # exact context must report the observed .5.
        b = _probe(4, 2, 3, seed=0)
        for t in range(60):
            obs = (1., 1. if t % 3 == 0 else 0., 0., .5) if t % 2 else (0., 0., 1., .5)
            y = .5 if obs[0] else 0.
            b._pending = (obs, 1); b.learn((0., 0., 0., y))
        x = (1., 1., 0., .5)
        stacked = b._excitation((0, 1), 1, 3)[0]  # factor-only noisy-OR
        self.assertGreater(stacked, .6)           # the double-counting exists
        self.assertAlmostEqual(b.predict(x, 1, 3), .5, delta=.05)

    def test_uncertainty_is_wide_for_unseen_cue_and_narrow_for_known_context(self):
        b = _probe(4, 2, 3, seed=0)
        for _ in range(40):
            b._pending = ((1., 0., 0., .5), 1); b.learn((0., 0., 0., .5))
            b._pending = ((1., 0., 0., .5), 0); b.learn((0., 0., 0., 0.))
        _, sd_known = b._predict((1., 0., 0., .5), 1, 3, 0)
        _, sd_unseen_cue = b._predict((1., 1., 0., .5), 0, 3, 0)
        self.assertLess(sd_known, .05)
        self.assertGreater(sd_unseen_cue, 4. * sd_known)
        self.assertGreater(sd_unseen_cue, .15)

    def test_timing_cells_are_context_specific(self):
        # After cue 0, the event at lag 2 happens only when channel 1 is on.
        b = _probe(4, 2, 3, seed=0)
        for k in range(40):
            gate = float(k % 2)
            seq = [((0., 0., 0., 0.), (1., 0., 0., 0.)),
                   ((1., 0., 0., 0.), (0., gate, 0., 0.)),
                   ((0., gate, 0., 0.), (0., 0., 0., gate)),
                   ((0., 0., 0., gate), (0., 0., 0., 0.)),
                   ((0., 0., 0., 0.), (0., 0., 0., 0.))]
            for x, y in seq:
                b._pending = (x, 0); b.learn(y)
        a_on, b_on = b._timing_posterior(0, 3, 2, 0, (1,))
        a_off, b_off = b._timing_posterior(0, 3, 2, 0, ())
        self.assertGreater(a_on / (a_on + b_on), .7)
        self.assertLess(a_off / (a_off + b_off), .3)

    def test_irrelevant_uncontrollable_channel_does_not_fragment_timing_cells(self):
        # After cue 0 the event 3 follows at lag 2; channel 1 is iid noise.
        import random
        rng = random.Random(1)
        b = _probe(5, 2, 4, seed=0)
        prev, cues = (0.,) * 5, [0., 0.]
        for k in range(400):
            y = [0.] * 5
            y[1] = float(rng.random() < .5)
            y[0] = float(rng.random() < .2 and not any(cues))
            y[3] = cues[0]
            cues = [cues[1], y[0]]
            b._pending = (prev, rng.randrange(2)); b.learn(tuple(y))
            prev = tuple(y)
        self.assertGreater(b._p_uncontrollable(1), .5)
        gen = b.m.hazard[(0, 3, 2, 0)].n
        for ctx in ((), (1,)):
            a, c = b._timing_posterior(0, 3, 2, 0, ctx)
            self.assertGreater(a / (a + c), .8)
            # the reduced level pools the evidence over the noise split
            self.assertGreater(b._reduced_count((0, 3, 2, 0), ctx), .9 * gen)

    def test_controllability_evidence_is_not_diluted_by_unattended_noise(self):
        # Channel 1 depends (rarely) on a rarely taken action. Four iid noise
        # channels split every exact context; the coarse (attended) evidence
        # sums the split cells, so it does not lose to per-context Occam
        # penalties the way the exact-context sum does.
        def run(noise):
            rng, nrng = random.Random(0), random.Random(5)
            b = _probe(8, 2, 7, seed=0)
            b.attended = {0, 1}
            prev = (1.,) + (0.,) * 7
            for _ in range(300):
                a = 1 if rng.random() < .1 else 0
                y = ([1., float(a == 1 and rng.random() < .3)]
                     + [float(noise and nrng.random() < .5) for _ in range(4)] + [0., .5])
                b._pending = (prev, a); b.learn(tuple(y)); prev = tuple(y)
            exact = 0.     # the same evidence summed over exact contexts
            for fk in {k[0] for k in b.m.ctx if len(k) == 3}:
                groups = [(c.k, c.n) for a in range(2)
                          for c in (b.m.ctx.get((fk, a, 1)),) if c is not None and c.n > 0]
                exact += split_log_bf(groups) if len(groups) > 1 else 0.
            return b.m.ctrl_total[1], exact, b._p_uncontrollable(1)
        coarse0, exact0, _ = run(False)
        coarse1, exact1, p_unc = run(True)
        self.assertGreater(coarse1, .9 * coarse0)
        self.assertLess(exact1, .8 * exact0)
        self.assertLess(p_unc, 1e-6)

    def test_change_point_forgetting_is_stable_when_predictable_and_fast_after_surprise(self):
        b = _probe(4, 2, 3, seed=0, global_hazard=0.)
        x = (1., 0., 0., 0.)
        b._pending = ((0., 1., 0., 0.), 0); b.learn((0., 0., 0., 0.))
        idle = b.m.ctx[((1,), 0, 3)]
        for _ in range(300):
            b._pending = (x, 1); b.learn((0., 0., 0., 0.))
        cell = b.m.ctx[((0,), 1, 3)]
        # Bayesian change-point memory saturates near 2/h *visits* (h = .01);
        # a cell receiving no observations does not forget at all.
        self.assertGreater(cell.n, 150.)
        self.assertEqual(idle.n, 1.)
        n_before = cell.n
        for _ in range(2):
            b._pending = (x, 1); b.learn((0., 0., 0., 1.))
        self.assertLess(cell.n, .5 * n_before)      # two surprises halve memory
        for _ in range(8):
            b._pending = (x, 1); b.learn((0., 0., 0., 1.))
        self.assertGreater(b.predict(x, 1, 3), .5)  # the new regime takes over

    def test_global_change_discounts_unvisited_cells_only_on_joint_surprise(self):
        b = _probe(4, 2, 3, seed=0, change_hazard=0.)
        for _ in range(200):
            b._pending = ((1., 0., 0., 0.), 0); b.learn((0., 1., 0., 0.))
            b._pending = ((0., 1., 0., 0.), 1); b.learn((1., 0., 0., 0.))
        avoided = b.m.ctx[((1,), 1, 0)]
        before = avoided.n
        b._pending = ((1., 0., 0., 0.), 0); b.learn((0., 1., 0., 0.))   # predictable
        self.assertAlmostEqual(avoided.n, before, delta=1e-3 * before)
        b._pending = ((1., 0., 0., 0.), 0); b.learn((1., 0., 1., 1.))   # one outlier
        self.assertLess(b.last_global_change, .1)               # robust: costs ~eps
        self.assertGreater(avoided.n, .8 * before)
        for _ in range(3):                                      # rules changed
            b._pending = ((1., 0., 0., 0.), 0); b.learn((1., 0., 1., 1.))
        self.assertGreater(b.last_global_change, .9)
        self.assertLess(avoided.n, .1 * before)

    def test_stored_regime_is_reinstated_after_reversal_back(self):
        # Regime A: action 0 -> channel 2; regime B: action 0 -> channel 3.
        b = _probe(4, 2, 1, seed=0, change_hazard=0.)
        x = (1., 0., 0., 0.)
        A, B = (0., 0., 1., 0.), (0., 0., 0., 1.)
        for _ in range(150):
            b._pending = (x, 0); b.learn(A)
        for _ in range(150):
            b._pending = (x, 0); b.learn(B)
        self.assertEqual(len(b._regimes), 1)          # A archived at the change
        self.assertLess(b.predict(x, 0, 2), .2)
        def back_to_a(brain):
            steps = 0
            while brain.predict(x, 0, 2) < .8 and steps < 100:
                brain._pending = (x, 0); brain.learn(A); steps += 1
            return steps
        fresh = _probe(4, 2, 1, seed=0, change_hazard=0.)
        fresh.regime_max = 0                          # same history, no regime memory
        for y in [A] * 150 + [B] * 150:
            fresh._pending = (x, 0); fresh.learn(y)
        n_with, n_without = back_to_a(b), back_to_a(fresh)
        self.assertEqual(b.regime_switches, 1)
        self.assertLessEqual(n_with, 6)               # savings: reinstated, not relearned
        self.assertGreater(n_without, 3 * n_with)

    def test_uncontrollable_channel_transfers_observation_across_actions(self):
        # Channel 2 follows cue 0 unless stop-cue 1 is present, regardless of
        # the action; the agent mostly takes action 1 and tries action 0 rarely.
        b = _probe(4, 2, 3, seed=0)
        for k in range(120):
            stop = float(k % 2)
            x = (1., stop, 0., 0.)
            b._pending = (x, 0 if k % 20 == 0 else 1)
            b.learn((0., 0., 1. - stop, 0.))
            b._pending = ((0., 0., 0., 0.), 1); b.learn((0., 0., 0., 0.))
        self.assertGreater(b._p_uncontrollable(2), .7)
        self.assertGreater(b.predict((1., 0., 0., 0.), 0, 2), .8)
        self.assertLess(b.predict((1., 1., 0., 0.), 0, 2), .2)

    def test_belief_memory_distinguishes_window_age(self):
        b = _probe(4, 2, 3, seed=0)
        b.m.hazard[(0, 3, 2, 0)] = Cell("binary", 5., 5.)
        b.m.hazard_index[(0, 3)] = [(0, 3, 2, 0)]
        b.m.event_root[(3, 0)] = Cell("binary", 50., 0.)
        b.windows = [Window(0, 3, 1)]
        # The window enters the belief state only at a lag with credible
        # non-baseline timing (here lag 2 = age 1 + 1); at lag 3 nothing is
        # known, so it promises nothing beyond the baseline.
        self.assertEqual(b._open_cues([], 3, 0), ((0, 1),))
        self.assertEqual(b._open_cues([], 3, 1), ())
        b.windows = [Window(0, 3, 0)]
        b._invalidate()
        self.assertEqual(b._open_cues([], 3, 1), ((0, 1),))   # tau advances in rollout

    def test_novel_predicted_state_has_uninformed_reward_prior(self):
        # A predicted next state that was never observed has an unknown
        # reward (uniform over the experienced range, centred on the status
        # quo), instead of contributing nothing.
        b = _probe(4, 2, 3, seed=0)
        for _ in range(30):
            b._pending = ((0., 0., 0., .5), 1); b.learn((1., 0., 0., 0.))
            b._pending = ((1., 1., 0., .5), 0); b.learn((1., 1., 0., .5))
        del b.m.reward[(0, 1)]         # the state {0, 1} has never been an observation
        known = b._model_based_value((0., 0., 0., .5), 1, 0, ())
        novel = b._model_based_value((1., 1., 0., .5), 0, 0, (0, 1))
        self.assertLess(known[0], .1)
        self.assertEqual(novel[0], .5)
        self.assertGreater(novel[1], .28)

    def test_event_bonus_counts_tests_in_reduced_context(self):
        # A channel shown not to modulate the timing cell (iid noise: same
        # rate with and without it, and not controllable) does not make a
        # test novel: the tests of both noise states count together.
        b = _probe(4, 3, 3, seed=0)
        b.windows = [Window(0, 3, 1, .8)]
        for ctx in ((1,), (1, 2)):
            b.m.hazard_ctx[(0, 3, 2, 0, ctx)] = Cell("binary", 10., 1.)
            b.m.hctx_index.setdefault((0, 3, 2, 0), []).append((0, 3, 2, 0, ctx))
        b.m.ctrl_total[2] = -20.                                # channel 2: uncontrollable
        b._invalidate()
        self.assertAlmostEqual(b._event_bonus((0., 1., 1., .5), 0, 0), .8 / 21.)
        b.m.ctrl_total[2] = 20.                                 # controllable: keeps its split
        b._invalidate()
        self.assertAlmostEqual(b._event_bonus((0., 1., 1., .5), 0, 0), .8 / 11.)

    def test_event_bonus_is_selective_for_salient_untested_windows(self):
        b = _probe(4, 3, 3, seed=0)
        x = (0., 1., 0., .5)
        # Salient (unpredicted) onset of cue 0, one step ago: lag 2 at depth 0.
        b.windows = [Window(0, 3, 1, .8)]
        self.assertAlmostEqual(b._event_bonus(x, 0, 0), .8)       # untested combination
        b.m.hazard_ctx[(0, 3, 2, 0, (1,))] = Cell("binary", 3., 0.)
        b.m.hctx_index.setdefault((0, 3, 2, 0), []).append((0, 3, 2, 0, (1,)))
        b._invalidate()                                           # counts edited directly
        self.assertAlmostEqual(b._event_bonus(x, 0, 0), .2)       # tested 3 times -> 1/(1+3)
        self.assertAlmostEqual(b._event_bonus(x, 1, 0), .8)       # other action still untested
        self.assertAlmostEqual(b._event_bonus(x, 1, 1), .8)       # lag 3: no horizon cut-off
        b.windows = [Window(0, 3, 1, 0.)]                         # predicted (self-caused) onset
        b._invalidate()
        self.assertEqual(b._event_bonus(x, 1, 0), 0.)
        b.windows = [Window(0, 3, 1, .8)]
        b.m.hazard[(0, 3, 2, 1)] = Cell("binary", 8., 8.)
        b.m.hazard_index[(0, 3)] = [(0, 3, 2, 1)]
        b.m.event_root[(3, 1)] = Cell("binary", 50., 0.)
        b._invalidate()                                           # counts edited directly
        self.assertEqual(b._event_bonus(x, 1, 0), 0.)             # effect already located

    def test_value_range_adaptation_is_affine_invariant(self):
        # The same world with an affinely re-scaled value channel produces the
        # same normalised predictions and the same actions (ranges wider than
        # value_scale are fully normalised).
        import random
        def run(f):
            b = PosteriorGraph(4, 3, 3, seed=2, value_scale=.1); rng = random.Random(5); acts = []
            x = (0., 1., 0., f(.5))
            for t in range(150):
                a = b.step(x); acts.append(a)
                cue = float(rng.random() < .3)
                w = 1. if (a == 0 and x[0] > .5) else (0. if a == 2 else .5)
                x = (cue, 1., float(rng.random() < .5), f(w)); b.learn(x)
            return acts, b.predict(x, 0, 3)
        a1, p1 = run(lambda w: w)
        a2, p2 = run(lambda w: .2 + .5 * w)
        self.assertEqual(a1, a2)
        self.assertAlmostEqual(p1, p2, places=9)

    def test_value_reference_point_and_narrow_range(self):
        b = PosteriorGraph(3, 2, 2, seed=0)          # value_scale .5
        def z(w):
            ref, S = b._value_frame()
            return .5 + (w - ref) / S
        for w in (.5, .5, .5, .45):
            b._see_value(w)
        self.assertAlmostEqual(z(.5), .5)            # status quo is the middle
        self.assertAlmostEqual(z(.5) - z(.45), .1)   # .05 / .5, not a full range
        for w in (0., 1.):
            b._see_value(w)
        self.assertEqual([round(z(w), 9) for w in (0., .5, 1.)], [0., .5, 1.])

    def test_imagined_successor_never_activates_value_channel(self):
        b = _probe(3, 2, 2, seed=0)
        for _ in range(20):
            b._pending = ((1., 0., .5), 0); b.learn((1., 0., .52))
        self.assertGreater(b.predict((1., 0., .5), 0, 2), .5)
        seen = []
        orig = b._predict
        b._predict = lambda x, a, j, depth, act=None: (seen.append((depth, x)),
                                                       orig(x, a, j, depth, act))[1]
        b._q((1., 0., .5), 0)
        self.assertTrue(all(x[2] <= .5 for depth, x in seen if depth == 1))

    def test_attention_ignores_noise_and_pools_over_it(self):
        # ch0: random cue; ch1: copy of ch0 one step later; ch2, ch3: noise.
        import random
        rng = random.Random(3)
        b = PosteriorGraph(5, 2, 4, seed=0)
        obs = (0., 0., 0., 0., .5)
        for _ in range(400):
            b.step(obs)
            nxt = (float(rng.random() < .3), obs[0], float(rng.random() < .5),
                   float(rng.random() < .5), .5)
            b.learn(nxt)
            obs = nxt
        self.assertTrue({0, 1} <= b.attended)
        self.assertFalse({2, 3} & b.attended)
        # An exact context seen rarely (cue with both noise channels on) is
        # predicted from its coarse parent (cue on), not from scratch.
        x = b._normalise((1., 0., 1., 1., .5))
        b._invalidate()
        self.assertGreater(b._predict(x, 0, 1, 0)[0], .8)

    def test_attention_granger_delayed_cause_and_sticky_noise(self):
        # ch1 copies ch0 two steps later (a delayed cause); ch2, ch3 are
        # independent sticky processes (spuriously correlated in raw counts).
        import random
        rng = random.Random(5)
        b = PosteriorGraph(5, 2, 4, seed=0)
        hist = [0., 0.]
        s2, s3 = 0., 1.
        obs = (0., 0., s2, s3, .5)
        for _ in range(600):
            b.step(obs)
            c = float(rng.random() < .3)
            if rng.random() > .95:
                s2 = float(rng.random() < .5)
            if rng.random() > .95:
                s3 = float(rng.random() < .5)
            nxt = (c, hist[0], s2, s3, .5)
            hist = [hist[1], c]
            b.learn(nxt)
            obs = nxt
        self.assertTrue({0, 1} <= b.attended)
        self.assertFalse({2, 3} & b.attended)

    def test_continuation_is_expectation_over_next_observation(self):
        b = _probe(4, 2, 3, seed=0)
        b.attended = set()
        b._best_next = lambda s: 1. if s[0] > .5 else 0.
        # Channel 0 will be on with p = .3: the modal successor (off) is worth
        # 0, the expectation over successors .3.
        self.assertAlmostEqual(b._expected_continuation((.3, 0., 0., .5)), .3)

    def test_timing_evidence_weighted_by_onset_surprise(self):
        b = _probe(3, 2, 2, seed=0)
        b.windows = [Window(0, 1, 0, 0.), Window(2, 1, 0, 1.)]
        b._update_timing((0., 1., 0.), 0, set(), (True, True, True))
        self.assertEqual((b.m.hazard[(0, 1, 1, 0)].k, b.m.hazard[(0, 1, 1, 0)].n), (0., 0.))
        self.assertEqual((b.m.hazard[(2, 1, 1, 0)].k, b.m.hazard[(2, 1, 1, 0)].n), (1., 1.))

    def test_teacher_forced_equivariance_all_small_permutations(self):
        # Dataset is fixed in physical coordinates. The complete transition
        # stream and chosen actions are transformed together for each learner.
        rows = []
        for _ in range(12):
            for s in range(4):
                for action in range(2):
                    x = (float(s & 1), float(s >> 1), 0., .5)
                    y = (x[0], x[1], 0., float(action))
                    rows.append((x, action, y))
        original = _probe(4, 2, 3, seed=3)
        for x, action, y in rows:
            original._pending = (x, action)
            original.learn(y)
        comparisons = 0
        for sensor_map in permutations(range(3)):
            for action_map in permutations(range(2)):
                mapped = _probe(4, 2, 3, seed=3)
                for x, action, y in rows:
                    xx, yy = [0.] * 4, [0.] * 4
                    for old, new in enumerate(sensor_map):
                        xx[new], yy[new] = x[old], y[old]
                    xx[3], yy[3] = x[3], y[3]
                    mapped._pending = (tuple(xx), action_map[action])
                    mapped.learn(yy)
                for x, _, _ in rows[:4]:
                    xx = [0.] * 4
                    for old, new in enumerate(sensor_map):
                        xx[new] = x[old]
                    xx[3] = x[3]
                    for action in range(2):
                        self.assertAlmostEqual(
                            original.predict(x, action, 3),
                            mapped.predict(xx, action_map[action], 3), places=12)
                        comparisons += 1
        self.assertEqual(comparisons, 96)

    def test_closed_loop_equivariance_under_sensor_and_action_relabeling(self):
        # A learner in a relabelled world (positions permuted, actions
        # permuted) behaves exactly as the relabelled original learner.
        try:
            import numpy as np
            from envs.tasks import BlindGateEnv, Relabel
        except ImportError:
            self.skipTest("numpy not available")
        for s3 in permutations(range(3)):
            for action_map in permutations(range(3)):
                inverse = [action_map.index(i) for i in range(3)]
                e0, e1 = BlindGateEnv(seed=88), BlindGateEnv(seed=88)
                x0, x1 = e0.reset(), e1.reset()
                b0, b1 = PosteriorGraph(12, 3, 11, seed=19), PosteriorGraph(12, 3, 11, seed=19)
                rng, wrapper = random.Random(901), Relabel(e1, tuple(s3) + (3, 4, 5, 6, 7),
                                                           action_map)
                for _ in range(20):
                    noise = [rng.random() for _ in range(3)]
                    noise1 = [0.] * 3
                    for a in range(3):
                        noise1[inverse[a]] = noise[a]
                    a0 = b0.step(x0, tie_noise=noise)
                    a1 = b1.step(wrapper.map(x1), tie_noise=noise1)
                    self.assertEqual(a1, inverse[a0])
                    x0n, r0, _, _ = e0.step(a0)
                    x1n, r1, _, _ = e1.step(action_map[a1])
                    self.assertEqual(r0, r1)
                    self.assertTrue(np.array_equal(wrapper.map(x0n), wrapper.map(x1n)))
                    b0.learn(x0n)
                    b1.learn(wrapper.map(x1n))
                    x0, x1 = x0n, x1n


if __name__ == "__main__":
    unittest.main()
