"""T0 checks of the environment family and its exact reference filter."""
import itertools
import unittest

import numpy as np

from envs.family import (Filter, WorldEnv, cue_delay, noise, t1_worlds, t1_worlds_v1, World,
                         constant, WORLD_SETS)


def _brute_force_predictive(world, xs, actions):
    """P(x_{T+1} = 1 | x_0..x_T, a_0..a_T) by summing over every hidden path."""
    init, trans, emit = world.joint()
    N = len(init)

    def lik(s, x):
        x = np.asarray(x) > .5
        return float(np.prod(np.where(x, emit[s], 1. - emit[s])))

    num = np.zeros(world.n_channels)
    den = 0.
    for path in itertools.product(range(N), repeat=len(xs) + 1):
        w = init[path[0]] * lik(path[0], xs[0])
        for t in range(1, len(xs)):
            w *= trans[actions[t - 1], path[t - 1], path[t]] * lik(path[t], xs[t])
        if w == 0.:
            continue
        w *= trans[actions[len(xs) - 1], path[len(xs) - 1], path[len(xs)]]
        num += w * emit[path[len(xs)]]
        den += w
    return num / den


class FamilyTests(unittest.TestCase):
    def test_filter_equals_brute_force_enumeration(self):
        w = World("t", 4, 2, 3, [cue_delay("cue", 0, 1, 2, .3, 2, p_out=.7),
                                 noise("n", 2, stay=.8), constant("v", 3, 2)])
        env, rng = WorldEnv(w, 3), np.random.default_rng(1)
        xs, acts = [env.reset()], []
        for _ in range(4):
            acts.append(int(rng.integers(2)))
            xs.append(env.step(acts[-1]))
        f = Filter(w)
        f.observe(xs[0])
        for t in range(1, len(xs)):
            f.update(acts[t - 1], xs[t])
        acts.append(1)
        np.testing.assert_allclose(f.predict(1), _brute_force_predictive(w, xs, acts), atol=1e-12)

    def test_every_t1_world_is_valid_and_has_a_learnable_gap(self):
        for name, w in [kv for f in WORLD_SETS.values() for kv in f().items()]:
            w.check()
            env, f, rng = WorldEnv(w, 0), Filter(w), np.random.default_rng(0)
            f.observe(env.reset())
            ll_f = ll_m = 0.
            counts = np.ones(w.n_channels)
            tot = 2.
            ch = [i for i in range(w.n_channels) if i != w.value_index]
            for t in range(1500):
                a = int(rng.integers(2))
                p = np.clip(f.predict(a), 1e-9, 1 - 1e-9)
                pm = counts / tot
                y = np.array(env.step(a))
                f.update(a, y)
                ll_f -= np.sum((y * np.log(p) + (1 - y) * np.log(1 - p))[ch])
                ll_m -= np.sum((y * np.log(pm) + (1 - y) * np.log(1 - pm))[ch])
                counts += y
                tot += 1.
            # d2-5x2-weak is a deliberately weak-signal world (gap .047 nats / channel,
            # 6-7 % of the marginal loss; decisions.md D-3 outcome): smaller bar
            self.assertLess(ll_f, ll_m * (.95 if name == "d2-5x2-weak" else .9), name)

    def test_filter_equals_brute_force_on_v1_worlds(self):
        for name, steps in [("noisy-cue-delay3", 5), ("context-hidden", 3), ("count3", 4)]:
            w = t1_worlds_v1()[name]
            env, rng = WorldEnv(w, 7), np.random.default_rng(2)
            xs, acts = [env.reset()], []
            for _ in range(steps):
                acts.append(int(rng.integers(2)))
                xs.append(env.step(acts[-1]))
            f = Filter(w)
            f.observe(xs[0])
            for t in range(1, len(xs)):
                f.update(acts[t - 1], xs[t])
            acts.append(0)
            np.testing.assert_allclose(f.predict(0), _brute_force_predictive(w, xs, acts),
                                       atol=1e-12, err_msg=name)

    def test_counter_fires_after_every_third_cue(self):
        w = t1_worlds_v1()["count3"]
        env = WorldEnv(w, 1)
        x, count, fired = env.reset(), 0, 0
        for _ in range(3000):
            y = env.step(0)
            if count == 3:
                self.assertEqual(y[1], 1.)
                self.assertEqual(y[0], 0.)
                count, fired = 0, fired + 1
            else:
                self.assertEqual(y[1], 0.)
            count += int(y[0] > .5)
        self.assertGreater(fired, 100)

    def test_context_reversal_switches_the_outcome_channel(self):
        w = t1_worlds()["context-reversal"]
        env = WorldEnv(w, 5)
        env.reset()
        seen = {0: set(), 1: set()}
        for _ in range(4000):
            c_before = env.c          # the outcome depends on the context of the previous step
            x = env.step(0)
            for ch in (1, 2):
                if x[ch] > .5:
                    seen[c_before].add(ch)
        self.assertEqual(seen, {0: {1}, 1: {2}})


    def test_d1_worlds_sizes_and_disjoint_channels(self):
        from envs.family import t1_worlds_d1
        sizes = {"d-3x3": 27, "d-6x2": 64, "d-4x3": 81, "d-4x2-noise4": 256, "d-mixed": 60}
        for name, w in t1_worlds_d1().items():
            self.assertEqual(len(w.joint()[0]), sizes[name])
            chans = [c for p in w.processes for c in p.channels]
            self.assertEqual(sorted(chans), list(range(w.n_channels)))

    def test_known_structure_references_predict_every_channel(self):
        from envs.family import t1_worlds_d1
        from eval.predictive import make
        w = t1_worlds_d1()["d-mixed"]
        env = WorldEnv(w, 0)
        x = env.reset()
        for name in ("kfact", "kjoint"):
            pr = make(name, w.n_channels, w.n_actions, w.value_index, 0, w)
            p = pr.predict(x, 0)
            self.assertEqual(p.shape, (w.n_channels,))
            self.assertTrue(np.all((p > 0.) & (p < 1.)))
            pr.learn(x, 0, env.step(0))


if __name__ == "__main__":
    unittest.main()
