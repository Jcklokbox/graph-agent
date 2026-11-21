"""Short counterfactual of the largest exposed v60 distal-delay deficit.

Run from development/ with ``python -m eval.scripts.distal_contact_probe``.
Includes a one-contact intervention and a separate, task-blind coverage control.
"""
import json

import numpy as np

from agents.v60 import Agent
from envs.tasks import SPLITS, VARIANTS
from models.factor_rollout import provisional_values
from research.cue_coupling import compare_cues


SEED, STEPS, CONTACT = 1905953054, 160, 40
CLEAN, NOISY = 'dist-delay3', 'dist-delay3-noise2'


class ActionOverride:
    """Force the real action after drawing the nominal choice as usual."""

    def __init__(self, rng):
        self.rng = rng
        self.force = None

    def random(self, *args, **kwargs):
        return self.rng.random(*args, **kwargs)

    def choice(self, *args, **kwargs):
        self.nominal = int(self.rng.choice(*args, **kwargs))
        return self.nominal if self.force is None else self.force


def episode(name, *, seed=SEED, contact=False, hide_first_cue=False, coverage=False):
    if (hide_first_cue and not contact) or (coverage and contact):
        raise ValueError('invalid combination of controls')
    _, _, eb, bb = SPLITS['dev']
    variant = VARIANTS[name]
    env = variant.make(eb + seed)
    agent = Agent(variant.n_inputs, variant.n_actions, 11, seed=bb + seed)
    agent.rng = ActionOverride(agent.rng)
    factor, obs = agent.factor, env.reset()
    head, original = factor.reward, factor.two_step_values
    snapshots, trace, interventions = {}, [], []

    def capture(current, uniforms, theta=None):
        q = original(current, uniforms, theta)
        if coverage and (head.slots >= 0).any():
            origin = int(head.code(factor._bits(current)))
            visits = head.trans_total[:, origin]
            if visits.any() and (visits == 0).any():
                agent.rng.force = int(np.where(visits == 0, q, -np.inf).argmax())
        if t in (CONTACT, 60, 80, 120, STEPS - 1):
            b = factor._bits(current)
            origin = int(head.code(b))
            rule = head.rule()
            snapshots[t] = dict(q=q.tolist(), fixed=provisional_values(factor, current, uniforms, False).tolist(),
                                slots=head.slots.tolist(), visits=head.trans_total[:, origin].tolist(),
                                credited=rule[0][rule[1]].tolist(), coupling=compare_cues(factor, current, uniforms))
        return q

    factor.two_step_values = capture
    for t in range(STEPS):
        agent.rng.force = 2 if contact and t == CONTACT else None
        act = agent.step(obs)
        if act != agent.rng.nominal:
            interventions.append(t)
        next_obs, reward, _, _ = env.step(act)
        if hide_first_cue and t == CONTACT:
            # Deliver the same arrival and payoff, but hide its initial cue.
            if not next_obs[3]:
                raise AssertionError('expected a cue at the forced arrival')
            next_obs = next_obs.copy()
            next_obs[3] = 0.
        trace.append((int(obs[:3].argmax()), int(obs[3]), act, float(reward)))
        agent.learn(next_obs)
        obs = next_obs
    rows = np.asarray(trace)
    return dict(variant=name, seed=seed, contact=contact, hide_first_cue=hide_first_cue,
                coverage=coverage, overrides=interventions,
                arrivals=np.bincount(rows[:, 0].astype(int), minlength=3).tolist(),
                cues=int(rows[:, 1].sum()), foods=int((rows[:, 3] == 1.).sum()),
                tail_mean=float(rows[-80:, 3].mean()), snapshots=snapshots)


def main():
    for contact, hide in ((False, False), (True, False), (True, True)):
        print(json.dumps(episode(CLEAN, contact=contact, hide_first_cue=hide)), flush=True)
    for name, seed in ((CLEAN, SEED), (NOISY, SEED), (CLEAN, 1774893116), ('canc-noise3', 15)):
        for coverage in (False, True):
            if name == CLEAN and seed == SEED and not coverage:
                continue  # already emitted above
            print(json.dumps(episode(name, seed=seed, coverage=coverage)), flush=True)


if __name__ == '__main__':
    main()
