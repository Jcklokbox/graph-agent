"""Compare two coverage definitions on short v60 prefixes, without changing v60.

Serial arms fork identical states immediately before a reversal.
For the sham reset, clear the active reward mode without changing the world.
"""
from copy import deepcopy
import json

import numpy as np

from agents.v60 import Agent
from envs.tasks import SPLITS, VARIANTS
from eval.scripts.distal_contact_probe import ActionOverride, CLEAN, SEED


POLICIES = ('off', 'active', 'witnessed')


def coverage_action(active, witnessed, q, *, policy):
    """Only propose at an origin with an active visit; rank unseen moves by Q."""
    if policy == 'off' or not active.any():
        return None
    if policy == 'active':
        unseen = active == 0.
    elif policy == 'witnessed':
        unseen = ~witnessed
    else:
        raise ValueError(policy)
    return int(np.where(unseen, q, -np.inf).argmax()) if unseen.any() else None


class RevisionGate:
    """Track live departures in the newly built code, not replayed rows."""

    def __init__(self):
        self.pending = False
        self.departed = set()

    def allows(self, origin):
        return not self.pending or origin in self.departed

    def observe(self, origin, *, revised, switched):
        if revised or switched:
            self.pending = revised and not switched
            self.departed.clear()
        elif self.pending and origin is not None:
            self.departed.add(origin)


class OverrideBudget:
    """Charge actual overrides; only a newly inferred mode refills credits."""

    def __init__(self, limit, *, exhausted=False):
        if limit < 1:
            raise ValueError('coverage budget must be positive')
        self.limit = limit
        self.remaining = 0 if exhausted else limit

    def spend(self):
        if not self.remaining:
            raise AssertionError('action override without coverage credit')
        self.remaining -= 1

    def on_switch(self, *, recalled):
        if not recalled:
            self.remaining = self.limit


def warmup(name, seed, steps):
    _, _, eb, bb = SPLITS['dev']
    variant = VARIANTS[name]
    env = variant.make(eb + seed)
    agent = Agent(variant.n_inputs, variant.n_actions, 11, seed=bb + seed)
    return advance((agent, env, env.reset()), steps)


def advance(checkpoint, steps):
    agent, env, obs = checkpoint
    for _ in range(steps):
        obs, _, _, _ = env.step(agent.step(obs))
        agent.learn(obs)
    return agent, env, obs


def continuation(checkpoint, start, steps, *, policy, sham_reset=False, limit=None):
    if limit is not None and policy != 'active':
        raise ValueError('budget applies to active-bank coverage only')
    # A warm fork does not get fictitious startup credits at the checkpoint.
    budget = None if limit is None else OverrideBudget(limit, exhausted=start != 0 and not sham_reset)
    agent, env, obs = deepcopy(checkpoint)
    agent.rng = ActionOverride(agent.rng)
    factor = agent.factor
    head, original = factor.reward, factor.two_step_values
    if sham_reset:
        if not (head.slots >= 0).any():
            raise AssertionError('sham reset requires a learned controlled origin')
        head._switch(recalled=False)
    interventions, repeated, switches, recalls, revisions = [], [], [], [], []
    rewards, blocked, revision_trials = [], [], []
    gate = RevisionGate()
    original_switch = head._switch

    def capture_switch(recalled, permutation=None):
        switches.append(t)
        if recalled:
            recalls.append(t)
        original_switch(recalled, permutation)
        if budget is not None:
            budget.on_switch(recalled=recalled)

    head._switch = capture_switch

    def capture(x, uniforms, theta=None):
        nonlocal origin
        q = original(x, uniforms, theta)
        if (head.slots >= 0).any():
            origin = int(head.code(factor._bits(x)))
            if policy == 'off' or (budget is not None and not budget.remaining):
                return q
            witnessed = head.recent_trans[:, origin, :].any(axis=1)
            force = coverage_action(head.trans_total[:, origin], witnessed, q,
                                    policy='active' if policy == 'revisited' else policy)
            if force is not None and not gate.allows(origin) and policy == 'revisited':
                # A proposal was suppressed; the nominal action may still agree.
                blocked.append(t)
                force = None
            agent.rng.force = force
            agent.rng.repeat_witness = force is not None and bool(witnessed[force])
        return q

    factor.two_step_values = capture
    for t in range(start, start + steps):
        agent.rng.force, agent.rng.repeat_witness = None, False
        slots = head.slots.copy()
        origin = None
        action = agent.step(obs)
        if action != agent.rng.nominal:
            if budget is not None:
                budget.spend()
            interventions.append(t)
            if not gate.allows(origin):
                revision_trials.append(t)
            if agent.rng.repeat_witness:
                repeated.append(t)
        obs, reward, _, _ = env.step(action)
        agent.learn(obs)
        revised = not np.array_equal(slots, head.slots)
        if revised:
            revisions.append(t)
        gate.observe(origin, revised=revised, switched=bool(switches and switches[-1] == t))
        rewards.append(float(reward))
    return dict(total=float(np.sum(rewards)), mean=float(np.mean(rewards)),
                first30=float(np.mean(rewards[:30])),
                last80=float(np.mean(rewards[-80:])), overrides=interventions, repeated=repeated,
                switches=switches, recalls=recalls, revisions=revisions,
                revision_trials=revision_trials, blocked=blocked)


def main():
    for name, seed, prefixes in (('aut-serial300', 6, ((300, 80, False), (600, 80, False))),
                                 ('aut-serial300', 1822923777, ((300, 80, False),)),
                                 ('aut', 6, ((300, 80, True), (300, 80, False))),
                                 (CLEAN, SEED, ((0, 160, False),))):
        checkpoint = warmup(name, seed, 0)
        elapsed = 0
        for start, steps, sham in prefixes:
            checkpoint = advance(checkpoint, start - elapsed)
            elapsed = start
            for policy in POLICIES:
                result = continuation(checkpoint, start, steps, policy=policy, sham_reset=sham)
                print(json.dumps(dict(task=name, seed=seed, start=start, sham_reset=sham,
                                      policy=policy, **result)), flush=True)


if __name__ == '__main__':
    main()
