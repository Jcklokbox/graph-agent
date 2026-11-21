"""AR-6 decision trace: per-step Q decomposition for the exposed canc rows.

Records, for every step: bits, action, reward, the learned reward likelihood per
control state, the provisional bound, the two-step Q vector, the chosen action
and (when a payoff is reachable) what each action would have earned. Read-only
instrumentation of the deployed controller; nothing is injected into learning.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from agents import agent_class, load
from envs.tasks import SPLITS, VARIANTS
from harness.journal import append


def q_terms(head, f, x, u):
    """Mirror provisional_values for read-only diagnostics; return per-action terms."""
    b = f._bits(x)
    rule = head.rule()
    atoms = rule[3]
    trans = head.all_transitions()
    code = int(head.code(b))
    input0 = head.input(b, f.hist)[1:]
    r, p, _, _ = head.likelihood(rule, input0[None, :])
    splits = head.signed_rules(rule) if hasattr(head, 'signed_rules') else []
    seen_phase = head.phase(b, f.hist)
    atoms0 = atoms
    if splits:
        r2, p2, outcome, _, group0 = head.context_moment(
            splits, input0[None, :], seen_phase[None, :], r, p, atoms)
        atoms0 = outcome[0]
    r, p = r[0], p[0]
    incumbent = r.copy()
    bound, _ = head.provisional(rule) if hasattr(head, 'provisional') else (None, None)
    if bound is not None:
        incumbent = np.maximum(r, (bound * seen_phase).max(1))
    active = head.slots >= 0
    cols = head.slots[active]
    states = (np.flatnonzero(np.any(trans[:, code, :] > 0., axis=0))
              if getattr(head, 'sparse_outcomes', False) else head.ci)
    out = dict(code=code, states=states.tolist(), r=np.round(r[states], 4).tolist(),
               inc=np.round(incumbent[states], 4).tolist(),
               trans=np.round(trans[:, code, states], 3).tolist(),
               p=np.round(p[states], 3).tolist(),
               bound_states=[None if bound is None else np.round(bound[states].max(0), 3).tolist()])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('variant', nargs='+')
    p.add_argument('--model', default='v48')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--every', type=int, default=1)
    p.add_argument('--partial', action='store_true',
                   help='perturb every move decision by measuring counterfactual reward')
    a = p.parse_args()
    tree = os.environ.get('PG_TREE', 'local')
    eb, bb = SPLITS['dev'][2], SPLITS['dev'][3]
    for variant in a.variant:
        for seed in a.seeds:
            v = VARIANTS[variant]
            env = v.make(eb + seed)
            brain = agent_class(load(a.model))(v.n_inputs, v.n_actions, 11, seed=bb + seed)
            f = brain.factor
            head = f.reward
            x = env.reset()
            rows = []
            for t in range(a.steps):
                u = brain.rng.random((brain.draws, f.J + 3 + int(f.reward_control)))
                q = f.two_step_values(x, u, None)
                try:
                    terms = q_terms(head, f, x, u)
                except Exception as exc:  # instrumentation must not change the run
                    terms = {'err': repr(exc)}
                act = brain.step(x)
                y, r, _, info = env.step(act)
                brain.learn(y)
                if t % a.every == 0:
                    cue, lamp = x[3] > .5, x[7] > .5
                    row = dict(t=t, b=[int(z) for z in f._bits(x)], a=int(act),
                               q=np.round(q, 4).tolist(), cue=int(cue), lamp=int(lamp),
                               pos=int(np.argmax(x[:3])), r=float(r),
                               slots=head.slots.tolist(),
                               col=terms.get('code'), states=terms.get('states'),
                               inc=terms.get('inc'), tr=terms.get('trans'),
                               prob=terms.get('p'), bnd=terms.get('bound_states'))
                    rows.append(row)
                x = y
            R = [z['r'] for z in rows]
            sumry = dict(n=len(rows), mean=float(np.mean(R[-200:])),
                         payoffs=int(sum(1 for z in rows if z['r'] == 1.)),
                         shocks=int(sum(1 for z in rows if z['r'] == 0.)),
                         late_shocks=int(sum(1 for z in rows[-200:] if z['r'] == 0.)),
                         visits=head.control_mass.tolist(), slots=head.slots.tolist(),
                         anchors=head.anchors.tolist() if hasattr(head, 'anchors') else None,
                         assoc=np.round(head.assoc, 2).tolist() if hasattr(head, 'anchors') else None,
                         splits=[[int(j), int(m), int(l), int(s.sum()), round(float(np.asarray(g).sum()), 3)]
                                 for j, m, l, s, g, d in (head.signed_rules(head.rule())
                                                          if hasattr(head, 'signed_rules') else [])],
                         onset_total=head.onset_total.round(1).tolist(),
                         present=head.present.round(0).tolist(),
                         rule_idx=head.rule()[0].tolist(), rule_w=head.rule()[1].astype(int).tolist(),
                         ctx=(head.ctx_n.sum(axis=(0, 1, 2, 4)).round(1).tolist()
                              if hasattr(head, 'ctx_n') else None),
                         rows=rows)
            append(a.out, dict(k=[tree, seed, f'{a.model}|{variant}|trace|{a.steps}'], m=sumry))
            print(variant, seed, {k: sumry[k] for k in ('mean', 'payoffs', 'shocks', 'late_shocks',
                                                        'slots', 'anchors', 'splits')}, flush=True)


if __name__ == '__main__':
    from harness.beacon import cli

    with cli():
        main()
