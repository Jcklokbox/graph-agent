"""Short end-to-end scenarios for PosteriorGraph; never invokes the legacy brain.

    python run_posterior_scenarios.py           probes + single episodes (~10 s)
    python run_posterior_scenarios.py --dev     8 DEV seeds only, for model selection
    python run_posterior_scenarios.py --suite   everything incl. 12 HELD-OUT seeds

Episodes run in parallel processes (PG_PROCS=1 forces serial; same output).
"""
from __future__ import annotations
import random
import numpy as np
from posterior_graph import PosteriorGraph
from blind_gate import BlindGateEnv
from benchmark_envs import CancellationGateEnv, VariableDelayEnv, DistantGateEnv


class Relabel:
    def __init__(self, env, sensors, actions):
        self.env, self.sensors, self.actions = env, tuple(sensors), tuple(actions)
    def reset(self): return self.map(self.env.reset())
    def map(self, obs):
        out=np.asarray(obs,dtype=float).copy()
        out[:8]=np.asarray(obs,dtype=float)[list(self.sensors)]
        return out
    def step(self, action):
        physical=self.actions[action]
        obs,r,d,info=self.env.step(physical)
        return self.map(obs),r,d,info
    def reverse(self):
        self.actions=(self.actions[2],self.actions[1],self.actions[0])
        p=list(self.sensors); a,b=p.index(3),p.index(4); p[a],p[b]=p[b],p[a]; self.sensors=tuple(p)


def interact(env, brain, n, noise_rng=None):
    obs=env.reset(); rewards=[]; shocks=0
    for _ in range(n):
        a=brain.step(obs)
        nxt,r,_,info=env.step(a)
        brain.learn(nxt)
        shock = info.get("received_shock",False) if isinstance(info,dict) else getattr(info,"received_shock",False)
        rewards.append(float(r)); shocks+=int(bool(shock))
        obs=nxt
    return float(np.mean(rewards[-min(n,200):])),shocks/n


def structure_probe():
    out={}
    for kind in ("AND","noise"):
        b=PosteriorGraph(4,2,3,seed=1,pair_min_samples=5)
        for i in range(80):
            a,x=(i//2)%2,i%2
            y=(a&x) if kind=="AND" else int((i*7%13)<6)
            b._pending=((float(a),float(x),0.,.5),1); b.learn((0.,0.,0.,float(y)))
        out[kind]=(len(b.births),b.predict((1.,1.,0.,.5),1,3))
    return out


def inhibition_probe():
    b=PosteriorGraph(4,2,3,seed=2)
    for obs,y in [((1,0,0,0),1),((1,0,1,0),0),((0,1,0,0),1)]:
        for _ in range(30):
            b._pending=(tuple(map(float,obs)),1); b.learn((0.,0.,0.,float(y)))
    states={name:b.predict(obs,1,3) for name,obs in {
        "A":(1.,0.,0.,0.),"AX":(1.,0.,1.,0.),
        "B":(0.,1.,0.,0.),"BX":(0.,1.,1.,0.)}.items()}
    return states


def permutation_closed_loop(steps=40, results=None):
    """Exhaust 3! sensor relabelings x 3! action relabelings in paired loops."""
    jobs = equivariance_jobs(steps)
    if results is None:
        results = dict(zip(jobs, pmap(_job, jobs)))
    checked = 0
    for job in jobs:
        ok, n, msg = results[job]
        checked += n
        if not ok:
            return False, checked, msg
    return True, checked, "all paired closed-loop transitions/actions exact"


def equivariance_jobs(steps=40):
    from itertools import permutations
    return [("equivariance", tuple(s3), tuple(am), steps)
            for s3 in permutations(range(3)) for am in permutations(range(3))]


def permutation_closed_loop_one(s3, action_map, steps=40):
    """One paired closed loop: physical learner vs relabeled learner."""
    checked = 0
    sensor_map=tuple(s3)+(3,4,5,6,7)
    inverse=[action_map.index(i) for i in range(3)]
    e0=BlindGateEnv(seed=88); e1=BlindGateEnv(seed=88)
    x0=e0.reset(); x1=e1.reset()
    b0=PosteriorGraph(12,3,11,seed=19); b1=PosteriorGraph(12,3,11,seed=19)
    rng=random.Random(901); wrapper=Relabel(e1,sensor_map,action_map)
    for _ in range(steps):
        noise=[rng.random() for _ in range(3)]
        noise1=[0.]*3
        for a in range(3): noise1[inverse[a]]=noise[a]
        a0=b0.step(x0,tie_noise=noise)
        a1=b1.step(wrapper.map(x1),tie_noise=noise1)
        if a1 != inverse[a0]: return False,checked,"action mismatch"
        x0n,r0,_,_=e0.step(a0); x1n,r1,_,_=e1.step(action_map[a1])
        x1m=wrapper.map(x1n)
        if r0 != r1 or not np.array_equal(wrapper.map(x0n),x1m):
            return False,checked,"transition mismatch"
        b0.learn(x0n); b1.learn(x1m); x0,x1=x0n,x1n
        checked += 1
    return True,checked,"exact"


def cancellation_oracle(obs):
    if obs[0] > .5: return 2
    if obs[2] > .5: return 0
    if obs[3] > .5 and obs[7] < .5: return 0
    return 1


def distant_oracle_factory():
    state = {"mission": 0}
    def policy(obs):
        if obs[2] > .5 and obs[3] > .5:
            state["mission"] = 2
        if state["mission"] > 0:
            state["mission"] -= 1
            return 0
        if obs[0] > .5 or obs[1] > .5:
            return 2
        return 1
    return policy


def run_policy(env, policy, n):
    obs = env.reset(); rewards = []; shocks = 0
    for _ in range(n):
        obs, r, _, info = env.step(policy(obs))
        shock = info.get("received_shock", False) if isinstance(info, dict) else getattr(info, "received_shock", False)
        rewards.append(float(r)); shocks += int(bool(shock))
    return float(np.mean(rewards[-min(n, 200):])), shocks / n


# ----------------------------------------------------------- parallel jobs
# Every brain episode is independent and fully seeded, so episodes run in a
# process pool (fork start method: workers inherit the module state, including
# a monkeypatched `PosteriorGraph`) and results are identical to a serial run.
# Set PG_PROCS=1 to force serial execution.
_ENVS = {"autonomous": BlindGateEnv, "cancellation": CancellationGateEnv,
         "distant-value": DistantGateEnv, "variable-delay": VariableDelayEnv}


def pmap(fn, jobs):
    import os, multiprocessing as mp
    jobs = list(jobs)
    procs = int(os.environ.get("PG_PROCS", os.cpu_count() or 1))
    if procs <= 1 or len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as pool:
        return pool.starmap(fn, jobs, chunksize=1)


def brain_episode(task, env_seed, brain_seed, n):
    return interact(_ENVS[task](seed=env_seed), PosteriorGraph(12, 3, 11, seed=brain_seed), n)


def reversal_episode(env_seed, brain_seed, first=(200, 400), after=(600, 800)):
    env = Relabel(BlindGateEnv(seed=env_seed), (4,0,2,3,1,5,6,7), (2,1,0))
    brain = PosteriorGraph(12, 3, 11, seed=brain_seed); obs = env.reset(); r_first = []; r_after = []
    for t in range(after[1]):
        if t == 400: env.reverse()
        a = brain.step(obs); obs, r, _, _ = env.step(a); brain.learn(obs)
        if first[0] <= t < first[1]: r_first.append(r)
        if after[0] <= t < after[1]: r_after.append(r)
    return float(np.mean(r_first)), float(np.mean(r_after))


def _job(kind, *args):
    return {"brain": brain_episode, "reversal": reversal_episode,
            "equivariance": permutation_closed_loop_one, "hazard": hazard_bins_one}[kind](*args)


def multi_seed_suite(seeds=range(8), env_base=100, brain_base=0, results=None):
    """Brain vs stay/random/oracle on identical env seeds; tail = last 200 steps.

    `results` may hold precomputed brain episodes (see `suite_jobs`)."""
    from blind_gate import oracle_policy
    specs = [("autonomous", BlindGateEnv, oracle_policy, 600),
             ("cancellation", CancellationGateEnv, cancellation_oracle, 600),
             ("distant-value", DistantGateEnv, None, 800)]
    if results is None:
        jobs = suite_jobs(seeds, env_base, brain_base)
        results = dict(zip(jobs, pmap(_job, jobs)))
    rows = {}
    for name, cls, oracle, n in specs:
        res = {"brain": [], "brain_shock": [], "stay": [], "random": [], "oracle": []}
        for sd in seeds:
            m, sh = results[("brain", name, env_base + sd, brain_base + sd, n)]
            res["brain"].append(m); res["brain_shock"].append(sh)
            res["stay"].append(run_policy(cls(seed=env_base + sd), lambda o: 1, n)[0])
            rr = random.Random(sd)
            res["random"].append(run_policy(cls(seed=env_base + sd), lambda o: rr.randrange(3), n)[0])
            pol = oracle if oracle is not None else distant_oracle_factory()
            res["oracle"].append(run_policy(cls(seed=env_base + sd), pol, n)[0])
        rows[name] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
        wins = sum(b > s + 1e-9 for b, s in zip(res["brain"], res["stay"]))
        rows[name]["brain>stay seeds"] = f"{wins}/{len(res['brain'])}"
        from math import comb
        k_n = len(res["brain"])
        rows[name]["sign-test p"] = round(sum(comb(k_n, k) for k in range(wins, k_n + 1)) / 2 ** k_n, 4)
    rev_b, rev_a = [], []
    for sd in seeds:
        before, after = results[("reversal", env_base + 100 + sd, brain_base + sd)]
        rev_b.append(before); rev_a.append(after)
    rows["reversal"] = {"before": (float(np.mean(rev_b)), float(np.std(rev_b))),
                        "after": (float(np.mean(rev_a)), float(np.std(rev_a)))}
    return rows


def suite_jobs(seeds, env_base, brain_base):
    """Job keys of one multi-seed suite, longest episodes first."""
    jobs = [("reversal", env_base + 100 + sd, brain_base + sd) for sd in seeds]
    for name, n in (("distant-value", 800), ("autonomous", 600), ("cancellation", 600)):
        jobs += [("brain", name, env_base + sd, brain_base + sd, n) for sd in seeds]
    return jobs


def hazard_bins_one(env_seed, brain_seed, n=600):
    """cue3 -> event5 (events, exposures) per lag for one VariableDelay learner."""
    env = VariableDelayEnv(seed=env_seed); brain = PosteriorGraph(12, 3, 11, seed=brain_seed); obs = env.reset()
    for _ in range(n):
        a = brain.step(obs); obs, _, _, _ = env.step(a); brain.learn(obs)
    bins = {}
    for (cue, target, age, action), (ev, risk) in brain.hazard.items():
        if cue == 3 and target == 5:
            e, r = bins.get(age, (0., 0.)); bins[age] = (e + ev, r + risk)
    return bins


def hazard_calibration(seeds=range(5), results=None):
    """Pooled cue3->event5 conditional hazards vs truth (delay uniform on {2,3,4})."""
    jobs = [("hazard", 300 + sd, sd) for sd in seeds]
    if results is None:
        results = dict(zip(jobs, pmap(_job, jobs)))
    bins = {}
    for job in jobs:
        for age, (ev, risk) in results[job].items():
            e, r = bins.get(age, (0., 0.)); bins[age] = (e + ev, r + risk)
    truth = {1: 0.0, 2: 1/3, 3: 1/2, 4: 1.0}
    return {k: (round(e / r, 3) if r else None, truth.get(k)) for k, (e, r) in sorted(bins.items())}


def main(suite=False):
    print("structure (births, conjunction p):",structure_probe())
    print("summation/memory probabilities:",inhibition_probe())
    tasks=[("autonomous",12,600),("cancellation",31,600),("variable-delay",47,600),("distant-value",59,800)]
    jobs=[("brain",name,env_seed,5,n) for name,env_seed,n in tasks]
    jobs.append(("reversal",77,77))
    jobs+=equivariance_jobs()
    jobs.append(("hazard",47,23))
    suites=[]
    if suite:
        # Development seeds (used to choose hyperparameters) and held-out seeds.
        suites=[("DEV", dict(seeds=range(8), env_base=100, brain_base=0)),
                ("HELD-OUT", dict(seeds=range(12), env_base=5000, brain_base=900))]
        for _, kw in suites:
            jobs+=suite_jobs(**kw)
        jobs+=[("hazard",300+sd,sd) for sd in range(5)]
    # Longest episodes first for load balance; results are keyed, order-free.
    order=sorted(set(jobs), key=lambda j: -(j[-1] if j[0]=="brain" else 800 if j[0]=="reversal" else 600 if j[0]=="hazard" else 80))
    results=dict(zip(order, pmap(_job, order)))
    for name,env_seed,n in tasks:
        mean,shock=results[("brain",name,env_seed,5,n)]
        print(f"{name}: tail_mean={mean:.3f}, shock_fraction={shock:.3f}, steps={n}")
    # Reversal in the same learner; the environment changes cue/action mapping.
    before,after=results[("reversal",77,77)]
    print(f"reversal: before={before:.3f}, after={after:.3f}")
    print("closed_loop_equivariance:",permutation_closed_loop(results=results))
    # Delay model calibration summary for cue-channel 3 to event-channel 5.
    bins=results[("hazard",47,23)]
    print("cue3->event5 hazard bins (lag: events/exposures):",
          {k:(round(e,1),round(r,1)) for k,(e,r) in sorted(bins.items())})
    for label, kw in suites:
        print("=== %s ===" % label)
        for name, row in multi_seed_suite(results=results, **kw).items():
            print(name, {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v) for k, v in row.items()})
    if suite:
        print("hazard calibration (lag: estimated, true):", hazard_calibration(results=results))


def dev_only():
    """Quick model-selection check: the 8 DEV seeds only (never the held-out ones)."""
    kw = dict(seeds=range(8), env_base=100, brain_base=0)
    jobs = suite_jobs(**kw)
    results = dict(zip(jobs, pmap(_job, jobs)))
    rows = multi_seed_suite(results=results, **kw)
    for name, row in rows.items():
        print(name, {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v) for k, v in row.items()})
    return rows


if __name__=="__main__":
    import sys
    if "--dev" in sys.argv:
        dev_only()
    else:
        main(suite="--suite" in sys.argv)
