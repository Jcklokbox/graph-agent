"""Blind Gate environment and benchmark policies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
@dataclass(frozen=True)
class StepInfo:
    step:int; position:int; action:int; precursor_side:int|None; food_side:int|None; got_food:bool; received_shock:bool
class BlindGateEnv:
    LEFT=0; CENTER=1; RIGHT=2
    def __init__(self,seed:int=0,event_interval:tuple[int,int]=(4,5),noise_probability:float=0.5):
        lo,hi=event_interval
        if lo<2 or hi<lo: raise ValueError("event_interval must satisfy 2 <= low <= high")
        if not 0<=noise_probability<=1: raise ValueError("noise_probability must be in [0, 1]")
        self.seed=int(seed); self.event_interval=(int(lo),int(hi)); self.noise_probability=float(noise_probability); self.rng=np.random.default_rng(self.seed); self.reset()
    def reset(self,seed:int|None=None)->np.ndarray:
        if seed is not None: self.seed=int(seed)
        self.rng=np.random.default_rng(self.seed); self.step_count=0; self.position=1; self.wellbeing=.5; self._precursor_side=None; self._food_side=None
        self._next_event_step=int(self.rng.integers(self.event_interval[0],self.event_interval[1]+1)); return self._observation()
    def _observation(self)->np.ndarray:
        obs=np.zeros(12,dtype=np.float32); obs[self.position]=1
        if self._precursor_side==0: obs[3]=1
        elif self._precursor_side==1: obs[4]=1
        if self._food_side==0: obs[5]=1
        elif self._food_side==1: obs[6]=1
        obs[7]=float(self.rng.random()<self.noise_probability); obs[11]=self.wellbeing; return obs
    @staticmethod
    def _decode_action(action:int|np.ndarray|list[float])->int:
        if isinstance(action,(int,np.integer)):
            r=int(action)
            if r not in (0,1,2): raise ValueError("action must be 0, 1, or 2")
            return r
        v=np.asarray(action,dtype=float)
        if v.shape!=(3,) or not np.all(np.isfinite(v)): raise ValueError("action scores must be a finite vector of shape (3,)")
        return int(np.argmax(v))
    def step(self,action:int|np.ndarray|list[float])->tuple[np.ndarray,float,bool,StepInfo]:
        chosen=self._decode_action(action); old_precursor=self._precursor_side
        self.position=int(np.clip(self.position+chosen-1,0,2)); self.step_count+=1
        self._food_side=old_precursor; self._precursor_side=None
        if self.step_count==self._next_event_step:
            self._precursor_side=int(self.rng.integers(0,2)); gap=int(self.rng.integers(*self.event_interval,endpoint=True)); self._next_event_step=self.step_count+gap
        got=(self.position==0 and self._food_side==0) or (self.position==2 and self._food_side==1)
        shock=self.position!=1 and not got; self.wellbeing=1.0 if got else (0.0 if shock else .5)
        info=StepInfo(self.step_count,self.position,chosen,self._precursor_side,self._food_side,bool(got),bool(shock))
        return self._observation(),self.wellbeing,False,info

def oracle_policy(obs:np.ndarray)->int:
    if obs[0]>.5:return 2
    if obs[2]>.5:return 0
    if obs[3]>.5:return 0
    if obs[4]>.5:return 2
    return 1
def food_reactive_policy(obs:np.ndarray)->int:
    if obs[0]>.5:return 2
    if obs[2]>.5:return 0
    if obs[5]>.5:return 0
    if obs[6]>.5:return 2
    return 1
def stay_policy(_:np.ndarray)->int:return 1
def random_three_policy(rng:np.random.Generator)->Callable[[np.ndarray],int]:return lambda _obs:int(rng.integers(0,3))
def random_direction_policy(rng:np.random.Generator)->Callable[[np.ndarray],int]:return lambda _obs:int(rng.choice((0,2)))
def evaluate(policy:Callable[[np.ndarray],int],seeds:range|list[int]=range(10),steps:int=20000)->dict[str,float]:
    rewards=[]; shocks=meals=total=0
    for seed in seeds:
        env=BlindGateEnv(seed); obs=env.reset()
        for _ in range(steps):
            obs,r,_,info=env.step(policy(obs)); rewards.append(r); shocks+=int(info.received_shock); meals+=int(info.got_food); total+=1
    return {"mean_W":float(np.mean(rewards)),"shock_rate":shocks/total,"food_rate":meals/total,"steps":float(total)}
def run_benchmarks(seeds:int=20,steps:int=20000)->dict[str,dict[str,float]]:
    ss=list(range(seeds)); result={"oracle":evaluate(oracle_policy,ss,steps),"stay":evaluate(stay_policy,ss,steps)}
    for name,factory in (("random_uniform_3_actions",random_three_policy),("random_left_right_only",random_direction_policy)):
        rewards=[]; shocks=meals=total=0
        for seed in ss:
            env=BlindGateEnv(seed); obs=env.reset(); rng=np.random.default_rng(1000000+seed); policy=factory(rng)
            for _ in range(steps):
                obs,r,_,info=env.step(policy(obs)); rewards.append(r); shocks+=int(info.received_shock); meals+=int(info.got_food); total+=1
        result[name]={"mean_W":float(np.mean(rewards)),"shock_rate":shocks/total,"food_rate":meals/total,"steps":float(total)}
    result["reactive_to_food"]=evaluate(food_reactive_policy,ss,steps); return result
if __name__=="__main__":
    for name,metrics in run_benchmarks().items(): print(f"{name:25s} "+" ".join(f"{k}={v:.6f}" for k,v in metrics.items()))
