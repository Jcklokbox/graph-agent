"""Fast validation for the index-agnostic predictive graph controller."""
from __future__ import annotations
import inspect, json, time
from pathlib import Path
import numpy as np
from blind_gate import BlindGateEnv
from blind_gate_graph import BlindGateGraphBrain, make_observation
from test_autonomous_and_generalization import CancellationGateEnv, VariableDelayEnv, DistantGateEnv
from test_blind_gate_graph import controlled_transitions, train_transitions, chunking_transitions

class PermutedEnv:
    def __init__(self, env, seed: int, sensor_perm=None, action_perm=None):
        self.env=env
        rng=np.random.default_rng(seed)
        self.sensor_perm=np.asarray(sensor_perm if sensor_perm is not None else rng.permutation(8),dtype=int)
        self.action_perm=np.asarray(action_perm if action_perm is not None else rng.permutation(3),dtype=int)
    def reset(self): return self._obs(self.env.reset())
    def _obs(self, raw):
        out=np.asarray(raw,dtype=float).copy()
        out[:8]=np.asarray(raw,dtype=float)[self.sensor_perm]
        return out
    def step(self, action):
        raw,reward,done,info=self.env.step(int(self.action_perm[action]))
        return self._obs(raw),reward,done,info
    def reverse(self):
        self.action_perm[[0,2]]=self.action_perm[[2,0]]
        # Exchange the two precursor input wires without assigning their role
        # inside the brain.
        a,b=np.where(self.sensor_perm==3)[0][0],np.where(self.sensor_perm==4)[0][0]
        self.sensor_perm[a],self.sensor_perm[b]=self.sensor_perm[b],self.sensor_perm[a]


def test_1_inspection_and_control():
    source=inspect.getsource(BlindGateGraphBrain)
    forbidden=("(3, 4)","(5, 6)","[3:5]","[5:7]","[0:3]")
    inspection=not any(token in source for token in forbidden)
    b=BlindGateGraphBrain(seed=0)
    train_transitions(b,controlled_transitions(),80)
    links={
      "V3_to_V5":b.q_pred(3,5), "V4_to_V6":b.q_pred(4,6),
      "position_action_left":b.q_pred((1,8),0),
      "false_pair":b.q_pred((1,8),1),
      "right_to_center":b.q_pred((2,8),1),
      "position_action_right":b.q_pred((1,10),2),
      "left_to_center":b.q_pred((0,10),1),
    }
    redundant={"right_to_center_single_max":max(b.q_pred(2,1),b.q_pred(8,1)),
               "left_to_center_single_max":max(b.q_pred(0,1),b.q_pred(10,1))}
    # Third-order composition check on the historical controlled sequence.
    cb=BlindGateGraphBrain(seed=11)
    train_transitions(cb,chunking_transitions(),24)
    third_order=max((cb.q_pred((chunk, cue), 11) for chunk in cb.chunk_components for cue in cb.sensor_nodes), default=0.0)
    # Noise check uses an independent random policy and a fresh default brain.
    env=BlindGateEnv(seed=19); obs=env.reset(); n=BlindGateGraphBrain(seed=20); rng=np.random.default_rng(21)
    for _ in range(300):
        a=int(rng.integers(0,3)); n.forward(obs,a); nxt,_,_,_=env.step(a); n.learn(nxt); obs=nxt
    noise_edges=[q for (ants,j),q in n.pred_pool.nodes.items() if 7 in ants]
    noise_max=max(noise_edges,default=0.0)
    positive_links=[links[k] for k in ("V3_to_V5","V4_to_V6","position_action_left","right_to_center","position_action_right","left_to_center")]
    return {"inspection_no_sensor_roles":inspection,"links":links,"single_link_baselines":redundant,
            "third_order_max_Q":third_order,"noise_max_Q":noise_max,
            "pass":inspection and min(positive_links)>0.90 and links["false_pair"]<0.01
                   and noise_max<0.85 and third_order>0.50}


def test_2_permuted_autonomous():
    rows=[]
    for seed in range(5):
        env=PermutedEnv(BlindGateEnv(seed=seed),seed+900)
        obs=env.reset(); brain=BlindGateGraphBrain(seed=100+seed); rewards=[]
        for _ in range(1000):
            action=brain.forward(obs,None); nxt,r,_,_=env.step(action); brain.learn(nxt); obs=nxt; rewards.append(r)
        tail=np.asarray(rewards[-300:]); rows.append({"seed":seed,"mean_W":float(tail.mean()),"shock_rate":float(np.mean(tail==0.0))})
    return {"runs":rows,"pass":all(x["mean_W"]>0.56 and x["shock_rate"]<0.03 for x in rows)}


def test_3_permuted_cancellation_and_delay():
    # Cancellation with a random input/output wiring.
    env=PermutedEnv(CancellationGateEnv(seed=31),seed=31); obs=env.reset(); b=BlindGateGraphBrain(seed=31)
    for _ in range(900):
        a=b.forward(obs,None); nxt,_,_,_=env.step(a); b.learn(nxt); obs=nxt
    # Measure the learned cancellation on the actual permuted sensor channels.
    trap_input=np.zeros(12); safe_input=np.zeros(12)
    # Construct two contexts by mapping the base environment's center+cue/lamp.
    base=np.zeros(12); base[1]=1; base[3]=1
    for i,orig in enumerate(env.sensor_perm):
        safe_input[i]=base[orig]
    base[7]=1
    for i,orig in enumerate(env.sensor_perm): trap_input[i]=base[orig]
    safe_input[11]=trap_input[11]=0.5
    safe_a=b.select_action(safe_input); trap_a=b.select_action(trap_input)
    expected_safe=int(np.where(env.action_perm==0)[0][0]); expected_center=int(np.where(env.action_perm==1)[0][0])
    # Variable delay, separately permuted; default brain configuration.
    de=PermutedEnv(VariableDelayEnv(seed=47),seed=47); x=de.reset(); db=BlindGateGraphBrain(seed=47)
    for _ in range(900):
        a=db.forward(x,1); nxt,_,_,_=de.step(a); db.learn(nxt); x=nxt
    cue_channel=int(np.where(de.sensor_perm==3)[0][0]); food_channel=int(np.where(de.sensor_perm==5)[0][0])
    temporal=db.q_pred_temporal(cue_channel,food_channel)
    cue_channel_cancel=int(np.where(env.sensor_perm==3)[0][0])
    food_channel_cancel=int(np.where(env.sensor_perm==5)[0][0])
    red_channel=int(np.where(env.sensor_perm==7)[0][0])
    q_plus=b.q_pred(cue_channel_cancel,food_channel_cancel)
    q_minus=b.q_inhib(red_channel,food_channel_cancel)
    return {"Q_plus":q_plus,"Q_minus_red_to_food":q_minus,
            "Q_minus_max":max((q for ((ants,j),q) in b.pred_inhib.nodes.items()),default=0.0),
            "safe_action":safe_a,"cancel_action":trap_a,"temporal":temporal,
            "pass":q_plus>0.85 and q_minus>0.85 and safe_a==expected_safe and trap_a==expected_center
                   and temporal["fast"]<0.1 and temporal["mid"]>0.85 and temporal["slow"]>0.85}


def test_4_permuted_distant_gate():
    env=PermutedEnv(DistantGateEnv(seed=59),seed=59); obs=env.reset(); b=BlindGateGraphBrain(seed=59); rewards=[]
    for _ in range(1000):
        a=b.forward(obs,None); nxt,r,_,_=env.step(a); b.learn(nxt); obs=nxt; rewards.append(r)
    tail=np.asarray(rewards[-300:])
    return {"mean_W":float(tail.mean()),"shock_rate":float(np.mean(tail==0.0)),
            "solo_values":{str(k):v for k,v in b.solo_w.items()},
            "pair_values":{str(k):v for k,v in b.pair_w.items()},
            "pass":float(tail.mean())>0.53 and float(np.mean(tail==0.0))<0.05}


def test_5_reversal():
    env=PermutedEnv(BlindGateEnv(seed=77),seed=77); obs=env.reset(); b=BlindGateGraphBrain(seed=77); rewards=[]
    first=[]; second=[]
    for step in range(1200):
        if step==600: env.reverse()
        a=b.forward(obs,None); nxt,r,_,_=env.step(a); b.learn(nxt); obs=nxt; rewards.append(r)
        if 400<=step<600: first.append(r)
        if 900<=step<1200: second.append(r)
    first=np.asarray(first); second=np.asarray(second)
    return {"phase1_mean_W":float(first.mean()),"phase2_mean_W":float(second.mean()),
            "phase2_shock_rate":float(np.mean(second==0.0)),
            "reversal_resets":b.reversal_reset_count,
            "pass":float(first.mean())>0.56 and float(second.mean())>0.55 and float(np.mean(second==0.0))<0.05}


def run():
    start=time.perf_counter()
    results={"test_1":test_1_inspection_and_control(),"test_2":test_2_permuted_autonomous(),
             "test_3":test_3_permuted_cancellation_and_delay(),"test_4":test_4_permuted_distant_gate(),
             "test_5":test_5_reversal()}
    results["runtime_seconds"]=time.perf_counter()-start
    results["all_pass"]=all(results[f"test_{i}"]["pass"] for i in range(1,6)) and results["runtime_seconds"]<10.0
    return results

if __name__=="__main__":
    result=run(); print(json.dumps(result,indent=2,default=lambda x:float(x)))
    Path(__file__).with_name("despecialized_test_results.json").write_text(json.dumps(result,indent=2,default=lambda x:float(x))+"\n")
    if not result["all_pass"]: raise SystemExit("One or more criteria failed")
