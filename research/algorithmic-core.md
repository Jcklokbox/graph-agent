# Algorithmic core

## Prototype at a glance

`development/blind_gate.py` defines a partially observed three-position environment. An event precursor indicates which side will receive food on a later step; food is delivered only when the agent occupies that side, while visiting the wrong outer side is costly. Observations also contain position, food, noise, and wellbeing channels.

`development/blind_gate_graph.py` implements `BlindGateGraphBrain`. It encodes observations together with candidate actions into source features, activates antecedent sets, maintains learned directed predictive relations to target features, and evaluates candidate actions using predicted outcomes and value estimates. Its learning path updates graph edges from observed transitions, including temporally structured relations. The implementation includes compatibility-facing query and snapshot methods and legacy antecedent handling.

The tests and JSON files in `development/` exercise graph behavior, generalization, autonomous play, temporal delays, and task-specific scenarios. Their presence records test intent; it does not imply all behavioral targets currently pass. `development/README.md` describes the available commands and known limitations.

## Research perspective

The broader algorithmic idea is to learn reusable predictive structure from interaction rather than encode fixed sensor-role rules. Symmetric or index-agnostic feature treatment, action-conditioned transition prediction, temporal association, and value-guided action selection are central themes in the present prototype. They are useful organizing ideas, not restrictions on future methods. Research may compare alternate graph representations, inference procedures, exploration strategies, credit assignment, or planning horizons, even when those ideas are not yet represented in code.
