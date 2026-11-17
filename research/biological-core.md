# Biological core

## Motivation and epistemic status

The project investigates whether adaptive behavior can emerge from learned relations among sensory activity, internally generated actions, predicted outcomes, time, and wellbeing. The conceptual inspiration comes from animal learning and neural adaptation: predictive association, surprise-sensitive learning, blocking, conditional inhibition, traces across delays, habituation, novelty-oriented exploration, and value transfer through predictive chains.

These concepts are research motivations, not evidence that the current software reproduces a biological circuit. The implementation is a computational agent in a small synthetic task. It has not been validated against neural recordings, animal behavior, or a comprehensive biological model. Use cautious language: “inspired by,” “analogous to,” or “tests a hypothesis about,” unless stronger evidence is supplied.

## Mapping hypotheses to prototype

- **Predictiveness and background:** the learner estimates base activity rates and compares edge outcome estimates against references. This resembles prediction-relative learning in broad outline, but exact correspondence to Rescorla–Wagner learning, associability, or attention remains unproven.
- **Blocking/parsimony:** higher-order antecedents consult lower-order estimates. Whether this reliably blocks redundant compounds while admitting useful compounds is an empirical question.
- **Inhibition:** edge records include `q_minus`, and cancellation tasks probe suppression. The update rule and behavior should be evaluated for conditionality, retention of positive links, renewal, and reversal; do not infer Pavlovian inhibition from the field name.
- **Temporal association:** edge-local traces decay, register coarse lag statistics, and discharge when targets occur. This is a computational memory mechanism, not a validated synaptic or neural trace model.
- **Value:** wellbeing is a designated observation channel and action scoring uses wellbeing-related estimates. This is not equivalent to identifying a biological reward circuit or demonstrating secondary reinforcement.
- **Novelty and orienting:** treat local novelty-driven exploration as a stated project principle requiring direct implementation inspection and dedicated tests. Do not assume it follows from random exploration or from the existing graph structure.

## Research directions

The theoretical workspace can ask which learning phenomena are best represented by local predictive relations, and what observations would discriminate that account from alternatives. Candidate topics include attentional competition, habituation, latent inhibition, conditioned inhibition, trace conditioning, outcome devaluation, reversal/renewal, curiosity, and secondary reinforcement. For each, specify a task, behavioral signature, competing explanation, and limits on biological interpretation.

Separate three evidence levels in notes and reports:

1. **Implementation fact:** what the checked-in code does.
2. **Computational result:** what controlled simulations demonstrate under reported conditions.
3. **Biological claim:** what is supported by external empirical evidence or a validated model.

Do not move from level 1 or 2 to level 3 without independent evidence.
