# Stage 08 — Advanced Specializations and Research

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Practice research through a focused, falsifiable reproduction rather than treating exposure to every specialty as mastery.

**Prerequisites:** Stage 01 supplies the minimum experimental-design foundation. Use the track-specific prerequisites below; the tracks are independent.

## Ordered units

1. Reinforcement learning: MDPs, Bellman equations, tabular control, policy gradients, PPO, and evaluation on small environments.
2. Causal inference: causal graphs, interventions, confounding, identification assumptions, and synthetic studies with known mechanisms.
3. Interpretability: feature/representation interventions, attribution sanity checks, and evaluation of explanations.
4. Uncertainty estimation: probabilistic predictions, calibration, uncertainty sources, and distribution shift.
5. Mixture of experts: routing, sparse expert computation, load balancing, and small-model comparisons.
6. Advanced multimodal systems: small paired-data objectives, modality alignment, retrieval, and failure analysis.
7. Retrieval-augmented generation: corpus/chunking choices, retrieval and generation metrics, leakage controls, and small-system ablations.

<!-- availability:start -->

- reinforcement learning: **planned**
- causal inference: **planned**
- interpretability: **planned**
- uncertainty estimation: **planned**
- mixture of experts: **planned**
- advanced multimodal: **planned**
- retrieval augmented generation: **planned**
- research reproduction: **planned**

<!-- availability:end -->

## Implementation requirements

Choose one track and one bounded claim. Reuse established infrastructure only where it is not the object being studied. Implement the technique being investigated and document all departures from the original paper.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain the paper's claim, assumptions, metrics, and which part can be investigated locally.
- Implement a baseline and the selected technique with an explicit hypothesis and controlled changes.
- Verify the implementation and repeat the experiment with recorded data, seeds, configurations, and budgets.
- Evaluate ablations, uncertainty, negative findings, and reproduction fidelity in a technical report.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Research reproduction: select one local-scale claim, define the study before seeing results, establish baselines, run ablations/repeats, and publish evidence and limitations. Catalog coverage is separate from completion of an individual chosen track.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

All required studies must fit the same laptop budgets. Prefer small environments, synthetic causal data, tiny models, and bounded retrieval corpora. A paper's full-scale compute setup is not automatically a required experiment.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Readings

Primary papers chosen per track, with [CS229](https://cs229.stanford.edu/index.html-spr26) supplying statistical/RL context and [CS336](https://cs336.stanford.edu/) supplying modern LM/systems context.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Track-specific entry points

| Track | Additional preparation |
| --- | --- |
| RL | Stage 04 for neural policies; Stage 00 probability for tabular work |
| Causal inference | Stage 00 probability/statistics and Stage 01 evaluation |
| Interpretability | Stage 04; Stage 06 for Transformer investigations |
| Uncertainty | Stage 03 probabilistic models and Stage 01 calibration |
| MoE | Stages 06–07 for routing and resource comparisons |
| Multimodal systems | Stage 05 and the optional Stage 06 introduction |
| Retrieval-augmented generation | Stage 06 and retrieval/evaluation foundations |

The learner chooses one track; none requires completing all the others.

[Return to the learning path](../../README.md)
