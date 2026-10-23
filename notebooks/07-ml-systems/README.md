# Stage 07 — Advanced Training and ML Systems

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Improve a correct language model through measured systems and data changes while protecting the validity of quality comparisons.

**Prerequisites:** The Stage 06 decoder-language-model capstone; the optional generative-model branches are not required.

## Ordered units

1. Profiling and resource accounting: warmup, synchronization, timing distributions, FLOPs, activation/optimizer memory, and bottlenecks.
2. Efficient training and attention: mixed precision, accumulation, checkpointing, attention algorithms, correctness/performance comparisons, and conditional use of specialized kernels.
3. Distributed computation: communication costs and tiny CPU multi-process/all-reduce demonstrations; connect these to data/model parallel training concepts.
4. Scaling and data quality: small-budget scaling experiments, uncertainty/extrapolation limits, filtering, deduplication, data mixtures, and document-level evaluation.
5. Inference and quantization: KV caching, batching, latency/throughput, quantization error, and resource/quality tradeoffs.
6. Fine-tuning and LoRA: full versus low-rank updates, parameter/memory accounting, and small supervised adaptation.
7. Reinforcement learning for post-training: small MDPs, returns, action probabilities, policy gradients, baselines, reward/KL objectives, and independently checked toy examples. Advanced RL remains a Stage 08 track.
8. Preference optimization and evaluation: distinguish SFT, reward-based RL, and preference objectives such as DPO; verify sequence probabilities and compare under declared selection/evaluation budgets.
9. Inference-time computation and search: sampling/selection, bounded search, verifiers, budget-dependent evaluation, and small model-assisted decision processes.

<!-- availability:start -->

- profiling and resource accounting: **planned**
- efficient training and attention: **planned**
- distributed computation: **planned**
- scaling and data quality: **planned**
- inference and quantization: **planned**
- fine tuning and lora: **planned**
- reinforcement learning for post training: **planned**
- preference optimization and evaluation: **planned**
- inference time computation and search: **planned**
- language model systems: **planned**

<!-- availability:end -->

## Implementation requirements

Extend the Stage 06 project. Add performance tooling only when it serves a measured objective and works on the target machine. Specialized kernels must have a correct reference comparison; unsupported hardware features are not required.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain measured bottlenecks, precision/communication effects, and the mathematical adaptation objectives.
- Implement selected memory, inference, and adaptation techniques on the existing small LM.
- Verify equivalence/tolerances, cache behavior, accumulation, update restrictions, and checkpoint state before timing.
- Evaluate memory, throughput, runtime, and quality under declared matched conditions and report failed or neutral optimizations.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Language-model systems: optimize the previous model, compare a fixed baseline, add a small adaptation experiment, and produce reproducible quality/resource tables with uncertainty and limitations.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

One 6 GiB laptop GPU is the maximum required accelerator. Distributed exercises use small CPU processes; multi-GPU benchmarks are outside the required path. Scaling-law results from tiny runs must state their limited scale.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Research connection

Each developed unit includes a bounded primary-source study with an equation,
independent check, and a claim the local experiment can support. See the
[foundations map](../../docs/foundations-and-connections.md) and
[research practice](../../docs/research-practice.md). New coverage remains
planned until its notebook and measurements pass publication gates.

## Readings

[CS336](https://cs336.stanford.edu/) for systems, scaling, data, and post-training; official tool documentation and primary technique papers selected at authoring time.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.


[Return to the learning path](../../README.md)
