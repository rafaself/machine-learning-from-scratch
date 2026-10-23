# Stage 04 — Neural Networks and Deep Learning Foundations

**Availability:** partial. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Understand differentiation, optimization, representation, and generalization as distinct parts of neural-network learning.

**Prerequisites:** Stage 01. This stage can be studied now after the available foundations; completing planned Stages 02–03 is not required first.

## Ordered units

1. Perceptron: label conventions, signed margins, updates, convergence assumptions, and representation limits.
2. Backpropagation: shared paths, cached parameter snapshots, batched affine/activation rules, and gradient checking.
3. Multilayer perceptron: one/two hidden layers, initialization, manual training, regularization, and validation.
4. Multiclass learning objectives: categorical likelihood, stable softmax, cross-entropy/KL, Jacobians, label conventions, and representation/decision distinctions.
5. Momentum and Adam: stateful updates, bias correction, schedules, optimizer comparisons, and checkpoint state.
6. Optimization geometry and diagnostics: curvature/spectrum, stochastic gradient variation, activation/gradient distributions, initialization, saturation, and controlled failure analysis.
7. Normalization and dropout: training/evaluation behavior, statistical axes, randomness, and reproducibility.
8. Automatic differentiation: build a small reverse-mode engine after deriving gradients manually; accumulation, graph reuse, broadcasting reductions, and elementary operations.

<!-- availability:start -->

- [perceptron](perceptron.ipynb): **ready**
- [backpropagation](backpropagation.ipynb): **ready**
- [multilayer perceptron](multilayer-perceptron.ipynb): **ready**
- multiclass learning objectives: **planned**
- momentum and adam: **planned**
- optimization geometry and diagnostics: **planned**
- normalization and dropout: **planned**
- automatic differentiation: **planned**
- autodiff and network: **planned**

<!-- availability:end -->

## Implementation requirements

Use NumPy. The autodiff engine is a learner-built algorithm, not a framework dependency. Compare its derivatives against manual backpropagation and finite differences.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain the graph, shapes, reductions, optimizer state, and evaluation-mode behavior.
- Implement both explicit network derivatives and a small reusable differentiation engine.
- Verify every parameter, shared path, broadcast reduction, and optimizer state transition.
- Evaluate training across seeds and fixed splits while separating optimization failures from representation or generalization limits.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Autodiff and network: validate the engine against independently derived gradients, then use it to train a small network and compare numerical behavior, work, and predictions with the manual version.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

The three retained chapters were rechecked locally on 2026-10-22 using
Python 3.12.15, NumPy 2.5.4, Matplotlib 3.11.2, CPU-only Ryzen 7 6800H,
and one OpenBLAS/OpenMP thread. These are measurements of available chapters,
not certification of the planned optimizer/autodiff material or full capstone.

| Retained unit | Fresh-kernel demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- |
| Perceptron | 4.017 / 197.0 | 1.789 / 205.5 |
| Backpropagation | 1.417 / 200.1 | 2.165 / 215.4 |
| Multilayer perceptron | 1.497 / 203.0 | 9.859 / 234.7 |

Elapsed time includes kernel startup, plotting, and shutdown. RSS is the
approximate 20 ms sampled sum for the driver and kernel descendants, including
shared pages counted per process and excluding browser memory. Each reference
phase also runs its demonstrations, omits answer cells, and leaves canonical
outputs untouched. No GPU is required for these supplied presets.

CPU only. Small batches, widths, and graphs keep engine/debugging work inspectable. Large-model training is introduced after this foundation.

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

[CS229](https://cs229.stanford.edu/index.html-spr26) for neural-network foundations; primary reverse-mode differentiation and optimizer sources selected as chapters are developed.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Current coverage

The original perceptron, backpropagation, and MLP chapters are available with
their original exercise IDs. The additional optimizer, normalization, and
autodiff units and the integrated capstone remain planned. Three available
chapters do not mean that the full stage has passed its completion gate.

[Return to the learning path](../../README.md)
