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
4. Momentum and Adam: stateful updates, bias correction, schedules, optimizer comparisons, and checkpoint state.
5. Normalization and dropout: training/evaluation behavior, statistical axes, randomness, and reproducibility.
6. Automatic differentiation: build a small reverse-mode engine only after deriving gradients manually; accumulation, graph reuse, broadcasting reductions, and elementary operations.

<!-- availability:start -->

- [perceptron](perceptron.ipynb): **ready**
- [backpropagation](backpropagation.ipynb): **ready**
- [multilayer perceptron](multilayer-perceptron.ipynb): **ready**
- momentum and adam: **planned**
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

CPU only. Small batches, widths, and graphs keep engine/debugging work inspectable. Large-model training is introduced after this foundation.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

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
