# Stage 05 — Deep Learning Architectures

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Use tensor/autograd infrastructure while continuing to build and explain model architectures and training loops.

**Prerequisites:** Stage 04, including manual gradients and the small autodiff engine. Stage 03 representation concepts are helpful.

## Ordered units

1. PyTorch bridge: tensors, devices, autograd, modes, parameter/state management, reproducibility, and an explicit training loop.
2. Convolutional and residual networks: receptive fields, parameter sharing, convolution shapes, residual gradient paths, and image classification.
3. Recurrent sequence models: unrolled graphs, BPTT, padding/masks, RNNs, LSTMs, GRUs, and controlled sequence tasks.
4. Embeddings and transfer learning: learned representations, frozen versus trainable components, source-task mismatch, and evaluation.
5. Self-supervised representations: contrastive and masked targets, augmentations/invariances, paired-modal intuition, representation collapse, and leakage-controlled downstream evaluation.

<!-- availability:start -->

- pytorch bridge: **planned**
- convolutional and residual networks: **planned**
- recurrent sequence models: **planned**
- embeddings and transfer learning: **planned**
- self supervised representations: **planned**
- image and sequence models: **planned**

<!-- availability:end -->

## Implementation requirements

Introduce PyTorch as an optional stage dependency at authoring time. Derive and check small operations, then assemble architectures explicitly. Prebuilt complete architectures are comparators, not substitutes for the learner's model.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain the architecture's computations and how tensor/autograd infrastructure relates to the manual foundations.
- Implement model assembly, losses, data iteration, checkpointing, and training/evaluation loops.
- Verify shapes, gradients, masking, mode transitions, and save/reload equivalence.
- Evaluate an image and sequence model with declared parameter counts, runtime, memory, baselines, and seed variation.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Image and sequence models: construct two small models, overfit a tiny diagnostic batch, train within the local budget, and compare quality and computational cost.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

Correctness checks run on CPU. Standard GPU presets must fit 5 GiB VRAM on the 6 GiB laptop GPU; published measurements must identify dtype, batch, input size, and software versions.

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

[Official PyTorch documentation](https://pytorch.org/docs/stable/index.html) and primary CNN, residual-network, LSTM/GRU, and representation-learning papers selected during authoring.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.


[Return to the learning path](../../README.md)
