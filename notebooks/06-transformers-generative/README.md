# Stage 06 — Transformers and Generative Models

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Construct a complete small language-model pipeline whose tokenizer, data boundaries, architecture, training objective, generation, and evaluation are all understood.

**Prerequisites:** Stage 05 and the probability, optimization, and evaluation conventions of earlier stages.

## Ordered units

1. Generative-modeling foundations: autoregressive factorization, latent-variable/variational objectives, denoising/score intuition, vector fields, and sampling versus likelihood. This shared overview is required; full branch implementations below are optional.
2. Tokenization and text data: bytes/Unicode, BPE, train/validation/test document boundaries, licensed sources, small streaming data, and token accounting.
3. Attention: query/key/value projections, scaling, causal/padding masks, multi-head shapes, and numerical checks.
4. Decoder Transformers: residual blocks, normalization, positional representations, feed-forward layers, tied embeddings, and next-token objectives.
5. Sequence-architecture comparisons: recurrence, attention, state-space intuition, conditional expert computation, inductive assumptions, and arithmetic/memory tradeoffs on small examples.
6. Language-model training and generation: data packing, optimization, checkpointing, held-out loss, perplexity, sampling, and reproducibility.
7. Optional branch: variational autoencoders and latent-variable objectives.
8. Optional branch: small diffusion processes, denoising objectives, and score interpretation.
9. Optional branch: flow matching, conditional paths, vector-field regression, and small ODE sampling experiments.
10. Optional branch: multimodal representation alignment on tiny paired datasets.

<!-- availability:start -->

- generative modeling foundations: **planned**
- tokenization and text data: **planned**
- attention: **planned**
- decoder transformers: **planned**
- sequence architecture comparisons: **planned**
- language model training and generation: **planned**
- variational autoencoders: **planned** (optional branch)
- diffusion models: **planned** (optional branch)
- flow matching: **planned** (optional branch)
- multimodal introduction: **planned** (optional branch)
- decoder language model: **planned**

<!-- availability:end -->

## Implementation requirements

Use PyTorch tensor operations and explicit model/training code. Implement attention and the decoder rather than importing a complete Transformer or pretrained LM. Train tokenizer statistics only on the training partition.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain every tensor shape, causal dependency, loss shift, normalization, and generation decision.
- Implement a tokenizer-to-generated-text pipeline for a small decoder-only LM.
- Verify masks, shifted targets, attention, checkpoint continuation, and held-out evaluation against independent tiny cases.
- Evaluate actual learning against simple token baselines, document data leakage controls, and bound conclusions to the small model and corpus.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Decoder language model: train a tiny model from random initialization on a small licensed/local corpus; publish configuration, tokenizer/data provenance, loss curves, evaluation, generated samples, and measured resource use.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

CPU correctness cases and small local-GPU training presets. Model/context/batch settings are finalized only after measurement within 5 GiB VRAM and the standard runtime budget. No cloud service or downloaded large pretrained checkpoint is required.

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

[CS336](https://cs336.stanford.edu/) for the tokenizer/model/optimizer/data pipeline, plus the primary Transformer, positional-encoding, normalization, and generation sources selected during authoring.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Main route and optional branches

The decoder-language-model capstone is the gate to Stage 07. VAEs, diffusion,
and the multimodal introduction enrich this stage but are not prerequisites
for language-model systems. Registry entries mark these three branches
optional. The stage's coverage status describes authored material, not a demand
that every learner complete every optional branch.

[Return to the learning path](../../README.md)
