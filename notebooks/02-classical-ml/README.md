# Stage 02 — Classical Machine Learning

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Compare different inductive biases rather than treating one fitted model as a universal solution. Connect algorithmic choices to statistical and computational consequences.

**Prerequisites:** Stage 01, especially cross-validation, regularization, probability losses, and leakage control.

## Ordered units

1. Probability, information, and decisions: likelihood/MAP scaling, proper losses, entropy/cross-entropy/KL, mutual information, multiclass logits, and Bayes risk.
2. Nearest neighbors: distance, scaling, neighbor search, dimensionality, and local predictions.
3. Generative classifiers: class-conditional models, Gaussian discriminant analysis, Naive Bayes, smoothing, and log probabilities.
4. Decision trees: impurity as expected loss, split search, stopping, pruning, and instability.
5. Tree ensembles: bootstrap aggregation, feature subsampling, forests, additive boosting, and gradient boosting.
6. SVMs and kernels: margins, constrained objectives, dual intuition, kernel validity, and small-scale optimization.
7. Statistical learning theory: hypothesis classes, finite-class concentration, capacity, VC intuition, margins, and limits of guarantees.

<!-- availability:start -->

- probability information and decisions: **planned**
- nearest neighbors: **planned**
- generative classifiers: **planned**
- decision trees: **planned**
- tree ensembles: **planned**
- support vector machines and kernels: **planned**
- statistical learning theory: **planned**
- classical benchmark: **planned**

<!-- availability:end -->

## Implementation requirements

Keep core algorithms in NumPy. Introduce any established-library comparator only after the manual version is explained and checked; it is a benchmark reference, not the exercise implementation.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain each algorithm's representation and assumptions and the role of its hyperparameters.
- Implement small instances with explicit stopping rules and bounded compute.
- Verify split scores, posterior calculations, kernel matrices, and optimization conditions on hand-checkable cases.
- Evaluate all candidates with matched splits and selection budgets, including runtime, memory, failure cases, and seed variability.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Classical benchmark: compare several manual classifiers/regressors on shared synthetic and small licensed datasets. Include constant and linear baselines, preprocessing, validation effort, complexity, and error analysis.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

CPU only. Bound kernel matrices, distance tensors, tree counts, and search grids; report arithmetic/memory differences rather than allocating unbounded all-pairs arrays.

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

[CS229](https://cs229.stanford.edu/index.html-spr26) and the original papers or official documentation selected when each unit is authored.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.


[Return to the learning path](../../README.md)
