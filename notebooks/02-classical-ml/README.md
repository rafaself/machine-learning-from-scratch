# Stage 02 — Classical Machine Learning

**Availability:** partial. Ready units are linked below; planned units
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

- [probability information and decisions](probability-information-and-decisions.ipynb): **ready**
- [nearest neighbors](nearest-neighbors.ipynb): **ready**
- [generative classifiers](generative-classifiers.ipynb): **ready**
- [decision trees](decision-trees.ipynb): **ready**
- [tree ensembles](tree-ensembles.ipynb): **ready**
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

### Published-unit measurements

Measured on 2026-10-23 using the Ryzen 7 6800H, Python 3.12.15,
NumPy 2.5.4, Matplotlib 3.11.2, CPU only, and one OpenBLAS/OpenMP thread.
Each phase starts a fresh kernel. Time includes startup, plotting, and shutdown.
Memory is the 20 ms sampled sum of driver/kernel-descendant RSS, including
shared pages counted per process and excluding browser memory. It is approximate
aggregate accounting, not an exact physical-memory peak.

| Published unit | Demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- |
| Probability, information, and decisions | 1.411 / 179.6 | 1.624 / 186.7 |
| Nearest neighbors | 1.561 / 179.4 | 1.559 / 184.4 |
| Generative classifiers | 1.563 / 179.9 | 1.527 / 183.7 |
| Decision trees | 1.250 / 178.6 | 1.408 / 182.0 |
| Tree ensembles | 1.851 / 179.5 | 2.603 / 186.2 |

The reference phase executes demonstrations and all displayed Python in order,
excluding answer cells. This unit uses small categorical matrices, 32 normal
observations, and at most 600 Bernoulli observations per illustrated sample.
Nearest-neighbor investigations use 80 regression rows and a separate 300-row,
32-feature classification example, at most 120 validation queries, and default
32-query/128-training-row search blocks. They do not allocate a full 3D distance
tensor. Generative-classifier experiments fit three covariance assumptions to
120 training and 200 validation rows in four dimensions, for two regimes and
three seeds. Discrete examples use tiny generated count matrices and local text.
Tree comparisons use 120 training and 160 validation rows in two dimensions,
three depth limits and four pruning penalties over three seeds. A separate
80-row classification example checks four one-label perturbations against a
fixed 400-row probe set. Growth is bounded to depth six or four respectively.
Ensemble comparisons use 100 training and 160 validation rows in four dimensions,
12 depth-four trees, two candidate-feature settings and three seeds. Boosting
uses 120/160 rows in three dimensions, 12 depth-two stages and two declared losses.
Out-of-bag contributor indices are retained and uncovered rows are reported.
No GPU allocation is required. Other Stage 02 units and its capstone remain
planned and will receive separate measurements before publication.

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

The first unit's bounded reading is
[Shannon (1948), Part I §6](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf),
with an equation translation and an independently checked counterexample about
forecast entropy versus correctness. The reading does not require reproducing
the paper's full communication-theory results.

The nearest-neighbor unit studies
[Cover and Hart (1967), Section VI](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf),
deriving a conditional binary mismatch calculation and distinguishing its
large-sample assumptions from a finite experiment.

The generative-classifier unit uses
[CS229 Generative Learning Algorithms, §§1.2–1.3](https://cs229.stanford.edu/summer2020/cs229-notes2.pdf)
for a bounded equation audit of shared-covariance estimates and logistic
posterior odds, separating model-family identities from fitted performance.

The tree unit audits
[CS229 Decision Trees, §§1.2–1.5](https://cs229.stanford.edu/summer2023/cs229-notes-decision_trees.pdf),
checking weighted gain and refining monotone-transform invariance with an exact
midpoint-threshold counterexample.

The ensemble unit studies
[Breiman (2001), §§3.1 and 4](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
for OOB contributors and node-level feature sampling, with an explicit hard/soft
voting distinction and limits on internal-validation claims.

[CS229](https://cs229.stanford.edu/index.html-spr26) and the original papers or official documentation selected when each unit is authored.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.


[Return to the learning path](../../README.md)
