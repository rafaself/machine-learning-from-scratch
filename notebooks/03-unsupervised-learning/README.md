# Stage 03 — Unsupervised and Probabilistic Learning

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Discover and model structure without confusing a visually persuasive representation with validated evidence.

**Prerequisites:** Stage 00 probability/matrix tools and Stage 01 experimental design. Stage 02 is recommended for algorithm comparisons but is not a mathematical prerequisite.

## Ordered units

1. Clustering: K-means, initialization, hierarchical linkage, distance choices, and cluster assessment.
2. Gaussian mixtures and EM: latent assignments, likelihood, responsibilities, lower-bound intuition, monotonicity assumptions, and singularities.
3. PCA and SVD: centering, orthogonal projections, variance and reconstruction objectives, rank, and scaling.
4. Density estimation and anomaly detection: density versus anomaly scores, threshold selection, contamination, and distribution shift.

<!-- availability:start -->

- clustering: **planned**
- gaussian mixtures and em: **planned**
- pca and svd: **planned**
- density estimation and anomaly detection: **planned**
- representation study: **planned**

<!-- availability:end -->

## Implementation requirements

Use NumPy with stable log responsibilities and explicit convergence diagnostics. NumPy's SVD/eigensolvers are numerical primitives; PCA preprocessing and objectives remain visible.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain objective differences between clustering, density modeling, and dimensionality reduction.
- Implement the essential updates and transformations, including training-fitted centering.
- Verify EM calculations, orthogonality, reconstruction errors, and degenerate cases on small inputs.
- Evaluate multiple starts and representation choices using appropriate metrics and external labels only when explicitly justified.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Representation study: compare clustering and reduced representations on the same unlabeled data, report sensitivity to scale/initialization, and identify where the chosen evaluation could mislead.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

CPU only. Use small dense matrices and bounded pairwise calculations. Large-corpus and distributed clustering are outside the required experiments.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Readings

[CS229](https://cs229.stanford.edu/index.html-spr26) for EM, mixtures, and PCA; primary algorithm papers selected with each authored chapter.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.


[Return to the learning path](../../README.md)
