# Stage 01 — Machine Learning Fundamentals

**Availability:** partial. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Turn predictive questions into explicitly defined models, objectives, fitting procedures, and defensible evaluations. Complete the foundations before expanding the algorithm family.

**Prerequisites:** Stage 00 diagnostic-level competence. No prior ML framework experience is required.

## Ordered units

1. Linear regression: prediction arrays, residual geometry, half-MSE, gradients, closed-form one-feature fitting, identifiability, and held-out evaluation.
2. Gradient descent: local approximations, stability, full/stochastic/mini-batch gradients, scaling, work budgets, and reproducibility.
3. Logistic regression: odds, likelihood, stable cross-entropy, regularized gradients, probabilities, and decisions.
4. Regularization and generalization: empirical/population risk, squared-loss bias-variance, learning curves, L2 objectives, ridge fitting, and conceptual L1 comparison.
5. Evaluation and model selection: baselines, regression/classification/probability metrics, calibration, thresholds, cross-validation, leakage, and finite-sample uncertainty.

<!-- availability:start -->

- [linear regression](linear-regression.ipynb): **ready**
- [gradient descent](gradient-descent.ipynb): **ready**
- [logistic regression](logistic-regression.ipynb): **ready**
- regularization and generalization: **planned**
- evaluation and model selection: **planned**
- ml pipeline: **planned**

<!-- availability:end -->

## Implementation requirements

Implement models, gradients, preprocessing, folds, metrics, and selection procedures using NumPy. Implement L2; study L1's sparsity and nondifferentiability conceptually in this delivery. Do not import fitted estimators.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain which assumptions justify the prediction family, loss, and evaluation procedure.
- Implement regression and classification with explicit averaging and an unpenalized intercept.
- Verify gradients against an independently defined objective and protect shapes and numerical domains.
- Evaluate against simple baselines using training-only preprocessing, validation-based choices, and one final test evaluation.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

ML pipeline: work on deterministic synthetic regression and classification tasks; implement training-only standardization, training-fold cross-validation, validation threshold selection, numerical gradient audits, and a final report with one held-out test evaluation.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

CPU only. Small datasets and bounded search grids are sufficient; no downloaded data, accelerator, or paid service is required.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Readings

[CS229](https://cs229.stanford.edu/index.html-spr26) for supervised learning, model selection, and learning theory; [Stanford's public CS229 notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) as optional alternative explanations.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Using the retained chapters

The first three chapters preserve their original exercises, demonstrations, and
worked references. Their elementary algebra/calculus refreshers are optional
for the assumed audience. The new applied review in Stage 00 and the later
Stage 01 chapters provide the additional statistical and evaluation depth.

[Return to the learning path](../../README.md)
