# Stage 01 — Machine Learning Fundamentals

**Availability:** ready. Ready units are linked below; planned units
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
- [regularization and generalization](regularization-and-generalization.ipynb): **ready**
- [evaluation and model selection](evaluation-and-model-selection.ipynb): **ready**
- [Capstone: ml pipeline](../../projects/01-ml-pipeline/ml-pipeline.ipynb): **ready**

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

Measured locally on 2026-10-22: Python 3.12.15, NumPy 2.5.4,
Matplotlib 3.11.2, Ryzen 7 6800H; CPU only and float64 by default.
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Each phase uses a fresh kernel.
Elapsed time includes kernel startup, plotting, and shutdown. The approximate
memory metric is a 20 ms sampled sum of driver/kernel descendant RSS (shared
pages counted per process), excluding the browser; it is not exact peak or
array-only memory.

| Unit | Demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- |
| Linear regression | 1.633 / 189.1 | 1.862 / 194.6 |
| Gradient descent | 2.064 / 210.4 | 2.735 / 213.8 |
| Logistic regression | 1.448 / 202.7 | 2.073 / 211.6 |
| Regularization and generalization | 1.745 / 184.0 | 1.713 / 190.1 |
| Evaluation and model selection | 1.466 / 184.2 | 1.615 / 191.9 |
| ML pipeline project | 1.396 / 189.2 | 2.691 / 194.1 |

Reference phases execute all displayed Python in order plus demonstrations,
excluding answers. The project uses 300 rows per task, five features, five
training-only folds, four penalties, and 800 full-batch updates per fit, followed
by training-only fitting, validation threshold selection and final reporting.
The uncertainty exercises use 600 resamples. No GPU allocation is required.
These supplied runs fit well below the 6 GiB host limit; larger variants and
learner implementations need separate measurements. Two hours is a resumable
study-session budget, not a required training duration. Reproduce calculations
by entering displayed snippets in order in a disposable notebook copy.

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
