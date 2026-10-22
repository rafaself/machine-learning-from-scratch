# Stage 00 — Mathematical Tools for Machine Learning

**Availability:** ready. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Apply existing calculus, linear algebra, and programming knowledge to ML arrays, statistical objectives, and finite-precision computation. This is a compact applied review, not an elementary mathematics course.

**Prerequisites:** Multivariable differentiation, vectors/matrices, and Python programming. The probability chapter develops the statistical conventions needed by the rest of the path.

## Ordered units

1. Arrays and matrix calculus: row-wise observations; broadcasting; Jacobian orientation; JVPs and VJPs; scalar matrix gradients; Hessian-vector products; central differences.
2. Probability and statistics: conditioning and Bayes' rule; expectation and covariance; Bernoulli/categorical/Gaussian models; likelihood; estimators; uncertainty and resampling.
3. Numerical computing: float32/float64; cancellation; stable log-domain calculations; conditioning; solves; independent RNG streams; bounded-memory computations.

<!-- availability:start -->

- [arrays and matrix calculus](arrays-and-matrix-calculus.ipynb): **ready**
- [probability and statistics](probability-and-statistics.ipynb): **ready**
- [numerical computing](numerical-computing.ipynb): **ready**
- [Capstone: mathematical audit](../../projects/00-mathematical-audit/mathematical-audit.ipynb): **ready**

<!-- availability:end -->

## Implementation requirements

Use NumPy and Matplotlib. Matrix operations, probability calculations, gradient checks, and streaming accumulations are learner implementations; NumPy linear algebra supplies numerical primitives.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain the distinction between a Jacobian, gradient, JVP, and VJP, including shapes and reduction axes.
- Implement shape-safe calculations without accidental sample-to-sample broadcasting.
- Verify likelihood gradients, covariance divisors, and stable computations against independent calculations.
- Evaluate roundoff, conditioning, sampling uncertainty, and memory costs without confusing them.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Mathematical audit: estimate Gaussian summaries, evaluate a stable Gaussian log-likelihood without an inverse, check the mean gradient, compare precision/conditioning, and produce an assumption-and-failure report.

Submit learner work on a study branch. The reference supplies instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

Measured locally on 2026-10-22: Python 3.12.15, NumPy 2.5.4,
Matplotlib 3.11.2, Ryzen 7 6800H, CPU only, float64 by default,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Fresh-kernel elapsed time includes
startup, plotting, and shutdown. Memory is the sampled sum of RSS for the
validation driver and its kernel descendants at 20 ms intervals, including
shared pages counted per process; it is an approximate upper accounting metric,
not isolated array memory or an exact peak. Browser memory is excluded.

| Unit | Demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- |
| Arrays and matrix calculus | 1.516 / 180.8 | 2.016 / 193.4 |
| Probability and statistics | 1.412 / 188.4 | 1.591 / 192.9 |
| Numerical computing | 1.594 / 187.2 | 1.680 / 190.7 |
| Mathematical audit project | 1.449 / 191.6 | 1.547 / 193.3 |

Reference runs include the chapter demonstrations and all displayed Python
snippets in order, in a separate kernel per notebook. Answer cells are excluded.
The largest required generated dataset here has 1200 scalar observations;
the Gaussian project uses 512 training and 256 held-out rows of three features.
No GPU allocation is required. These measurements establish that the supplied
presets fit the budget; learner implementations and larger variants need their
own measurements. Reproduce the calculations by entering the displayed snippets
in order in a disposable copy, keeping the reference notebook unchanged.

CPU only; tiny matrices and synthetic samples. Resource measurements are added when the published chapters and project are validated.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Readings

[NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [covariance conventions](https://numpy.org/doc/stable/reference/generated/numpy.cov.html), [linear solves](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), and the probability/linear-algebra refreshers linked from [CS229](https://cs229.stanford.edu/index.html-spr26).

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Diagnostic and skip route

Experienced learners may attempt the diagnostic exercises at the beginning of
the three chapters before reading their explanations. Produce the requested
shape traces, derivatives, probability calculations, and numerical failure
explanations, then compare with the separately collapsed references.

Skip a chapter's review only when you can explain those answers and reproduce
its numerical checks independently. The mathematical-audit capstone is the
integrated readiness check; familiarity with calculus alone does not establish
statistical or numerical readiness.

[Return to the learning path](../../README.md)
