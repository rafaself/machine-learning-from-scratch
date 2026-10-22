# Stage 00 — Mathematical Tools for Machine Learning

**Availability:** planned. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Apply existing calculus, linear algebra, and programming knowledge to ML arrays, statistical objectives, and finite-precision computation. This is a compact applied review, not an elementary mathematics course.

**Prerequisites:** Multivariable differentiation, vectors/matrices, and Python programming. The probability chapter develops the statistical conventions needed by the rest of the path.

## Ordered units

1. Arrays and matrix calculus: row-wise observations; broadcasting; Jacobian orientation; JVPs and VJPs; scalar matrix gradients; Hessian-vector products; central differences.
2. Probability and statistics: conditioning and Bayes' rule; expectation and covariance; Bernoulli/categorical/Gaussian models; likelihood; estimators; uncertainty and resampling.
3. Numerical computing: float32/float64; cancellation; stable log-domain calculations; conditioning; solves; independent RNG streams; bounded-memory computations.

<!-- availability:start -->

- arrays and matrix calculus: **planned**
- probability and statistics: **planned**
- numerical computing: **planned**
- mathematical audit: **planned**

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
