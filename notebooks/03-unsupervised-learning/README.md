# Stage 03 — Unsupervised and Probabilistic Learning

**Availability:** partial. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Discover and model structure without confusing a visually persuasive representation with validated evidence.

**Prerequisites:** Stage 00 probability/matrix tools and Stage 01 experimental design. Stage 02 is recommended for algorithm comparisons but is not a mathematical prerequisite.

## Ordered units

1. Clustering: K-means, initialization, hierarchical linkage, distance choices, and cluster assessment.
2. Gaussian mixtures and EM: latent assignments, likelihood, responsibilities, lower-bound derivation, monotonicity assumptions, and singularities.
3. PCA and SVD: centering, orthogonal projections, variance and reconstruction objectives, rank, spectrum, conditioning, and scaling.
4. Bayesian and variational inference: conjugate posterior/predictive examples, point versus distributional estimates, latent variables, Jensen/KL bounds, and approximation limits.
5. Density estimation and anomaly detection: density versus anomaly scores, threshold selection, contamination, and distribution shift.

<!-- availability:start -->

- [clustering](clustering.ipynb): **ready**
- [gaussian mixtures and em](gaussian-mixtures-and-em.ipynb): **ready**
- [pca and svd](pca-and-svd.ipynb): **ready**
- bayesian and variational inference: **planned**
- density estimation and anomaly detection: **planned**
- representation study: **planned**

<!-- availability:end -->

## Implementation requirements

Use manual NumPy objectives, updates and assessment calculations. The published
clustering chapter specifies initialization, tie rules, empty-cluster policies,
iteration status, linkage units and partition-metric boundary cases. The
mixture calculations use stable log responsibilities, manually derived likelihood
gradients, a fixed covariance constraint and explicit gain/budget/failure
diagnostics.
The published PCA chapter implements training-fitted centering/scaling,
projection/reconstruction, whitening and independent objective diagnostics.
NumPy's SVD/eigensolvers are numerical primitives; PCA preprocessing and
objectives remain visible. Bayesian/variational and density units remain planned.

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

Measured on 2026-10-23 on the Ryzen 7 6800H laptop with Python 3.12.15,
NumPy 2.5.4 and Matplotlib 3.11.2, with BLAS/OpenMP threads set to one:

| Published unit | Demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- |
| Clustering | 1.646 / 180.6 | 2.766 / 198.3 |
| Gaussian mixtures and EM | 1.269 / 180.2 | 1.661 / 190.6 |
| PCA and SVD | 1.881 / 180.0 | 2.328 / 197.3 |

Each phase starts a fresh kernel; time includes startup, plots and shutdown.
Memory is the approximate 20 ms sampled sum of driver/kernel-descendant RSS,
with shared pages counted per process and browser memory excluded. It is not
an exact physical-memory peak. The reference phase runs all displayed Python
and demonstrations in chapter order, excluding learner answers. Default Run
All runs only demonstrations and leaves the displayed references unexecuted.

Clustering initialization compares two methods, eight starts each, on 180
training and 120 independent probe rows, three centers and a 100-step budget.
Fixed component counts are declared; observations are drawn independently
conditional on those allocations. Geometry/linkage comparisons use 48 rows,
two preprocessing choices and four linkage methods. The null study uses 160
training and 120 probe rows, four cluster counts with eight starts each and
four bootstraps with four starts each. All measured fits stabilize within
their budgets; tests separately retain deliberate unfinished and empty cases.

The auditable hierarchy is capped at 80 rows, silhouette at 400 rows and
exponential two-group oracles at ten rows. Query blocks bound distance scratch
arrays, while returned pairwise matrices still require explicit quadratic
storage.

The mixture comparison uses 180 training and 160 independent validation rows,
two covariance modes, two components, four starts per mode, variance floor
0.02, 100 steps and relative gain tolerance 1e-7. All eight runs stop by their
gain criterion after 22–35 iterations; that flag does not prove parameter
stationarity or a global optimum. A three-row floor stress study makes four
separate fixed-floor fits, each stopping after three updates. A two-row
auxiliary-bound audit uses direct E/M calculations. Generic fitting is capped
at 2,000 rows, eight components/features, 200 steps and 16 starts. Numerical
zero-mass/factorization failures are retained rather than silently repaired.

PCA comparisons use seven retained-dimension choices (0–6), including the
mean-only baseline, on 120 training and 160 validation rows in six features.
Three two-component geometry/whitening procedures use the same samples with a
declared unit conversion. Six tiny precision cases compare direct SVD with
formed covariance across two dtypes and three weak singular values; a four-row
target audit distinguishes variance retention from predictive information.
Fitting is capped at 2,000 rows and 64 features and uses thin factors. Whitening
rejects numerically null directions under the published tolerance. No large
sample-by-sample projection matrix is allocated.

The default experiments use no GPU and stay far below the 6 GiB host-memory
limit. Later units remain unmeasured and planned.

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

The clustering unit studies
[Arthur and Vassilvitskii, K-means++, algorithm and Theorem 1.1](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf).
After implementing D-squared seeding, translate the potential/expectation
statement and enumerate a tiny example exactly. The local study distinguishes
an expectation guarantee from a guarantee for every seed; it does not reproduce
the full approximation proof or the paper's original benchmarks.
[Modern Statistics for Modern Biology, §§5.6–5.7](https://web.stanford.edu/class/bios221/book/05-chap.html)
supplies an alternative explanation of hierarchical linkage and validation.
Audit units and objective factors when translating among sources.

The mixture unit studies
[CS229 mixture notes, pp.2–3](https://see.stanford.edu/materials/aimlcs229/cs229-notes7b.pdf)
and [EM notes, §2 equations (3)–(6)](https://see.stanford.edu/materials/aimlcs229/cs229-notes8.pdf).
After implementing posterior normalization and the bound, translate the summed
likelihood to an average. Reproduce the equality-conditioned ascent chain and
construct a finite counterexample to an arbitrary loose auxiliary update.
The local audit does not claim general parameter convergence or latent recovery.

The PCA unit studies
[CS229 PCA notes, p.2 and pp.5–6](https://see.stanford.edu/materials/aimlcs229/cs229-notes10.pdf),
translating its preprocessing and projection argument into row-wise observations
and independently checking variance/reconstruction. A finite target table
tests the limits of interpreting retained variance as predictive information.
[NumPy's SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
defines the numerical primitive's thin-factor shapes and spectrum conventions.

[CS229](https://cs229.stanford.edu/index.html-spr26) supplies scope for later
inference units; select exact primary passages when authored.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

See the [Stage 03 publication record](../../docs/stage-03-validation.md) for
independent numerical, preservation, execution and rendering evidence.


[Return to the learning path](../../README.md)
