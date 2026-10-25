# Stage 03 — Unsupervised and Probabilistic Learning

**Availability:** partial. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Discover and model structure without confusing a visually persuasive representation with validated evidence.

**Prerequisites:** Stage 00 probability/matrix tools and Stage 01 experimental
design. The Bayesian/variational unit additionally uses the
[Stage 02 information/decision chapter](../02-classical-ml/probability-information-and-decisions.ipynb)
for entropy and KL. The other Stage 02 algorithms are useful comparison
experience rather than prerequisites for the clustering, mixture or PCA units.

## Ordered units

1. Clustering: K-means, initialization, hierarchical linkage, distance choices, and cluster assessment.
2. Gaussian mixtures and EM: latent assignments, likelihood, responsibilities, lower-bound derivation, monotonicity assumptions, and singularities.
3. PCA and SVD: centering, orthogonal projections, variance and reconstruction objectives, rank, spectrum, conditioning, and scaling.
4. Bayesian and variational inference: conjugate posterior/predictive examples, point versus distributional estimates, latent variables, Jensen/KL bounds, and approximation limits.
5. Density estimation and anomaly detection: normalized Gaussian KDE, independent bandwidth selection, manual derivatives, density units/typicality, exact split-conformal ranks, contamination, and distribution shift.

<!-- availability:start -->

- [clustering](clustering.ipynb): **ready**
- [gaussian mixtures and em](gaussian-mixtures-and-em.ipynb): **ready**
- [pca and svd](pca-and-svd.ipynb): **ready**
- [bayesian and variational inference](bayesian-and-variational-inference.ipynb): **ready**
- [density estimation and anomaly detection](density-estimation-and-anomaly-detection.ipynb): **ready**
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
objectives remain visible. The Bayesian/variational chapter implements exact
conjugate prediction, Gaussian KL, sequential mean-field updates and manual
reparameterization gradients. Density/anomaly detection adds blocked log-domain
Gaussian KDE, a fixed-floor Gaussian baseline, validation-only bandwidth
selection, manual log-bandwidth gradients and independently calibrated rank
decisions. Score semantics, finite-sample marginal validity, detection power
and shift robustness are separate claims.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain objective differences between clustering, density modeling and dimensionality reduction, distinguish parameter posteriors from predictive distributions, and state the probability space and assumptions of a false-alarm guarantee.
- Implement the essential updates, transformations and local inference calculations, including training-fitted centering, manual variational/bandwidth gradients and exact calibrated ranks.
- Verify EM, orthogonality, reconstruction, conjugate probabilities, KL, fixed-noise derivatives, density normalization/Jacobians and finite tied-rank decisions against independent small calculations.
- Evaluate multiple starts, representation/prior choices, approximation error, contamination and distribution shift with declared information boundaries and uncertainty; use external labels only when justified.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Representation study: compare clustering and reduced representations on the same unlabeled data, report sensitivity to scale/initialization, and identify where the chosen evaluation could mislead.

Submit learner work on a study branch. The planned capstone reference will supply instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

CPU only. Use small dense matrices and bounded pairwise calculations. Large-corpus and distributed clustering are outside the required experiments.

Measured on the Ryzen 7 6800H laptop with Python 3.12.15, NumPy 2.5.4 and
Matplotlib 3.11.2, with BLAS/OpenMP threads set to one:

| Published unit | Measured date | Demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- | --- |
| Clustering | 2026-10-23 | 1.646 / 180.6 | 2.766 / 198.3 |
| Gaussian mixtures and EM | 2026-10-23 | 1.269 / 180.2 | 1.661 / 190.6 |
| PCA and SVD | 2026-10-23 | 1.881 / 180.0 | 2.328 / 197.3 |
| Bayesian and variational inference | 2026-10-23 | 1.276 / 181.3 | 1.911 / 197.3 |
| Density estimation and anomaly detection | 2026-10-25 | 1.421 / 184.9 | 1.954 / 193.9 |

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

Inference sensitivity compares three priors at two declared count samples,
with 4,000 posterior draws per case. A separate predictive check uses 12,000
four-observation batches, drawing one shared parameter per batch. The Gaussian
mean-field case uses two parameters, a 100-sweep budget and gain tolerance
1e-9; it stops after 25 sweeps, retaining its nonzero family KL and mean-gradient
diagnostic. Distribution-mass comparisons use 12,000 draws each from the target
and approximation. Gradient experiments use eight child seeds and three paired
sample sizes, at most 2,048 noise rows in two dimensions. Generic helpers cap
observed counts at 10,000, future count tables at 200, Gaussian dimensions at
eight, coordinate sweeps at 200 and supplied noise rows at 20,000.

Density experiments use 240 training, 160 validation and 199 calibration rows
in two features. Each of two fitting conditions compares five bandwidths and
one Gaussian with fixed variance floor .02. Six method/condition records
evaluate the same 400 nominal, 400 gap-probe and 400 shifted nominal rows,
including separate training- and calibration-contamination interventions.
All candidates, thresholds, counts and conditional Wilson summaries are
retained. A separate known-uniform-score study retains 128 independent
calibration repeats at sizes 19 and 99, paired by prefixes within repeats.
The 64-dimensional radius illustration uses 4,000 draws; the unit-conversion
audit repeats the small two-feature comparison. Generic fitting is capped at
2,000 rows / 16 features, scoring/calibration at 20,000 rows, query blocks at
256 and bandwidth grids at twelve. Featurewise blocks avoid a three-axis
distance tensor, but total KDE work still scales with all train/query pairs.

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

The inference unit studies
[Stanford CS228 Bayesian Learning, Setup and Conjugate Priors/Beta Distribution](https://ermongroup.github.io/cs228-notes/learning/bayesian/)
and [Blei, Kucukelbir and McAuliffe, §§2.2–2.4, equations (13), (15), (18)](https://arxiv.org/html/1601.00670v9).
Reproduce conjugacy and the Gaussian coordinate update, then audit realized
versus expected variance and optimized versus accurate approximation. The
paper version is v9 (2018-05-09); both sources were accessed on 2026-10-23.
The bounded work does not reproduce the full mixture experiments or establish
general uncertainty/calibration guarantees.

[CS229](https://cs229.stanford.edu/index.html-spr26) supplies scope for later
inference units; select exact primary passages when authored.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

See the [Stage 03 publication record](../../docs/stage-03-validation.md) for
independent numerical, preservation, execution and rendering evidence.


[Return to the learning path](../../README.md)
