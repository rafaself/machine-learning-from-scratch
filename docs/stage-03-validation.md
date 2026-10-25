# Stage 03 publication record

Current coverage: clustering, Gaussian mixtures/EM, PCA/SVD, Bayesian/
variational inference and density/anomaly detection, with 82 exercises.
Stage 03 is partial; its representation-study capstone remains planned.

## Clustering — 2026-10-23

Published one chapter with 17 exercises, all six exercise categories, 78 cells,
two demonstration cells and eleven displayed Python reference blocks. Every
answer is blank and immediately followed by separate closed hint/solution
cells. Stage 03 is partial; mixture/EM, PCA/SVD, Bayesian/variational inference,
density/anomaly detection and its representation-study capstone remain planned.
At this publication, the repository has 22 published chapter/project notebooks
and 337 exercises. The 21 previously published notebook byte hashes are
unchanged, including learner answers, cell/exercise IDs, code and outputs.

### Mathematical and numerical review

The chapter derives the sum-SSE centroid gradient and Hessian, mean
decomposition, ordered-pair identity with divisor `2m`, and Ward's size-weighted
SSE increment. It separates a nonincreasing objective from stronger convergence
or optimality claims, and documents deterministic ties, empty clusters and
budget exhaustion. Ward heights store the actual SSE increment, explicitly
distinguishing transformed distance conventions.

The 62 independent numerical cases check scalar coordinate distances,
large-offset cancellation avoidance, block invariance, shapes, nonfinite and
overflow inputs, numerical seeding underflow, row-weighted RNG replay with
duplicates, distinct-center limits, centroid finite-difference gradients and
Hessians, pairwise SSE factors, Lloyd hand states, short-budget output
consistency, both empty policies, complete restart records and first-minimum
ties. Hierarchy checks independently recompute every member-level merge cost,
all-pair averages, unequal-size Ward weights, nested cuts, child/count validity
and Ward's telescoping SSE. Assessment checks cover silhouette hand values,
singletons/zero distances, label renaming and ARI against exact rational pair
counts for every pair of four-row binary partitions, including negative and
trivial results.

The tiny two-group optimum is independently enumerated with rational means.
The K-means++ seed expectation is independently summed with rational row
weights. For `(0,0,2,10)`, the optimum is `8/3`, expected initialization SSE
`10234/1287`, and a possible initialization costs `64`, exceeding the roughly
`57.453807` expectation bound. Lloyd repairs that tiny bad seed; the example
does not claim its initial cost persists after optimization or prove the
general approximation theorem.

The primary paper's indexed introduction and Theorem 1.1 were checked at the
authors' Stanford source. A direct full-PDF request failed at TLS transport;
the full proof was not audited. The author-hosted biology textbook's §§5.6–5.7
were inspected for linkage and validation. Chapter prompts/examples are
original. Source statements and the local finite-instance audit have separate
scopes.

### Execution and measured experiments

The full local suite passes **621 checks**: 184 structure, 412 numerical/project
checks, three offline-data checks and 22 fresh-kernel demonstration checks.
`uv sync --locked` succeeds with unchanged dependencies and lockfile.
CI includes the new clustering checks in its numerical step. Default CI loads
displayed function definitions for independent checks and separately executes
demonstrations, without running or grading learner answers.

A separate authoring audit executes all displayed references and demonstrations
in order in a fresh kernel, excluding answers. The chapter's stage guide
publishes the exact presets and approximate resource accounting: demonstrations
1.646 seconds / 180.6 MiB sampled aggregate RSS; full references 2.766 seconds /
198.3 MiB. These are measured execution costs, not estimates of study time,
exact physical-memory peaks or browser memory.

Both eight-start initialization searches reach selected SSE `104.8136034`,
probe mean distortion `0.56843639` and silhouette `0.73993862` on this fixed
blob example. Their selected traces differ; this is not evidence that the
methods are universally equivalent. The 48-row linkage experiment displays
eight partitions under two declared geometries without using generating
labels for selection. The single-Gaussian null and fixed-probe bootstrap
experiment produces ARIs approximately `0.6341, 0.7299, 0.6143, 0.4557`.
These four conditional sensitivity values are not a confidence interval or
evidence of semantic classes. Every measured fit stabilizes within its fixed
budget; deliberate unfinished cases are covered independently by tests.

### Rendering and preservation

Browser HTML preview contains all nine sections, 34 initially closed
disclosures, one default three-panel figure and no MathJax errors. Visual
inspection covers the Lloyd move, centroid loss curve and Ward height axis,
as well as all three separately executed experimental figures. Representative
JupyterLab review verifies rendered equations, blank written/code answers,
adjacent separate disclosures, and keyboard opening/closing of written and
displayed-code solutions. References remain Markdown; opening them does not
execute their code. Reviews operate on scratch copies and preserve canonical
notebook bytes.

Existing notebook bytes, dependencies and `main` are preserved. This unit adds
no framework, external dataset or reusable model implementation to the package.

## Gaussian mixtures and EM — 2026-10-23

Published one chapter with 16 exercises, all six categories, 74 cells, two
demonstration cells and nine displayed Python reference blocks. Answers remain
blank with separate adjacent closed hints/solutions. At this publication the
repository contains 23 published chapter/project notebooks and 353 exercises.
All 22 preceding notebook byte hashes, including clustering, are preserved.
PCA/SVD, Bayesian/variational inference, density/anomaly detection and the
representation-study capstone remain planned; the stage remains partial.

### Mathematical and numerical review

The text distinguishes densities, latent posteriors and point-valued parameters.
It derives the average likelihood, KL-gap identity, exact/generalized EM ascent
conditions, weighted ML statistics, fixed covariance-floor update and observed
NLL derivatives for means, softmax logits and covariance matrix differentials.
The support qualification for zero auxiliary probabilities is explicit.
The fixed spectral constraint clips covariance eigenvalues, rather than adding
jitter or changing the objective mid-run. Monotone likelihood, finite objective
limits, parameter convergence, global recovery and semantic interpretation are
kept as separate claims. The hard-assignment limit holds only under declared
equal-weight/common-isotropic-variance restrictions.

The 54 new independent numerical cases cover one/two-dimensional normalizing
constants and scalar quadratic forms, extreme log tails, shape/finiteness/SPD
contracts, strict mixture weights, independent density integration and posterior
normalization. Scalar entropy/KL calculations verify the bound and zero-support
cases without reconstructing log posteriors from underflowed probabilities.
Weighted scalar moments check the ML denominator and row-duplication invariance;
scalar constrained searches and rotated singular scatter check the covariance
update and distinguish eigenvalue clipping from added jitter.

Central differences of an independently evaluated scalar mixture likelihood
verify every mean/logit/covariance parameter at one and four observations,
including both entries in shared symmetric off-diagonal directions. Fitting
checks cover the hand update, fixed-floor ascent chain, short-budget and loose
tolerance semantics, consistent returned posteriors, copied initial state,
single-component statistics, label permutations and initial feasibility.
Instrumented starts retain deliberate numerical failures and budget-limited
fits; all-failed outcomes return their records while malformed inputs still
raise contract errors. A one-dimensional hard-limit check preserves exact ties.

The selected primary passages in the official Stanford Engineering Everywhere
mixture notes (pp.2–3) and EM notes (§2, equations (3)–(6)) were inspected.
The example `(−3,3)` with uniform auxiliary rows verifies a narrow source audit:
the bound rises from about `−9.91893853` to `−2.51755082`, while observed average
log likelihood falls from about `−1.61208570` to `−2.51755082`. The missing
old-model tightness equality explains the failure of the modified ascent claim.
This does not contradict exact EM, prove a general convergence theorem or
reproduce another course's assignments.

### Execution, resources and measured results

The full local suite passes **684 checks**: 192 structure, 466 numerical/project,
three offline-data and 23 isolated demonstration checks. The locked environment
sync succeeds with unchanged dependencies. CI adds the mixture cases to the
separate numerical step, without executing or grading learner answers.

A separate authoring audit runs every displayed reference and demonstration in
chapter order in a fresh kernel. The measured final layout requires 1.269
seconds / 180.2 MiB sampled aggregate RSS for demonstrations and 1.661 seconds /
190.6 MiB for the full references. The stage guide records exact settings and
the approximate driver/kernel-descendant accounting, excluding browser memory.
These are execution measurements, not study-time estimates or exact physical
memory peaks.

On the predeclared correlated-mixture sample, selected full/diagonal models
reach training average log densities `−2.42528958 / −2.79998553` and validation
scores `−2.51491963 / −2.80177083`. All eight starts stop by their relative-gain
criterion after 22–35 updates; no global-optimum or universal ranking follows.
The four tiny floor fits stop after three updates each. Their training scores
rise as floors shrink, while the fixed stress-probe score becomes very poor.
That probe is explicitly a stress set, not an IID test sample or final model
selection result. Every fit and status is reported without fabricated outcomes.

### Rendering and preservation

HTML review verifies all nine sections, 32 initially closed disclosures, one
default three-panel plot and no MathJax errors. All three actual figures were
visually inspected. An overlapping colorbar was corrected with constrained
layout, re-executed and reviewed before publication. Representative JupyterLab
review checks equations, blank written/code answers, adjacent independent
disclosures and opening/closing of displayed-code references without execution.

Review copies and reference-run outputs stay outside the canonical repository.
No earlier notebook, production dependency or package model implementation was
changed. The branch remains `curriculum-notebooks`; `main` is not merged.

## PCA and SVD — 2026-10-23

Published one chapter with 16 exercises, all six categories, 75 cells, two
demonstration cells and eight displayed Python reference blocks. Every answer
is blank with separate adjacent closed hints/solutions. At this publication
the repository contains 24 published chapter/project notebooks and 369
exercises. All 23 earlier notebook byte hashes are preserved. The stage remains
partial: Bayesian/variational inference, density/anomaly detection and the
representation-study capstone are still planned.

### Mathematical and numerical review

The chapter derives the Rayleigh and trace arguments for orthogonal PCA,
discarded-eigenvalue distortion, thin SVD/covariance spectrum relation and an
independently differentiable half-reconstruction objective. It states the
population covariance divisor, feature-weighted geometry, frozen training
preprocessing, score/reconstruction shapes, whitening inverse and numerical
rank tolerance. Zero-component, constant, wide and rank-deficient inputs have
explicit conventions. Sign changes, tied eigenspaces and boundary-nonunique
projectors are separated from uniquely determined subspaces.

The 43 independent numerical cases check the hand rectangle, scalar covariance
statistics and an independent covariance eigensolver for tall/wide data, sign
conventions and numerical rank. Other cases cover covariance divisors,
whitened training covariance and inverse reconstruction, shifted probes with
frozen preprocessing, zero components, empty queries, constant features,
numerically null whitening, tiny/huge scale normalization and visible raw-range
failures. Shape, work-limit and boolean-option contracts are checked directly.

Central differences of a nested scalar reconstruction objective validate every
trial-basis gradient at one/four rows and zero/one/two columns, without assuming
orthonormality during perturbation. Row duplication checks the observation
average. A zero stationary trial matrix is explicitly worse than the leading
PCA basis. Independent Rayleigh grids, tied-projector examples, affine/raw-SVD
contrasts, an outlier calculation, variance/target counting and squared
condition numbers check separate mathematical claims. A scaled-metric test
uses a separated eigenspectrum rather than requiring an arbitrary tied basis.

Selected passages in the official CS229 PCA notes (p.2, pp.5–6) and NumPy's
SVD interface/notes were inspected. The original four-row target audit retains
`100/101` of variance while its exact one-dimensional score has best table
accuracy `1/2`; the original second coordinate determines the target exactly.
The analytic counting argument is separated from numerical roundoff in the
fitted projector. It limits a predictive-information interpretation, without
contradicting the variance/reconstruction optimum or claiming a four-row
generalization estimate.

### Execution, resources and measured results

The full local suite passes **736 checks**: 200 structure, 509 numerical/project,
three offline-data and 24 fresh-kernel demonstration checks. Locked sync
succeeds with unchanged dependencies. CI runs the new PCA cases in its
numerical group; its folded command retains all preceding cases. Learner
answers are neither executed nor graded.

The separate authoring audit executes every displayed Python reference and
demonstration in chapter order without answers. Measured demonstrations take
1.881 seconds / 180.0 MiB approximate sampled aggregate RSS; full references
take 2.328 seconds / 197.3 MiB. The stage guide records settings and approximate
driver/kernel-descendant accounting, with shared pages counted per process
and browser memory excluded. No GPU is used.

On the fixed six-feature example, two components retain about `0.99244199` of
training variance, with training/validation distortion `0.08244960 / 0.09602235`.
All six directions reconstruct both samples within numerical roundoff; this
is not a latent-dimension discovery criterion. Standardized and whitened
reconstructions agree to a maximum difference about `8.88e-16`, while score
distances and covariances change as specified.

For ideal weak singular value `1e-10`, float64 direct SVD has relative weak
variance error about `1.31e-6`; formed covariance has a negative computed weak
eigenvalue around `−1.39e-17`, which is reported. Rounding that dense input to
float32 removes the weak direction; both its native SVD and a float64 SVD of
the already rounded matrix return zero. These are measured finite cases,
not a promise of universal SVD accuracy or recovery of lost input information.

### Rendering and preservation

HTML review verifies all nine sections, 32 initially closed disclosures, actual
rendered equations, one default three-panel figure and no MathJax errors.
Visual inspection covers the projection residuals, spectrum and tail-error
plot, as well as all three additional experimental figures. JupyterLab review
checks written/code answer placement, displayed equations and keyboard opening/
closing of separate reference disclosures. Review copies stay outside the
canonical curriculum and references remain unexecuted Markdown in Run All.

All 23 earlier notebooks, dependencies and package model boundaries remain
unchanged. Publication stays on `curriculum-notebooks`, without merging `main`.

## Bayesian and variational inference — 2026-10-23

Published one chapter with 16 exercises, all six categories, 74 cells, two
demonstration cells and nine displayed Python reference blocks. Answers stay
blank with separate adjacent closed hints/solutions. At this publication the
repository contains 25 published chapter/project notebooks and 385 exercises.
All 24 preceding notebook byte hashes are preserved. Stage 03 remains partial:
density/anomaly detection and the representation-study capstone remain planned.

### Mathematical and numerical review

The chapter derives Beta–Bernoulli conjugacy, ordered-sequence versus count
evidence, beta-binomial prediction and shared-parameter future dependence.
It separates parameter variance, single-outcome variance and batch variance.
Gaussian square completion supplies an exact posterior/predictive example and
the L2 coefficient under an explicitly averaged half-squared-loss objective.
The MAP/Jacobian example states the coordinate dependence of density modes.

The variational section defines continuous KL/support conditions, the ELBO gap,
Gaussian KL, the unique optimum in a diagonal Gaussian family and sequential
coordinate updates. Reparameterization gradients are derived manually at fixed
noise, with one sample average. Exact Gaussian expectation, sampled KL,
estimated IID error, optimizer stopping and family restriction are separate
objects. The information/decision chapter is an explicit conceptual prerequisite;
no live functions are imported from an earlier notebook or a package model.

The 74 new independent numerical cases use rational factorial integrals for
Beta evidence/predictive tables and moments, polynomial density quadrature,
endpoint limits, sequential/batch equivalence and count/range checks. Separate
tests verify future covariance, one-step variance, coordinate MAP and the exact
prior-predictive total-variance identity. Scalar Gaussian square completion
checks posterior stationarity, prediction, sample-size/L2 scaling and visible
normalization-range failures.

Gaussian KL is compared with an independent two-dimensional determinant/
quadratic formula, including divergence direction. Precision-conditional
variances, sequential coordinate snapshots, every trace entry, copied inputs,
budget/tolerance semantics and nonzero family error are checked independently.
Central differences of separate scalar objectives verify exact and fixed-noise
mean/log-scale gradients. A deterministic moment-exact probe checks the quadratic
expectation without interpreting it as IID Monte Carlo. Negative sampled KL
is retained, and dimension/sample/variance-range cases are explicit. The
two-dimensional probability-contour radius is checked against its radial CDF.

Primary readings inspected on 2026-10-23 were Stanford CS228's Setup/Conjugate
Priors/Beta Distribution passages and Blei, Kucukelbir and McAuliffe, v9
(2018-05-09), §§2.2–2.4, equations (13), (15), (18). Original finite examples
audit the selected statements; coursework answers and the paper's full mixture
experiments are not reproduced.

The Beta(10,1) counterexample increases realized variance after a failure from
about `0.00688705` to `0.01068376`. Prior-predictive expected posterior variance
is about `0.00631313`, and the variance of posterior means supplies the remaining
`0.000573921`. Rational checks establish the exact total-variance identity.
The example rejects an overbroad realized-decrease reading without contradicting
expected variance reduction under the prior joint model.

### Execution, resources and measured results

The full local suite passes **819 checks**: 208 structure, 583 numerical/project,
three offline-data and 25 isolated demonstration checks. Locked sync succeeds
with unchanged dependencies. CI includes the new inference file in its
separate numerical group; it never executes or grades unfinished learner code.

A fresh-kernel authoring audit executes every displayed reference and
demonstration in order without answers. Final measured demonstrations require
1.276 seconds / 181.3 MiB approximate sampled aggregate RSS; full references
require 1.911 seconds / 197.3 MiB. The stage guide records exact presets and the
20 ms driver/kernel-descendant sampling convention, shared pages counted per
process and browser memory excluded. No GPU is used.

The six prior/count combinations report actual posterior moments and finite
4,000-draw interval estimates. A 12,000-batch predictive simulation gives mean
about `2.49608` and variance `1.25982`, versus analytic `2.5 / 1.25` and plug-in
variance `0.9375`. This is a specified-model Monte Carlo check, not calibration
evidence for an arbitrary real-data prior/likelihood.

The correlated-Gaussian coordinate run stops by its gain tolerance after 25
sweeps: KL is about `0.5108256243`, only `4.87e-10` above the best diagonal
value, with mean-gradient norm about `3.12e-5`. Diagonal variances remain
`0.36` versus true marginal variances one. The approximation's 95%-mass region
has estimated target mass `0.69967` (Monte Carlo standard error about `0.00418`),
while estimated own-model and target-own-region masses are `0.95108 / 0.95042`.
These probabilities name their distributions; they are not repeated-dataset
coverage claims. The 24 paired gradient estimates retain all eight child-seed
trials at three sample sizes and distinguish replicate ranges from intervals.

### Rendering and preservation

HTML review verifies all nine sections, 32 initially closed disclosures,
rendered equations and one default three-panel figure without MathJax errors.
All four actual figures were visually inspected. The coordinate-KL axis and
sample-size ticks were clarified, then re-executed and reviewed. Representative
JupyterLab review checks written/code answer placement and keyboard opening/
closing of separate displayed-code references without executing them. Review
copies stay outside the canonical curriculum.

All 24 preceding notebooks, their IDs/answers/outputs, dependencies and package
boundaries remain unchanged. Publication stays on `curriculum-notebooks`;
`main` is preserved.


## Density estimation and anomaly detection — 2026-10-25

Published one chapter with 17 exercises, all six categories, 79 cells, two
demonstration cells and nine displayed Python reference blocks. Every learner
answer stays blank, followed by separate closed hint and solution cells.
The repository now contains 26 published chapter/project notebooks and 402
exercises. All 25 preceding notebook byte hashes are unchanged. Stage 03
remains partial until its representation-study capstone is developed.

### Mathematical and numerical review

The chapter derives the normalized multivariate Gaussian KDE, the
change-of-variables density factor, stable log-domain scoring and the manual
derivative of mean validation NLL with respect to log bandwidth. A separate
Gaussian baseline uses ML covariance divisor `n` and a fixed eigenvalue floor.
Training-only invertible centering/scaling, bounded featurewise query blocks,
first-minimum grid selection and original-coordinate likelihood units remain
explicit. Calibration/final rows never enter fitting or bandwidth selection.

The anomaly section derives an inclusive extreme-rank p-value and its equivalent
strict order-statistic threshold, including ties, rank resolution and an
infinite threshold when no rank is rejectable. Rejection uses integer arithmetic
at a declared decimal-rational alpha, rather than rounded p-values or an
interpolating quantile API. The proof states exchangeability conditional on an
independently frozen score, with probability marginal over calibration and a
new nominal row. It does not claim conditional-on-calibration, subgroup,
stream-wise, power, semantic correctness or drift guarantees.

The 88 new numerical cases independently check scalar Gaussian mixtures,
one-/two-dimensional density quadrature and mixture moments, scalar-objective
finite differences across dimensions and block sizes, loss-average scaling,
Gaussian covariance/floor algebra, training-only preprocessing/selection,
copied inputs, density Jacobians and decision invariance. Exact rational
comparisons exhaust every held-out index in distinct and tied small multisets.
They verify conservative rank rejection, the hand threshold, decimal boundaries,
monotone score transformations and empty queries. Further cases expose raw
underflow versus valid log scores, arithmetic-range failure, training
self-likelihood divergence, indexed LOO duplicate limitations, invalid shapes,
row/grid/block budgets, noninvertible projection and Wilson endpoint behavior.
Controlled experiment records are checked against independent scalar rank
counts, and paired calibration repetitions retain every trial.

Primary readings inspected on 2026-10-25 are Shalizi's chapter 14 (2024-03-12
version), section 14.4 equation 14.19 and section 14.4.2, and Angelopoulos/Bates
arXiv:2107.07511v6 (2022-12-07 metadata), section 4.4 proposition 3,
section 3.2 and Appendix D equation 84. The chapter gives exact links,
prerequisites, a bounded reproduction and limitations. Its source/code audit
shows that `quantile(scores, k/n, method='higher')` can overshoot the intended
one-based order statistic: for scores 1..9 and alpha .35 it returns eight
rather than seven. This is more conservative and can lose power; it does not
invalidate the paper's mathematical rank bound. The independent implementation
uses the stated order statistic directly.

### Execution, resources and measured results

The full local suite passes **916 checks** in 52.90 seconds: 216 structure,
671 numerical/project, three offline-data and 26 isolated demonstration checks.
Locked environment sync succeeds without dependency/package changes. The CI
numerical group includes the density file separately from structural and
fresh-kernel execution validation. Default CI does not grade unfinished answers
or run displayed reference solutions.

A separate fresh-kernel author audit runs all nine displayed Python blocks and
demonstrations in order without learner answers. Demonstrations take 1.421
seconds / 184.9 MiB approximate sampled aggregate RSS; the full reference phase
takes 1.954 seconds / 193.9 MiB. The measurements include kernel startup,
plots and shutdown, with one BLAS/OpenMP thread and no GPU. Memory is the
20 ms sampled sum of driver/kernel-descendant RSS, shared pages counted per
process and browser excluded, not an exact physical-memory peak. Presets and
bounded work sizes are documented in the stage guide.

The hand KDE densities are `0.2419707245 / 0.1079819330` for bandwidths one
and one half, with manual NLL log-bandwidth derivatives `0 / -3`. Rank scores
1..9 at alpha .35 produce threshold seven, p-values `.4, .3, .2, .1` for
queries `7, 7.1, 9, 10`, and strict rejection of the last three only.

At the declared seed, clean Gaussian/KDE nominal flags are `30 / 29` out of
400, gap detections `4 / 400` out of 400, and shifted-nominal flags `243 / 222`
out of 400. The training-contaminated condition reports nominal `37 / 23`,
gap `0 / 400`, and shifted `30 / 224`. Replacing 40 of 199 calibration rows
with distant points produces zero flags in all three evaluation populations
for both methods; all those outcomes are retained. These are measured
fixture-specific results, not universal contamination directions or guaranteed
power. Shifted inputs are declared operationally nominal but violate the
original statistical exchangeability assumption.

Across 128 independent uniform-score calibrations, mean true conditional
false-alarm rates are about `.09050 / .09990` for calibration sizes 19 / 99,
with across-repeat standard deviations `.05998 / .03025`. The exact marginal
rate is .1 for both sizes. The experiment evaluates each realized threshold's
conditional rate analytically; it does not confuse calibration variability
with finite probe noise. Positive unit conversion shifts both models' scores
by `log(10)` while retaining all decisions.

### Rendering and preservation

HTML inspection confirms nine sections, 34 initially closed disclosures,
rendered equations and a default two-panel plot without MathJax errors.
The default and both additional reference figures were visually inspected;
axis labels and candidate ticks were clarified before the final author audit.
Representative JupyterLab review confirms blank code answers and separate
reference cells, keyboard opening/closing of displayed code and rendered
mathematics without errors. Review copies remain outside the repository.

All 25 preceding notebooks, including IDs, answers, executable sources and
retained outputs, remain byte-for-byte unchanged. No dependency or reusable
package was added. Publication stays on `curriculum-notebooks`; `main` is
preserved.
