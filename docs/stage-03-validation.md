# Stage 03 publication record

Current coverage: clustering and Gaussian mixtures/EM, with 33 exercises.
Stage 03 is partial; subsequent units and its capstone remain planned.

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
