# Stage 02 publication record

Current coverage: four published chapters with 65 exercises. Stage 02 remains
partial; its three other chapters and integrated capstone are planned.

## Probability, information, and decisions — 2026-10-23

At this unit's publication: one chapter, 17 exercises, two demonstration cells, and
eight displayed Python reference blocks. Stage 02 is partial; the six algorithm/
theory units and integrated capstone are still planned. The existing 13 notebooks
were compared against their byte hashes before this expansion and remain
unchanged, including IDs, learner answers, executable cells, and outputs.

The full local suite passes 253 checks. The 20 new numerical cases independently
check information identities/support, mutual information with zero/tiny categories,
categorical logit and affine parameter gradients, observation/penalty scaling,
curvature/nullspace, cost-sensitive actions/ties, normal conjugate prediction,
MAP-to-ridge scaling, and the demonstration's exact formulas.

All demonstrations execute in fresh kernels with learner answers and displayed
solutions excluded. A separate authoring run executes every displayed reference
in notebook order with its demonstrations, in an isolated kernel. It does not
modify student answers or expose reference functions during default Run All.
Only the new chapter received demonstration outputs; old notebooks were untouched.

The primary paper's bibliographic details and selected Section 6 were verified
against the original reprint. Browser previews and JupyterLab review check
equations, the plot, blank answer placement, and separate closed references.
The [stage guide](../notebooks/02-classical-ml/README.md) gives measured workload,
time, and approximate memory accounting. No dependencies or frameworks were added.

Reproduce using the locked environment:

~~~bash
uv sync --locked
uv run --locked --no-sync pytest
uv run --locked --no-sync pytest tests/test_curriculum.py
uv run --locked --no-sync pytest tests/test_numerics.py
uv run --locked --no-sync pytest tests/test_execution.py
~~~

These checks establish the stated tested contracts, not a proof for arbitrary
inputs, a grade for learner work, or completion of the remaining stage.

## Nearest neighbors — 2026-10-23

Added one chapter with 15 exercises, two demonstration cells, and seven
displayed Python reference blocks. The full local suite passes 279 checks.
The 17 new numerical cases compare block search with an independent scalar
distance/sort oracle across k and block sizes; check ties, exact duplicates,
large-offset distances, empty queries and invalid ranges; verify weighted
classification/regression; and independently replay class-stratified CV with
fold-local transforms and pooled validation weighting.

Demonstrations and every displayed reference execute in isolated authoring
kernels. The reference presets use generated local arrays, direct coordinate
distance accumulation, and bounded query/training blocks. Measured resource
accounting is in the stage guide; no dependency or framework was added.

The Cover–Hart paper's Section VI was checked against the authors' archived
original. Its binary limiting-risk interpretation is explicitly separated from
finite-sample measured accuracy. Browser/JupyterLab review verifies plots,
answer cells, and initially closed references. All 13 notebooks present at the
start of this goal remain byte-identical.

## Generative classifiers — 2026-10-23

Added one chapter with 17 exercises, two demonstration cells and nine displayed
Python reference blocks. The full local suite passes 316 checks. The 28 new
numerical cases independently check score normalization and extreme contrasts,
Gaussian densities against scalar two-dimensional formulas, fixed-covariance
mean gradients by finite differences, within-class MLE covariance divisors,
replicated-data invariance, singularity and explicit jitter, linear/quadratic
posterior odds, Bernoulli/multinomial likelihoods and smoothing denominators,
empty-document semantics, stable long-document scoring, frozen vocabularies and
unknown-token counts, and exact dependence/misspecification demonstrations.

All demonstrations and displayed references execute in separate fresh kernels.
The Gaussian experiment compares three covariance assumptions on identical
generated arrays for each regime and seed. It prints actual validation records
and times without supplying invented student conclusions. Resource measurements
and workload are in the stage guide; required experiments use no GPU or new
dependencies. All 13 notebooks present at the start of this goal are unchanged.

The selected CS229 source sections were checked for the shared-covariance
estimators and logistic-posterior connection. The bounded study distinguishes
an algebraic family identity from any claim about fitted finite-sample quality.
Browser and JupyterLab review found no rendered equation errors; both plots,
adjacent blank answers, and initially closed independent references were checked.

## Decision trees — 2026-10-23

Added one chapter with 16 exercises, two demonstration cells and seven displayed
Python reference blocks. The full local suite passes 359 checks. The 34 new
numerical cases independently check four impurities, all legal one-step
partitions with several leaf-size rules, unequal child weights, tie/repeated-value
behavior, extreme/adjacent-float thresholds, regression/classification traversal,
size and stopping limits, conflicting duplicates, the deliberate positive-gain
XOR limitation, and fixed-tree pruning against exhaustive subtree enumeration.
Exact tests distinguish global risk from local gain and training-order
invariance from midpoint-based predictions after a nonlinear transformation.

Demonstrations and every displayed reference run in fresh separate kernels.
Measured workloads include matched growth/pruning comparisons and four controlled
one-label perturbations, with timing and model-size records. Resource details
are in the stage guide; no dependencies or GPU work were introduced. All 13
starting notebooks remain byte-identical. The selected CS229 sections were
checked for weighted gain, finite partitions and regularization scope.

Structural validation caught an unescaped less-than expression inside a hint
that confused raw disclosure parsing. It was corrected with LaTeX notation and
all checks rerun. Browser and JupyterLab inspection verifies the equations,
both plots, adjacent blank answer cells, and initially closed references.
