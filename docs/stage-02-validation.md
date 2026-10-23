# Stage 02 publication record

Current coverage: seven published chapters with 110 exercises. Stage 02 remains
partial; its integrated capstone is planned.

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

## Tree ensembles — 2026-10-23

Added one chapter with 15 exercises, two demonstration cells and seven displayed
Python reference blocks. The full local suite passes 399 checks. The 31 new
numerical cases check small base learners and global class columns, instrument
per-node feature draws, independently replay bootstrap streams, reconstruct
every OOB contributor, verify missing coverage and missing-class bags, compare
hard/soft voting, independently finite-difference both boosting objectives,
check score normalization and extreme logits, and verify one-step and multistep
boosting states against scalar residual means and exhaustive stump costs.
Analytical variance/coverage and exponential-loss reweighting examples also
receive independent checks. The original checks remain present.

Demonstrations and all displayed references execute in separate fresh kernels.
Paired forest experiments use identical bags across feature settings and report
actual OOB coverage. Bounded boosting experiments print measured selection
records and plot actual training/validation traces without invented conclusions.
The stage guide records resources and workload; no dependencies or GPU work
were added. All 13 starting notebooks remain byte-identical.

The original Breiman source sections were checked for OOB contributor definitions
and node-level feature sampling. The reading task explicitly distinguishes hard
votes, soft probabilities, finite coverage, tuning reuse and broader statistical
claims. Browser/JupyterLab inspection checks equations, plotted analytical
curves, blank answers and separately collapsed references.

## SVMs and kernels — 2026-10-23

Added one chapter with 16 exercises, two demonstration cells and eight displayed
Python reference blocks. The full local suite passes 450 checks. The 42 new
numerical cases independently check hinge subgradients away from kinks and a
valid selected subgradient at a kink, loss-normalization/replication behavior,
explicit polynomial features and scalar RBF distances, invalid kernel examples,
box/equality projection against exact small solutions, primal/dual optima and
KKT conditions, budget exhaustion, bias recovery without a separating feature,
an exact polynomial XOR solution, query blocks and input preservation.

Fresh isolated kernels execute demonstrations and all displayed references.
The fixed-budget ring experiment includes 12 runs, of which six meet the relative
gap target and six remain unfinished at 2000 steps. These statuses and actual
gaps are retained and summarized in the stage guide. No run was dropped or
silently granted extra work. Exact small optimization cases converge and match
independent primal/dual calculations. Dense Gram limits and measured resources
are explicit; no dependencies or GPU work were added.

The selected CS229 kernel section was checked for feature-map and validity
arguments. The mandatory reading does not claim reproduction of its later SMO
solver. Browser/JupyterLab review verifies equations, both plots, blank answers
and initially closed separate references. All 13 starting notebooks remain
byte-identical, including their IDs, executable content and retained outputs.

## Statistical learning theory — 2026-10-23

Added one chapter with 14 exercises, two demonstration cells and six displayed
Python reference blocks. The full local suite passes 499 checks. The 40 new
numerical cases independently check concentration radii, sample-count ceilings,
monotonicity/vacuity and large family counts; exact rational binomial examples
and deterministic endpoints; threshold/interval pattern counts, duplicates and
shattering counterexamples; fixed-table errors/ties; analytical population risk
against an independent definition-based integration; and the ERM comparison on
small uniformly controlled arrays.

Demonstrations and every displayed reference execute in separate fresh kernels.
The null-selection and known-risk experiments keep their families fixed before
evaluated observations, print actual repeated-sampling records and distinguish
Monte Carlo evidence from distribution-free proofs. The stage guide records
workload/time/memory; no dependencies, GPU work or large datasets were added.
All 13 notebooks present at the start of this goal remain byte-identical.

The selected finite-family passage was reviewed through indexed text from the
primary CS229 notes. Direct PDF retrieval timed out or failed TLS in this
environment; the full document was not downloaded. The chapter's derivation and
numerical examples were independently reviewed and checked, without claiming
reproduction of the full VC proof. Browser/JupyterLab review verifies equations,
both plots, blank answers and initially closed separate references.
