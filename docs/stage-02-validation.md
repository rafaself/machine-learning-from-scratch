# Stage 02 publication record

Current coverage: two published chapters with 32 exercises. Stage 02 remains
partial; its five other chapters and integrated capstone are planned.

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
