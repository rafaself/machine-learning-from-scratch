# Stage 02 publication record

## Probability, information, and decisions — 2026-10-23

Published coverage: one chapter, 17 exercises, two demonstration cells, and
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
