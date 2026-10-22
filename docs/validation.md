# First-delivery publication record

Validated locally on 2026-10-22 on the Ryzen 7 6800H laptop. This record covers
the migration, complete roadmap, Stages 00–01, and retained Stage 04 chapters.
It does not certify planned stages as implemented or benchmarked.

## Inventory and preservation

- Nine documented stages, eleven published chapters, and two project starters.
- 192 exercises with dedicated answer cells and separate closed hints and
  worked references: 108 preserved exercises plus 84 newly authored ones.
- A comparison against the six-notebook starting snapshot at
  `f3b0ac861972755dae090a721040d7b95611b328` preserves every cell ID and its
  metadata, all answer/code sources, execution counts, and outputs. Only
  cross-directory Markdown destinations changed in those notebooks.
- New notebooks use explicit stage/chapter or project prefixes. Planned
  registry entries have no placeholder notebook files.

## Automated checks

The original 95-check suite is retained and expanded through registry-based
discovery. The final local run passed **223 checks**. The structural checks cover schemas, nine sections, numbering,
types, adjacency, closed disclosures, displayed-code compilation, links,
readiness, prerequisites, and project documentation. Completed student answer
content is allowed by the tests.

Numerical checks load only displayed reference function definitions. They use
independent scalar objectives and central differences, hand values, alternate
linear algebra calculations, and explicit edge cases. Coverage includes
affine VJP/JVP/HVP, covariance denominators, Bayes normalization, Bernoulli and
Gaussian likelihoods, stable log-domain reductions, streaming moments, both
L2 objectives, unpenalized intercepts, row-duplication invariance, ridge
stationarity, metric boundaries/ties, ROC pairwise equivalence, fold coverage,
training-fold preprocessing, and independently replayed fixed-work CV.

Execution checks run demonstrations from each published chapter/project in
a fresh kernel and temporary working directory. They omit learner answers
and displayed solutions, require a plot, check that reference functions were
not defined, and leave canonical notebook bytes unchanged. This is separate
from structural and numerical validation; it does not grade student work.

Reproduce using the unchanged locked environment:

~~~bash
uv sync --locked
uv run --locked --no-sync pytest
uv run --locked --no-sync pytest tests/test_curriculum.py
uv run --locked --no-sync pytest tests/test_numerics.py
uv run --locked --no-sync pytest tests/test_execution.py
~~~

The existing minimal GitHub Actions workflow runs those three validation
groups separately. No framework or production dependency was added.

## Reference and resource audit

In a separate local authoring audit, every displayed Python reference was
executed in notebook order together with its demonstrations, in a fresh kernel
per notebook, omitting answer cells. Thus reference experiments and their
snippet dependencies were checked without exposing functions to default Run
All. References remain fenced Markdown; their executed copies and project
checkpoint files were kept outside the reference repository.

Only new notebooks received newly executed demonstration outputs. Existing
notebook outputs were retained exactly. Resource tables and workload settings
are published in the [Stage 00](../notebooks/00-mathematical-tools/README.md)
and [Stage 01](../notebooks/01-ml-fundamentals/README.md) guides. Measurements
include fresh-kernel overhead and approximate sampled aggregate process RSS;
they exclude browser memory and do not claim exact peak physical memory.
Required workloads use CPU only and small deterministic synthetic arrays.

## Rendering and pedagogical review

New notebooks were exported with nbconvert and inspected in browser previews
for headings, equations, plots, answer placement, and initially closed
references. Representative notebooks were also inspected in JupyterLab,
including opening worked references separately from answer cells. Mathematical
rendering errors and missing images were checked. The curriculum retains
worked examples, six exercise categories, and independent demonstrations.

Projects distinguish explaining assumptions/derivations, implementing,
verifying numerically, and evaluating a bounded experimental claim. Report
templates contain prompts and empty evidence fields, not invented findings.

Passing these checks supports the stated contracts and tested calculations.
It is not a proof of every derivation, a grade for a learner's implementation,
or evidence that tiny synthetic experiments generalize to arbitrary datasets.
