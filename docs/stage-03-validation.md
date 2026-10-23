# Stage 03 publication record

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
