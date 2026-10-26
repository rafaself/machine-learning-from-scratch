# Stage 04 publication record

Current coverage: the three retained binary neural chapters and the new
multiclass-learning-objectives chapter, 74 exercises in four notebooks. Stage 04
remains partial: Momentum/Adam, optimization geometry, normalization/dropout,
the manually built differentiation engine and the integrated capstone are
planned. This record documents new units; retained-chapter checks and timings
are also recorded in the stage guide and original publication record.

## Multiclass learning objectives — 2026-10-25

Published 19 exercises covering all six exercise categories, 88 cells, two
demonstration cells and ten displayed Python reference blocks. Dedicated
learner answers stay blank, followed by separate adjacent closed hints and
solutions. The new prefix is `04.04`; original neural exercise/cell IDs and
notebook bytes remain unchanged. The repository now has 28 published
chapter/project notebooks and 438 exercises. All 27 preceding notebook byte
hashes are preserved. Dependencies and the reusable package are unchanged.

### Mathematical and numerical review

The chapter derives row-wise categorical probabilities, stable log softmax,
reverse Jacobian products, weighted mean cross-entropy gradients, target
entropy/KL and logit Hessian products. It distinguishes `1/T` and `1/T^2`,
the common-shift null direction, logit-space positive semidefiniteness and
nonconvex neural parameter objectives. The reverse products use linear
class storage without explicit batch Jacobian tensors.

Integer labels `(n,)`, declared class-column order and soft targets/probabilities
`(n,K)` are explicit; retained binary `(n,1)` conventions are preserved and the
editing agreement now names both. Positive row weights normalize once.
Inactive rows use zero target mass, may carry an ignored integer and still
require finite logits. All-masked means fail explicitly. Kernel derivatives
retain computed target mass to preserve the common-shift derivative under
admitted target-sum roundoff, including the one-class case. Log probabilities
remain available when exponentiated probabilities underflow; no clipping
changes the likelihood or gradient.

Manual linear and tanh-network gradients cover every weight, bias and input
at one snapshot, with one-dimensional biases and weight-only L2 penalties.
The two-logit/binary comparison derives the effective regularization factor
under a minimum-norm common-shift representation. Hidden arithmetic is checked
before tanh can conceal a nonfinite preactivation; a zero penalty does not
evaluate an unused overflowing squared-weight norm.

Uniform smoothing is explicitly a changed target and induces `KL(u || p)`,
not reverse KL. Positive class weights induce a tilted population optimum;
cost-sensitive actions are a separate decision rule. A finite temperature grid
uses separate calibration data, records every NLL/derivative and retains its
boundary status. The convex coordinate is inverse temperature; a finite binary
counterexample has negative curvature in log temperature. Grid selection,
stationarity, finite-sample calibration and subgroup/shift validity are
different claims.

NLL, accuracy, class-summed Brier and fixed-bin top-label ECE have explicit
reductions. ECE retains counts and empty bins, with right-closed boundaries
and zero included in the first bin. A five-row example has zero aggregate ECE
and opposite subgroup discrepancies. Expected-cost/reject actions preserve the
declared vocabulary columns. Raw-score class decisions are retained under
positive temperature even if rounded probabilities tie.

The language-model bridge distinguishes valid-token and equal-sequence
weighting, padding and perplexity units, including the chain rule rather than
token-independence reasoning. A zero direct logit-loss derivative does not
erase another path through a shared context value: the worked masked graph
retains a derivative of about `1.29131261`. Zero ignored-input sensitivities
in the row-independent classifiers are not generalized to arbitrary sequence
architectures or attention/input masks.

The 92 new independent numerical cases use scalar log-sum-exp and rational hand
values, scalar finite differences for reverse products, logits/log temperature,
all network parameters and inputs, and independent dense small-matrix curvature
oracles. They verify class/temperature factors, masking, row replication,
weight-scale invariance including large totals, explicit relative-weight/range
failures, entropy/KL identities, regularizer/bias conventions, binary mappings,
zero-hidden symmetry, one-class/empty queries, target/vocabulary contracts and
training-only scaler behavior. Metric/temperature/cost tests use explicit scalar
tables and bin boundaries. Separate cases check token/sequence/perplexity
reductions, shared masked paths, complete candidate order, final-outlier/label
isolation, cached final inference, consumption after an injected inference
failure, second-call rejection and paired-row resampling arithmetic.

The displayed independent scalar parameter/input audit does not call vector
forward, softmax or weight-normalization helpers. Its maximum absolute
differences are about `1.59e-11` for the linear model and `4.25e-11` for the
MLP, below the declared `1e-7` tolerance. Automated tests independently check
both the analytic definitions and this displayed auditor.

Primary readings inspected on 2026-10-25 are Stanford CS231n's softmax,
information-theory and numerical-stability passages; Szegedy et al.,
arXiv:1512.00567v3 (2015-12-11), section 7; and Guo et al., PMLR 70 (2017),
section 2 equations (2)–(3) and section 4.2 equation (9). The notebook records
links, versions/access dates, notation translation and bounded reproduction
limits. It checks local formulas/experiments, not the papers' large-network
benchmarks or a universal calibration claim.

### Execution, resources and measured results

The full local suite passes **1,112 checks**: 232 structural, 849 numerical/
project, three licensed-data and 28 isolated demonstration checks. Locked
environment sync and diff checks succeed. CI adds this file to the separate
numerical group; default CI never executes or grades unfinished learner code.

Every displayed Python block plus demonstrations executes in order in a fresh
author-audit kernel, excluding answers. Final demonstrations take 1.587 seconds
/ 180.7 MiB approximate sampled aggregate RSS; the full reference phase takes
8.533 seconds / 198.8 MiB. Both include kernel startup, plots and shutdown on
the declared Ryzen 7 6800H laptop with Python 3.12.15, NumPy 2.5.4,
Matplotlib 3.11.2, one BLAS/OpenMP thread and no GPU. Memory is the 20 ms sampled
sum of driver/kernel-descendant RSS, shared pages counted per process and
browser memory excluded, not exact physical peak memory.

Independent child streams generate 240/120/199/300 train/validation/calibration/
final rows in two features and three declared integer classes. Preprocessing is
fit on training only. Two architectures, three smoothing levels and three
seeds yield all 18 candidate records and 21,600 parameter updates. Every trace
contains its initial objective plus all 1,200 updates. Six candidates meet the
`1e-3` gradient diagnostic; all retain fixed-budget status and no extra work is
triggered by an unmet target. Equal updates do not imply equal compute,
optimization error or expressibility.

Validation chooses unsmoothed MLP seed 40402, candidate ten, with hard-target
NLL about `.23403309`. Its final gradient norm is about `.02293514`, so the
record does not assert stationarity or global optimality. Calibration selects
temperature `.5` from the six declared candidates; it is a grid boundary and
has nonzero log-temperature derivative about `.06584050`. The selection is
reported as a grid minimum, with its diagnostic retained.

Selected raw/scaled final NLL is about `.27449686 / .19908044`, Brier
`.14168175 / .10681369`, ECE `.12199322 / .02008843` and accuracy unchanged at
`.93333333`. Calibrated-minus-raw mean NLL is about `−.07541642`, with a
400-row-resample conditional percentile range `[−.10388289, −.04414452]`.
That range excludes new training/selection/calibration and dependency/shift
uncertainty. The training-only frequency baseline has NLL about `1.10238710`,
accuracy `.32666667` and ECE `.04370370`; low ECE alone is not a predictive
quality ranking. All candidate outcomes and full bin tables remain available.

The declared cost matrix produces raw/scaled realized mean costs about
`.18266667 / .12266667`, with 48 / 28 rejects out of 300. These are the
observed synthetic decision results, not guaranteed utility improvements.
Every model predicts the final features once. Raw/scaled analysis reuses cached
selected logits, and another final call on that bundle is refused, including
after an inference exception. The in-memory convention is not durable proof
against deliberate regeneration or cross-kernel reuse; that limitation is
stated. A separate 1,600-point fixed grid and training rows illustrate learned
geometry without further final-sample inference or selection.

The finite mask example gives valid-token NLL `2 log(2)` and equal-sequence
NLL `2.25 log(2)`, with exponentials `4 / 4.75682846`; arithmetic mean sequence
perplexity is a different `5.41421356`. The ignored logit row has zero direct
gradient, replication preserves the mean/parameter reduction, and an all-masked
mean is rejected. These are arithmetic examples with a declared counting unit,
not a trained language-model result.

### Rendering and preservation

HTML inspection confirms all nine sections, 38 initially closed disclosures,
rendered equations without MathJax errors and the default analytic two-panel
figure. All four actual figures are visually reviewed, including training/
validation, sparse-bin reliability and the fixed-grid geometry comparison.
Representative JupyterLab review checks blank answers, separate keyboard
opening/closing of displayed references and rendered formulas. Review and
execution copies remain outside the canonical curriculum.

All 27 preceding notebooks retain exact bytes, identifiers, answers, executable
sources and outputs. The package/dependency environment is unchanged. Stage 04
stays partial and future units have no placeholder notebooks. Publication stays
on `curriculum-notebooks`; `main` is preserved. The full nine-stage goal remains
active.
