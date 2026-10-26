# Stage 04 publication record

Current coverage: the three retained binary neural chapters, multiclass
objectives, Momentum/Adam and optimization geometry/diagnostics, 115 exercises
in six notebooks. Stage 04 remains partial: normalization/dropout,
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

## Momentum and Adam — 2026-10-25

The new `04.05` chapter contains 20 exercises in all six categories, 95 cells,
three demonstration cells and nine displayed Python reference blocks. Blank
learner answers retain separate adjacent closed hints and solutions. All 28
previously published notebooks retain exact bytes, IDs, learner work and
outputs. The repository now has 29 published chapter/project notebooks and
458 exercises. Stage 04 has five chapters / 94 exercises and stays partial;
geometry, normalization/dropout, autodiff and the capstone remain planned.
Dependencies and the reusable package are unchanged.

### Mathematical and numerical review

The chapter separates a derivative at one snapshot, the history-dependent
optimizer transition and the one-based update schedule. It derives the
unnormalized gradient-buffer recurrence, its normalized-EMA conversion and
the constant-rate mapping to displacement momentum. A two-rate hand example
shows why that last mapping fails under a schedule. A fixed scalar quadratic
has the explicitly restricted characteristic-root stability interval
`0 < rate * curvature < 2 * (1 + momentum)`; no nonconvex guarantee is inferred.

Adam's first/second raw moments, correction exponent and outside-root epsilon
are explicit. The equal-moment expectation argument does not require complete
independence, but does not remove distribution drift or make the nonlinear
adaptive direction unbiased. The corrected second moment is not centered
variance. Positive complete-gradient scaling changes the effective finite
epsilon; it does not establish general reparameterization invariance. Zero
gradient, zero rate, missing gradient and a skipped transition have different
declared effects on state and clock.

Objective L2 enters the supplied gradient and therefore the moment history.
Masked AdamW uses data-gradient moments and a simultaneous rate-scaled decay
from old parameters. A hand counterexample, bias-excluding mask and explicit
paper-to-notebook coefficient conversion prevent conflation with coupled Adam.
Plain-SGD equivalence is limited to the declared convention. Total zero-data-
direction shrinkage depends on the product over schedule/update factors.

Unequal-mass accumulation combines mean gradients at one snapshot, with one
normalization and one optimizer/schedule advance. Positive relative masses
outside float64 range fail visibly, including underflow after normalization.
Global clipping occurs after aggregation and before moments; it is not
coordinate clipping or an unbiasedness guarantee. Prescribed scalar-gradient
and vector-clipping cases show that accumulation, adaptation and clipping do
not generally commute. All-masked means and nonfinite arithmetic are rejected.

Manual tanh regression uses half-MSE and weight-only `lambda/2` penalties,
one-dimensional biases and a single output average. An independent Python-
scalar objective does not call vector forward/gradient or optimizer helpers.
Every parameter is centrally differenced; maximum fixture error at `1e-5`
is about `1.02e-11`, below `1e-7`. The displayed audit also measures large/small
finite-difference step sensitivity. Hidden nonfinite preactivations fail before
tanh; zero L2 skips an unused overflowing weight-square calculation.

The locked environment synchronized successfully. The complete local suite
passes **1,235 checks in 52.90 seconds**: 240 structural, 963 numerical/project,
three licensed-data and 29 isolated fresh-kernel execution cases. The latter
run demonstrations without learner answers or displayed reference snippets;
the separate author audit executes every displayed Python block in order.

The 114 new numerical cases independently check full scalar state sequences,
clipping and exact named shapes, raw/corrected moments, near-one/zero betas,
finite-epsilon placement, gradient scaling, state/counter failures, visible
range failures and input nonmutation. Network checks cover each parameter,
batch size/width/L2 variants, row replication, bias exclusion and the separate
displayed auditor. Schedule/accumulation checks cover both warmup extremes,
the zero-rate endpoint, safe large masses and unequal-mass/noncommuting cases.
Checkpoint tests use real files for all four algorithms, exact complete state
and history comparisons, incomplete restores, checksums, finite/schema/size/
counter/RNG validation, changed training data/order and transactional failures.
Experiment tests cover paired streams, all candidate order/outcomes, selection
means, validation/final isolation, nine cached final predictions, consumption
after an inference failure and explicitly retained failed-attempt groups.

Primary passages inspected on 2026-10-25 are Sutskever et al. §2 equations
(1)–(4); Kingma/Ba v9 Algorithm 1 and §§2–3; Loshchilov/Hutter v3 §2 and
Algorithm 2; and Reddi/Kale/Kumar v1 §3 Theorem 1 with Appendix A. The reading
table states prerequisites, local checks, notation conversion and limits.
The finite regression pilot reproduces no deep-network benchmark, online
regret counterexample or asymptotic proof.

### Actual execution and resources

Python 3.12.15, NumPy 2.5.4 and Matplotlib 3.11.2 on the CPU-only Ryzen 7 6800H,
with one BLAS/OpenMP thread. Fresh demonstrations take `1.446s / 183.7 MiB`;
all displayed references plus demonstrations take `7.808s / 201.0 MiB`.
Memory is approximate 20ms sampled aggregate driver/kernel-descendant RSS,
with shared pages counted per process and browser memory excluded; it is not
an exact physical-memory peak. Timings include kernel startup/plotting/shutdown.
GPU allocation is zero and the working set is well below 6 GiB host.

Independent train/validation/final streams have 192/96/192 rows in two
correlated features, with training-only means/scales. A 33-parameter, width-8
tanh network uses L2 `.002`. Three optimizers, three rates and three paired
initializations produce 27 complete 600-update runs: 16,200 updates and 388,800
sampled row occurrences. Batch size is 24 with replacement; the original
600-update cosine budget includes 20 warmup updates and a final zero-rate
transition. All traces, final full-gradient norms and measured candidate
times are retained; all supplied outcomes are `fixed_budget`, not converged.

Mean validation half-MSE chooses SGD `.12`, momentum `.024` and Adam `.03`,
with means about `.0864038 / .0436631 / .0141053`. Every selected rate is an
upper grid boundary, explicitly limiting tuning claims. Seeds are never
selected. The single frozen final panel caches all nine predictions and gives
mean half-MSE about `.1021331 / .0535660 / .0188181`; training-mean baseline
is `.4202362`. Seed ranges are sensitivity observations, not confidence
intervals or a universal ranking. The published synthetic data are an
author-visible learning fixture; independent comparisons need fresh frozen
final streams and declared tuning procedures.

The separate resume audit uses 2,054 total updates across continuous, partial,
complete-restored and two intentionally incomplete continuation calls. A
39,203-byte temporary checkpoint at update 173 restores moments, counter,
schedule, PCG64 state and trace and continues for 427 updates. The serialized
complete final state exactly matches continuous training on this environment.
Zeroed moments and a reset sampler preserve the clock/budget but produce final
parameter distances about `.20763697 / .07313312`; their similar objective
curves do not establish identical training. The JSON checksum detects content
changes, not authenticity or mathematical truth; no executable serialization
is loaded. Dataset arrays are supplied separately and fingerprint-checked.

### Rendering and preservation

HTML review confirms nine sections, 40 initially closed disclosures, rendered
equations with no MathJax errors and two default analytic figures. All four
actual figures are visually inspected: EMA/schedule, finite-epsilon response,
full-objective/all-validation-group comparison and incomplete-restore effects.
Representative JupyterLab review checks formulas, blank answers and keyboard
opening/closing of a separate displayed reference. Execution/review copies
remain outside the canonical notebooks. All 28 preceding notebook byte hashes
are preserved; publication remains on `curriculum-notebooks`, with `main`
unchanged. The full nine-stage goal remains active.

## Optimization geometry and diagnostics — 2026-10-25

The new prefix 04.06 adds 21 exercises in all six categories, 99 cells, three
demonstration cells and nine displayed Python blocks. Blank learner answers
retain separate adjacent closed hints/solutions. All 29 preceding notebook
byte hashes, IDs, learner work and outputs are preserved. The repository now
has 30 published chapter/project notebooks / 479 exercises. Stage 04 has six
chapters / 115 exercises and remains partial; normalization/dropout, manual
reverse-mode autodiff and the capstone remain planned. Dependencies/package
implementations are unchanged.

### Mathematical and numerical review

The chapter distinguishes local Taylor curvature, a segment-wide smoothness
bound and exact constant positive-definite quadratic eigenmode contraction.
Rate boundaries, the minimax constant rate and condition number have explicit
assumptions. A restricted two-weight linear-model example has identical
predictive functions but different coordinate-dependent Hessian spectra;
raw sharpness is not treated as a generalization guarantee.

Manual tangent-forward and tangent-reverse calculations give analytic
Hessian–vector products without a dense Hessian or autodiff framework. The
reverse tangent includes both differentiated matrix products and the hidden
activation's second derivative. Mean half-MSE averages once at the output;
L2 penalizes weights only, and inputs/targets/directions stay fixed during
parameter differentiation. Tanh/linear activations support smooth curvature;
ReLU has a declared zero first-order derivative at zero and its classical
Hessian products are deliberately outside the implementation's scope.

The full Hessian is separated from PSD Gauss–Newton plus penalty and the
residual-curvature term. A hand stationary tanh saddle has loss .5, a zero
gradient, full spectrum (-1,0,1,1), GN spectrum (0,0,0,1), and a directly checked
loss-decreasing direction. Zero-residual fixtures freeze their generated
targets before differentiation and give H-G errors at floating-point scale.
Dense inspection records raw symmetry error before using a symmetric part.

An independent list/Python-math objective calls no vector forward, unpack,
gradient or tangent helper. At step 1e-4, the displayed moderate-scale fixture
has gradient/product maximum absolute errors about 2.98e-10 / 1.45e-9, with
declared tolerances 1e-6. A separate saddle product check is about 9.44e-9.
Larger/smaller steps show truncation/cancellation sensitivity; the 1e-6-step
product error is about 9.19e-6 and is retained rather than called a pass at the
middle-step tolerance. Exact products still use floating-point arithmetic.

The locked environment synchronized successfully. The complete local suite
passes 1,319 checks in 54.85 seconds: 248 structural, 1,038 numerical/project,
three licensed-data and 30 isolated fresh-kernel demonstration cases. Learner
answers and displayed references are omitted from default execution checks;
the separate author audit executes all displayed Python blocks in order.

The 75 new cases independently verify all parameters, inputs, prediction
tangents and Hessian products across batch sizes, depths, tanh/linear choices
and penalties. Other cases check row replication, bias/penalty masks, product
linearity/symmetry, exact hand matrices, output Jacobians, changed-residual
curvature, zero-residual equality and a single-layer closed-form Hessian.
The displayed scalar auditor is separately checked and detects an injected
wrong product. Independent eigenbasis paths, initialization draw/scale pairing,
output-probe VJPs and moment identities, ReLU gates and smoothness boundaries,
fixed-row covariance laws, clipping counterexamples, failed work accounting,
training-only preprocessing and visible numerical-range errors are covered.

For fixed parameters the row-gradient covariance uses ddof=0. Exact replacement
batches give S/b; uniform subsets give (n-b)S/[b(n-1)]. Shared deterministic
penalty gradients change the mean but not centered covariance. Six finite
enumerations, including 4,096 ordered batches at n=8,b=4, agree below the
declared 1e-12 tolerance. A separate scalar (-4,2,2) distribution has mean zero
and clipped mean 1/3, refuting unrestricted clipping-unbiasedness reasoning.
Changing-parameter epoch sampling is explicitly a different conditional claim.

Primary passages inspected on 2026-10-25 are Pearlmutter's dated author copy
§§3/4.1 and equations (2)–(5); Glorot/Bengio §4.2; He et al. v1 §2.2 equations
(5)–(10); and Bottou/Curtis/Nocedal v3 equations (4.25)–(4.26) with the following
mini-batch discussion. The chapter declares prerequisite/local-check/
reproduction boundaries; it copies no coursework answers or paper benchmark.

### Actual execution, experiments and resources

Fresh demonstrations take 1.367s / 183.5 MiB; all displayed references plus
demonstrations take 3.587s / 210.2 MiB on Python 3.12.15 / NumPy 2.5.4 /
Matplotlib 3.11.2, CPU-only Ryzen 7 6800H, one BLAS/OpenMP thread. RSS is
approximate 20ms sampled aggregate driver/kernel-descendant memory, counting
shared pages per process and excluding the browser, not exact physical peak.
Timings include kernel startup, plotting and shutdown. GPU allocation is zero.

Three rotated quadratic paths retain all 80 updates. Final losses at rates
.03/.07/.09 are about .00405818 / .00000481 / 1.5771e17; all-mode instability
is retained and not hidden outside a contour viewport. The default figures
illustrate analytic contours/activations and covariance denominators without
defining completed learner implementations.

The 24 depth-five initialization profiles use widths (8,16,16,16,16,16,1),
1,249 parameters, 128 fixed Gaussian rows and a common independent output
probe with RMS 1. Hidden normal scales vary while nonzero output scaling is
fixed. Means, variances, second moments, derivative/low-derivative statistics
and sensitivity RMS are retained, including input VJPs. Positive rescaling
with zero-bias ReLU preserves gate patterns under these paired draws despite
different transmitted sensitivities. Normal draws match the cited Glorot
variance, not its uniform distribution or benchmark. No all-direction
Jacobian-conditioning or finite-depth variance guarantee is inferred.

The shallow mechanism study uses a 25-parameter network on separately
generated 96-row train/validation panels, each constructed from 48 draws and
their sign-reversed partners. Training-only scaling is explicit; within-panel
dependence is intentional and prevents an IID-interval interpretation.
Five prescribed initialization/rate conditions and three paired seeds produce
15 retained runs. The common objective ceiling 1e12 stops oversized-rate runs
after 7/7/6 accepted updates; all other runs use 300. Actual completed/attempted
work is 3,620 updates, not the 4,500 maximum budget. Numerical failures preserve
the last finite state and failed-attempt count. Initial Glorot vectors are
identical in the corresponding oversized-rate comparison.

Zero networks remain stationary with training loss about .4316997 and a
gradient norm about 8.67e-19; this is not labeled a minimum. Tiny shallow
networks learn to losses .0155–.0224, so the deep profile is not generalized
into a universal small-initialization failure rule. Every seed, ceiling-stopped
outcome, trace, actual count and descriptive validation score is retained.
There is no candidate selection or final-test claim in this mechanism study.

### Rendering and preservation

HTML review confirms nine sections, 42 closed disclosures, two default plots
and equations without MathJax errors. Seven actual figures are visually
reviewed, including trajectories, curvature, covariance, initialization and
all failed/stationary training outcomes. Representative JupyterLab inspection
checks the blank code-answer cell, separately keyboard-opened/closed reference
and rendered math. Review/execution copies remain outside the canonical
notebooks. The 29 previous notebooks stay byte-for-byte intact. Publication
remains on curriculum-notebooks, main is unchanged and the nine-stage goal
remains active.
