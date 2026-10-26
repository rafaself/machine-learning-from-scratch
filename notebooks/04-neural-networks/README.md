# Stage 04 — Neural Networks and Deep Learning Foundations

**Availability:** partial. Ready units are linked below; planned units
have no placeholder notebooks. Registry readiness describes published coverage.

## Goal and prerequisites

Understand differentiation, optimization, representation, and generalization as distinct parts of neural-network learning.

**Prerequisites:** Stage 01 and the applied Stage 00 tools. The retained neural
chapters can be studied directly after Stage 01 under their stated assumptions;
the published classical/probabilistic stages provide useful comparison
experience rather than mandatory prerequisites for every neural unit. The
multiclass and optimizer chapters build on the retained backpropagation and
MLP chapters. The valid-token accumulation connection uses the multiclass unit.

## Ordered units

1. Perceptron: label conventions, signed margins, updates, convergence assumptions, and representation limits.
2. Backpropagation: shared paths, cached parameter snapshots, batched affine/activation rules, and gradient checking.
3. Multilayer perceptron: one/two hidden layers, initialization, manual training, regularization, and validation.
4. Multiclass learning objectives: categorical likelihood, stable log softmax, vector/Hessian products, manual network derivatives, soft targets, weights/masks, temperature calibration, decision costs, and valid-token reductions.
5. Momentum and Adam: exact state/clock conventions, initialization correction, finite epsilon, L2 versus AdamW, weighted accumulation/clipping, schedules, paired-seed comparisons, and verified JSON checkpoint continuation.
6. Optimization geometry and diagnostics: curvature/spectrum, stochastic gradient variation, activation/gradient distributions, initialization, saturation, and controlled failure analysis.
7. Normalization and dropout: training/evaluation behavior, statistical axes, randomness, and reproducibility.
8. Automatic differentiation: build a small reverse-mode engine after deriving gradients manually; accumulation, graph reuse, broadcasting reductions, and elementary operations.

<!-- availability:start -->

- [perceptron](perceptron.ipynb): **ready**
- [backpropagation](backpropagation.ipynb): **ready**
- [multilayer perceptron](multilayer-perceptron.ipynb): **ready**
- [multiclass learning objectives](multiclass-learning-objectives.ipynb): **ready**
- [momentum and adam](momentum-and-adam.ipynb): **ready**
- optimization geometry and diagnostics: **planned**
- normalization and dropout: **planned**
- automatic differentiation: **planned**
- autodiff and network: **planned**

<!-- availability:end -->

## Implementation requirements

Use NumPy. The autodiff engine remains a planned learner-built algorithm, not
a framework dependency. The published multiclass unit derives and implements
log-domain categorical objectives, manual linear/tanh gradients, exact class
alignment and explicit weighted/masked reductions. Binary retained chapters
keep `(n,1)` targets/probabilities; multiclass labels are `(n,)` and target/
probability arrays are `(n,K)`. Weight matrices follow input/output widths and
biases remain one-dimensional. No reference model is moved into the package.
The future engine must be compared with these manual derivatives and
independent finite differences.

The optimizer unit implements complete named-shape NumPy transitions for SGD,
gradient-buffer momentum, corrected Adam and masked AdamW. It independently
audits a manual regression MLP, then separates objective/gradient correctness,
optimizer state correctness and experimental comparisons. Checkpoints include
moments, update counter, the original schedule budget, PCG64 sampler state,
processed-training fingerprint and trace. A real file round trip demonstrates
exact continuation on the measured environment; weights alone do not suffice.

Every published notebook uses the nine chapter sections, explicit notation,
worked examples, six exercise types, and dedicated answer cells followed by
separate collapsed hints and solutions. Demonstrations run without learner work.

## Mastery gate

- Explain the graph, shapes, categorical likelihood/decisions, reductions, optimizer state, and evaluation-mode behavior.
- Implement both explicit network derivatives and a small reusable differentiation engine.
- Verify every parameter/input, class-column alignment, weighted/masked reduction, shared path, broadcast reduction, and optimizer state transition.
- Evaluate training across seeds and fixed splits using common held-out metrics while separating optimization, representation, generalization and probability-calibration claims.

Keep explanations, code/checks, measured results, and limitations as separate
evidence. A completed reading checklist alone does not establish these abilities.

## Capstone

Autodiff and network: validate the engine against independently derived gradients, then use it to train a small network and compare numerical behavior, work, and predictions with the manual version.

Submit learner work on a study branch. The planned capstone will supply instructions,
blank implementation cells, self-check guidance, and a report template;
default CI does not execute unfinished learner implementations.

## Compute and reproducibility

The three retained chapters were rechecked locally on 2026-10-22; the new
multiclass and optimizer units were measured on 2026-10-25. They use Python 3.12.15,
NumPy 2.5.4, Matplotlib 3.11.2, the CPU-only Ryzen 7 6800H, and one
OpenBLAS/OpenMP thread. These are measurements of available chapters, not
certification of planned geometry, normalization/autodiff or the full capstone.

| Published unit | Measured date | Fresh-kernel demo seconds / RSS MiB | All displayed references seconds / RSS MiB |
| --- | --- | --- | --- |
| Perceptron | 2026-10-22 | 4.017 / 197.0 | 1.789 / 205.5 |
| Backpropagation | 2026-10-22 | 1.417 / 200.1 | 2.165 / 215.4 |
| Multilayer perceptron | 2026-10-22 | 1.497 / 203.0 | 9.859 / 234.7 |
| Multiclass learning objectives | 2026-10-25 | 1.587 / 180.7 | 8.533 / 198.8 |
| Momentum and Adam | 2026-10-25 | 1.446 / 183.7 | 7.808 / 201.0 |

Elapsed time includes kernel startup, plotting, and shutdown. RSS is the
approximate 20 ms sampled sum for the driver and kernel descendants, including
shared pages counted per process and excluding browser memory. Each reference
phase also runs its demonstrations, omits answer cells, and leaves canonical
outputs untouched. No GPU is required for these supplied presets.

The multiclass comparison generates independent 240/120/199/300-row
training/validation/calibration/final samples in two features and three classes.
Two architectures, three smoothing levels and three initialization seeds
produce 18 complete candidate records. Each gets 1,200 full-batch updates,
step size .2, L2 coefficient .001 and MLP width 24: 21,600 parameter updates
in total. All fixed-budget and gradient-target outcomes are retained.
Training-only preprocessing precedes hard-target validation-NLL selection;
the selected model's temperature uses six separate calibration candidates.
One final panel saves all 18 logit arrays plus baseline/raw/scaled metrics,
full bin tables, cost actions and 400 paired-row NLL resamples. A predeclared
1,600-row probe grid illustrates geometry without more final-row inference.
Generic helpers cap 2,000 rows, 32 input/class columns, hidden width 64,
3,000 training steps, 16 temperatures and 800 resamples. Memory stays far below
6 GiB host, with zero GPU allocation. These are program runtimes, not the time
needed to derive, implement and critique the material.

The optimizer pilot uses independent 192/96/192-row train/validation/final
samples, two correlated features, a 33-parameter tanh MLP and weight-only L2
coefficient .002. Three optimizers, three rates and three paired seeds give
27 complete runs, each with 600 updates, batch size 24 and a 20-update warmup/
cosine schedule. All 16,200 update records and 388,800 sampled row occurrences
are accounted for. Each algorithm's rate is chosen by mean validation
half-MSE across all three seeds; seeds are not selected. Every chosen rate
hits its declared upper grid boundary, an explicit tuning limitation. One
final panel caches all nine selected-rate/seed predictions and a training-mean
baseline. A separate 2,054-update resume audit compares complete restoration
with zeroed moments and a reset sampler at the same clock. Generic helpers
bound rows/features/widths, update counts, accumulation groups and JSON size;
required work stays far below the host limit and uses no GPU.

CPU only. Small batches, widths, and graphs keep engine/debugging work inspectable. Large-model training is introduced after this foundation.

Use fixed splits/seeds, record objective normalization and model selection,
and disclose unsuccessful runs. Standard experiments target about 30 minutes;
resumable capstones target roughly two hours. Correctness checks stay short.
Reported results and resource measurements must come from actual executions.

## Research connection

Each developed unit includes a bounded primary-source study with an equation,
independent check, and a claim the local experiment can support. See the
[foundations map](../../docs/foundations-and-connections.md) and
[research practice](../../docs/research-practice.md). New coverage remains
planned until its notebook and measurements pass publication gates.

## Readings

The multiclass unit studies [Stanford CS231n's softmax, information-theory and numerical-stability passages](https://cs231n.github.io/linear-classify/),
translating its column-vector notation into row-wise arrays and independently
checking scores, all parameter/input derivatives and reductions.
[Szegedy et al., v3 (2015-12-11), §7](https://arxiv.org/pdf/1512.00567v3)
supplies the bounded label-smoothing study: derive the mixture target,
target-to-prediction KL direction and changed population optimum.
[Guo et al. (2017), §2 equations (2)–(3) and §4.2 equation (9)](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
supplies fixed-bin ECE and temperature scaling. Retain bin counts and grid
boundaries, distinguish inverse-temperature convexity from nonlinear
parameterizations, and avoid translating empirical effectiveness into a
conditional or distribution-shift guarantee. The notebook states access dates
and reproduction limits; no paper's large-model benchmark is reproduced.

The optimizer unit studies [Sutskever et al. (2013), §2 equations (1)–(4)](https://proceedings.mlr.press/v28/sutskever13.pdf),
translating displacement velocity into a gradient buffer only under a constant
rate. [Kingma/Ba, v9 (2017-01-30), Algorithm 1 and §§2–3](https://arxiv.org/pdf/1412.6980v9)
supplies exact moment/correction and epsilon conventions.
[Loshchilov/Hutter, v3 (2019-01-04), §2 and Algorithm 2](https://arxiv.org/pdf/1711.05101v3)
supplies the coupled/decoupled decay comparison with explicit coefficient
conversion. [Reddi/Kale/Kumar, v1 (2019-04-19), §3 Theorem 1 and Appendix A](https://arxiv.org/pdf/1904.09237v1)
supplies a restricted counterexample study: distinguish online average regret,
finite objective training and universal convergence claims. The notebook
declares prerequisites, local checks, access dates and reproduction limits.

[CS229](https://cs229.stanford.edu/index.html-spr26) supplies broader
neural-network scope. Exact reverse-mode passages will be selected when that
unit is developed.

The lessons are original and self-contained. External coursework supplies
scope and alternative explanations, not copied assignment answers.

## Current coverage

The original perceptron, backpropagation and MLP chapters retain their original
exercise IDs and bytes. The new multiclass chapter uses prefix `04.04` and
adds 19 exercises. The optimizer chapter uses prefix `04.05` and adds 20,
bringing Stage 04 to 94 exercises in five published chapters. Optimization
geometry, normalization/dropout, autodiff and the integrated capstone remain
planned. Five published chapters do not mean that the full stage has passed
its completion gate.

See the [Stage 04 publication record](../../docs/stage-04-validation.md) for
current numerical, preservation, execution and rendering evidence.

[Return to the learning path](../../README.md)
