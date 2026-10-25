# Machine Learning From Scratch

A learning path from manually implemented regression to small Transformers,
language-model training, measured systems improvements, and focused research.
The audience already knows calculus, linear algebra, and programming.

The long-term goal is to read a new AI paper, explain its assumptions, derive
its essential equations, implement a bounded version, independently verify it,
and critique the evidence. The [foundations and topic connections](docs/foundations-and-connections.md)
and [research practice](docs/research-practice.md) describe how this develops
through the nine stages. Derivation accompanies explanation; critique accompanies
evaluation. Paper studies begin with small algorithms and grow with the path.

The curriculum is under active construction. **Stages 00–02 are complete,
and the existing neural-network chapters are retained in Stage 04.**
The availability table below records which material has passed
its publication gates. Later-stage plans are not finished lessons.

Use **curriculum-notebooks** as the reference branch while this work awaits
merging into main. Keep learner answers on a separate study branch.

## Learning path and availability

<!-- availability:start -->

| Stage | Learning objective | Availability |
| --- | --- | --- |
| [00 — Mathematical Tools for Machine Learning](notebooks/00-mathematical-tools/README.md) | Apply existing calculus, linear algebra, and programming knowledge to ML arrays, statistical objectives, and finite-precision computation. | ready |
| [01 — Machine Learning Fundamentals](notebooks/01-ml-fundamentals/README.md) | Turn predictive questions into explicitly defined models, objectives, fitting procedures, and defensible evaluations. | ready |
| [02 — Classical Machine Learning](notebooks/02-classical-ml/README.md) | Compare different inductive biases rather than treating one fitted model as a universal solution. | ready |
| [03 — Unsupervised and Probabilistic Learning](notebooks/03-unsupervised-learning/README.md) | Discover and model structure without confusing a visually persuasive representation with validated evidence. | partial |
| [04 — Neural Networks and Deep Learning Foundations](notebooks/04-neural-networks/README.md) | Understand differentiation, optimization, representation, and generalization as distinct parts of neural-network learning. | partial |
| [05 — Deep Learning Architectures](notebooks/05-deep-learning/README.md) | Use tensor/autograd infrastructure while continuing to build and explain model architectures and training loops. | planned |
| [06 — Transformers and Generative Models](notebooks/06-transformers-generative/README.md) | Construct a complete small language-model pipeline whose tokenizer, data boundaries, architecture, training objective, generation, and evaluation are all understood. | planned |
| [07 — Advanced Training and ML Systems](notebooks/07-ml-systems/README.md) | Improve a correct language model through measured systems and data changes while protecting the validity of quality comparisons. | planned |
| [08 — Advanced Specializations and Research](notebooks/08-research/README.md) | Practice research through a focused, falsifiable reproduction rather than treating exposure to every specialty as mastery. | planned |

<!-- availability:end -->

Follow Stages 00–07 in the recommended order. Prerequisites identify what is
actually needed: the available Stage 04 chapters can be studied after Stage 01
while Stages 02–03 are being developed. Stage 08 contains independent tracks;
choose a specialization rather than attempting every research field.

Stage 00 is an applied review. Experienced learners may use its diagnostics
and capstone to demonstrate readiness and skip explanations they already know.
The older foundational chapters retain optional elementary refreshers.

The breadth follows [Stanford CS229](https://cs229.stanford.edu/index.html-spr26);
the language-model implementation, data, training, systems, and evaluation
milestones draw on [Stanford CS336](https://cs336.stanford.edu/). These are
curriculum references. This repository's teaching examples and exercises are
original, self-contained work.

## How learning works

Every chapter provides:

1. Introduction and learning objectives.
2. Prerequisites and mathematical foundations.
3. Theoretical development and worked examples.
4. Guided exercises.
5. Implementation challenges.
6. Experiments and visualizations.
7. Interpretation and critical analysis.
8. Review questions and knowledge checks.
9. Summary and next learning steps.

Conceptual, mathematical, numerical, programming, experimental, and critical
tasks have dedicated editable answer cells. Each answer is followed by
**Show hint** and **Show worked solution — try the exercise first**, separately
collapsed by default. Reference Python is fenced Markdown text: Run All does
not execute it or populate solution functions. Named references may depend on
earlier snippets in the same chapter, never another notebook's live kernel.

Attempt your answer first, use a hint when needed, then compare reasoning.
Open-ended references are justified examples rather than unique answers.
Experiments give reproducible investigations and interpretation guidance;
results must come from running them.

Every stage ends with a project and four mastery requirements:

- **Explain:** mathematics, assumptions, dimensions, and objective.
- **Implement:** the essential algorithm and experimental procedure.
- **Verify:** independent numerical checks and meaningful edge cases.
- **Evaluate:** baselines, limitations, uncertainty, and measured resource use.

Project workspaces contain blank implementation cells, self-check guidance,
and a report template. Default validation checks the reference material and
demonstrations, not unfinished learner implementations.

## Installation and study branches

The current delivery uses Python, NumPy, and Matplotlib. JupyterLab, ipykernel,
and pytest are in the default development group.

~~~bash
git clone --branch curriculum-notebooks https://github.com/rafaself/machine-learning-from-scratch.git
cd machine-learning-from-scratch
uv sync --locked
git switch -c study/your-name
uv run jupyter lab
~~~

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if
needed. The committed .python-version selects Python 3.12, and uv.lock records
the environment. Choose **Python 3 (ipykernel)** in JupyterLab.

Keep completed answers, experiments, and project reports on your study branch.
Save or commit your work before switching. After the curriculum is merged into
main, use main as the clean reference in place of curriculum-notebooks. Do not
merge completed learner answers into the reference material.

## Implementation progression

| Stages | Implementation policy |
| --- | --- |
| 00–04 | Primarily NumPy; manual models, gradients, optimizers, and networks. Build a small autodiff engine after deriving backpropagation manually. |
| 05–06 | Introduce PyTorch for tensors, autograd, and GPU execution; implement architectures and training loops explicitly. |
| 07–08 | Add specialized tools only for a measured learning objective that fits the target machine. |

No new framework is required for this delivery. Advanced dependencies are
introduced when their stages are authored. The minimal package supplies the
licensed offline Iris data needed by the Stage 02 project; model references
remain in separately collapsed notebook cells.

Examples are rows of X, shape (n, d). The original regression chapters use
targets (n,); network chapters explicitly use (n, 1). Define reductions before
differentiating, and include an observation average once. Distinguish data
loss from a regularized objective; biases are unpenalized when stated.

Fit preprocessing on training data or each training fold. Use validation for
choices and reserve test data for a final evaluation. Record seeds, splits,
loss normalization, selection effort, work budgets, and failed runs.

## Local compute contract

The target machine inspected on 2026-10-22 has a Ryzen 7 6800H, approximately
14 GiB usable RAM, and an RTX 3060 Laptop GPU with 6 GiB VRAM.

- Standard experiments target at most **6 GiB host working memory** and
  **5 GiB GPU memory**.
- Correctness checks take a few minutes; ordinary experiments target about
  **30 minutes**; capstones support checkpointed sessions of roughly **two hours**.
- Required experiments run locally. No cloud account or multi-GPU hardware is
  required. Distributed concepts use small CPU process demonstrations.
- Every authored stage publishes measured configuration/runtime/memory
  evidence before being marked ready. Planned budgets are not measurements.

Stages 00–02 require CPU and small local datasets. Future Transformer
experiments use small models and bounded corpora; their presets are finalized
after measurement. Research reports must state what a local-scale experiment
can support.

## Repository and migration

~~~text
curriculum.json                 Stage order, prerequisites, readiness, chapter/project paths
notebooks/00-mathematical-tools/
notebooks/01-ml-fundamentals/
notebooks/02-classical-ml/
notebooks/03-unsupervised-learning/
notebooks/04-neural-networks/
notebooks/05-deep-learning/
notebooks/06-transformers-generative/
notebooks/07-ml-systems/
notebooks/08-research/
projects/                       Published project workspaces and report templates
src/ml_from_scratch/             Minimal package, including licensed offline project data
tests/                          Registry, structure, execution, and numerical checks
~~~

The original 01-foundations directory moved to 01-ml-fundamentals, and
02-neural-networks moved to 04-neural-networks. Original filenames, exercise
IDs, cell IDs, learner answers, executable sources, and outputs are preserved;
cross-directory Markdown links were updated.

Legacy exercises keep IDs such as 1.01 and 6.20. New chapters use stage/chapter
prefixes such as 00.01.01; projects use prefixes such as 01.P.01. Study order
comes from the registry rather than from those legacy identifiers.

## Validation

The [publication record](docs/validation.md) describes numerical, execution,
preservation, rendering, and resource checks for this delivery. Stage guides
publish measured time/memory for the supplied presets.
The [Stage 02 record](docs/stage-02-validation.md) covers its seven chapters,
licensed offline data and integrated benchmark capstone.
The [Stage 03 record](docs/stage-03-validation.md) covers the published
clustering, Gaussian-mixture/EM, PCA/SVD, Bayesian/variational and
density/anomaly chapters; its representation-study capstone remains planned.

~~~bash
uv run pytest
uv run pytest tests/test_curriculum.py
uv run pytest tests/test_numerics.py
uv run pytest tests/test_classical_benchmark.py
uv run pytest tests/test_clustering.py
uv run pytest tests/test_mixtures.py
uv run pytest tests/test_pca.py
uv run pytest tests/test_bayesian.py
uv run pytest tests/test_density.py
uv run pytest tests/test_datasets.py
uv run pytest tests/test_execution.py
~~~

Structural checks cover registry readiness, prerequisites, published files,
chapter structure, unique exercise IDs, local links, and collapsed references.
Numerical checks compare displayed implementations to independent objectives
and fixed values. Execution checks run published chapters and project
demonstrations in separate kernels and temporary directories, requiring plots
to render without evaluating learner answers or reference snippets.

Tests allow completed answer cells on study branches. They do not grade those
answers or prove every mathematical claim. Review equations, plots, and
experimental interpretations as well as code execution. No filter, hook, or
validation step clears retained notebook outputs.

GitHub Actions runs the check groups separately on pull requests and pushes to
main or curriculum-notebooks, using the locked environment and actions pinned
to commit SHAs with read-only repository permissions.
