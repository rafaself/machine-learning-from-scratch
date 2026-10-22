# Machine Learning From Scratch

An educational repository for understanding machine learning through mathematics,
manual implementations, and small experiments. Derive each algorithm, implement
it with Python and NumPy, then investigate its behavior with visualizations.

## Learning roadmap

| Stage | Topics | Status |
| --- | --- | --- |
| Foundations | Linear regression | Notebook outline ready; exercises pending |
| Foundations | Gradient descent, logistic regression | Planned |
| Neural networks | Perceptron, backpropagation, multilayer perceptron | Planned |
| Further studies | Regularization, optimization, and more advanced concepts | Planned |

Add notebooks and algorithm modules as each study begins. The initial notebook
contains prompts and empty code cells, with no completed exercises.

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/rafaself/machine-learning-from-scratch.git
cd machine-learning-from-scratch
uv sync
uv run jupyter lab
```

The project supports Python 3.12+. `.python-version` selects Python 3.12 by
default; uv can download it if needed. `uv sync` creates `.venv`, installs the
local package in editable mode, and includes the development dependencies.
`uv.lock` records the resolved dependency versions. Use `uv sync --locked` to
require the committed lockfile to match the project configuration.

In JupyterLab, open
[`notebooks/01-foundations/linear-regression.ipynb`](notebooks/01-foundations/linear-regression.ipynb)
and select **Python 3 (ipykernel)**. Starting JupyterLab through `uv run` uses
the project environment without a separate kernel registration step.

NumPy and Matplotlib support numerical work and plots. The default `dev` group
provides JupyterLab, ipykernel, and pytest.

## Repository organization

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── notebooks/
│   └── 01-foundations/
│       └── linear-regression.ipynb
└── src/
    └── ml_from_scratch/
        └── __init__.py
```

Keep derivations, experiments, and plots in `notebooks/`. The importable
`ml_from_scratch` package is an empty starting point: extract code into `src/`
when reuse or independent testing justifies it. Add `tests/` alongside the first
implementation; no algorithm tests exist yet. Once added, run them with:

```bash
uv run pytest
```

## Notebook and contribution conventions

Use these sections in every study notebook:

1. Learning objectives
2. Mathematical foundations
3. Implementation from scratch
4. Experiments and visualizations
5. Results and conclusions
6. Further exercises

Use English and standard Python naming and formatting. Explain assumptions,
notation, and array shapes before implementing an algorithm. Favor direct,
readable NumPy operations and small functions; add abstractions when a concrete
need arises. Implement the learning algorithms manually, without PyTorch,
TensorFlow, or scikit-learn.

Make experiments reproducible with explicit `numpy.random.default_rng(seed)`
seeds and documented parameters. Restart the kernel and run all cells before
sharing results; clear outputs and execution counts before committing notebooks.
For reusable code, test observable behavior, numerical tolerances, shapes, and
relevant edge cases.

Keep changes focused and dependencies minimal. Manage dependencies with `uv add`
(or `uv add --dev` for development tools), and commit `pyproject.toml` and
`uv.lock` together. Virtual environments, caches, notebook checkpoints, and
packaging output are excluded from Git.
