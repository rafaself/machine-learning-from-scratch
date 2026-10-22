# Machine Learning From Scratch

A notebook-based curriculum for learning machine learning through mathematical
derivations, small worked examples, NumPy implementations, and reproducible
experiments. Read it as an interactive textbook: understand a model, derive its
learning rules, implement them yourself, and examine what the evidence supports.

Learner answer cells start **blank**. Every numbered exercise has a dedicated
answer cell followed by a collapsed hint and a collapsed worked solution.
Try the exercise first, use a hint if needed, then compare your reasoning.
Complete your own work on a separate Git branch.

## Recommended study order

| Order | Notebook | Main questions |
| --- | --- | --- |
| 1 | [Linear regression](notebooks/01-foundations/linear-regression.ipynb) | How do data become matrices, residuals, an objective, and a fitted line? |
| 2 | [Gradient descent](notebooks/01-foundations/gradient-descent.ipynb) | How do batch, stochastic, and mini-batch updates differ, and when are steps stable? |
| 3 | [Logistic regression](notebooks/01-foundations/logistic-regression.ipynb) | How do scores become probabilities, likelihoods, gradients, and decisions? |
| 4 | [Perceptron](notebooks/02-neural-networks/perceptron.ipynb) | What can a linear boundary learn, and what does its convergence guarantee assume? |
| 5 | [Backpropagation](notebooks/02-neural-networks/backpropagation.ipynb) | How do local derivatives combine through shared paths and batched network layers? |
| 6 | [Multilayer perceptron](notebooks/02-neural-networks/multilayer-perceptron.ipynb) | How do one and two hidden layers change representation, training, and generalization? |

Follow this order for a first pass. Each chapter restates its notation and
foundations so it can also serve as an independent reference. No notebook
requires another notebook's live kernel state or imports a completed solution.
Possible later studies include multiclass models, other optimizers, and more
advanced regularization; these are outside the current six-chapter curriculum.

## Mathematical and programming preparation

Start with arithmetic, fractions, basic algebra, and introductory Python
(variables, arrays, functions, loops, and indexing). Linear regression introduces
vectors, matrix products, sums, partial derivatives, and the chain rule. Gradient
descent develops local approximations, curvature, averages, and sampling.
Logistic regression introduces odds, exponentials, logarithms, likelihood, and
binary cross-entropy. The neural-network chapters develop nonlinearities,
computation graphs, matrix derivatives, and initialization.

You do not need prior ML frameworks or a completed calculus course. Work through
numerical examples before implementing a general formula. Revisit an earlier
foundation when a symbol is unfamiliar. Advanced arguments, such as the
perceptron update bound, explain their assumptions and intermediate inequalities.

## Installation and execution

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

~~~bash
git clone https://github.com/rafaself/machine-learning-from-scratch.git
cd machine-learning-from-scratch
uv sync --locked
uv run jupyter lab
~~~

Python 3.12+ is supported; the committed `.python-version` selects 3.12.
uv can download the requested interpreter if unavailable. The committed
`uv.lock` records the dependency resolution. `uv sync` is also supported;
`--locked` additionally requires the lockfile to match project configuration.

In JupyterLab, open the first notebook and select **Python 3 (ipykernel)**.
Launching through uv uses the project's `.venv` without separate global kernel
registration. NumPy and Matplotlib support numerical work and plots. The default
development group includes JupyterLab, ipykernel, and pytest.

## Complete exercises on your own branch

Begin with a clean checkout of the reference material:

~~~bash
git switch main
git pull --ff-only
git switch -c study/your-name
uv run jupyter lab
~~~

Keep completed answers and experiments on this branch. Commit progress normally,
and return to a clean `main` to inspect the original learner answer cells.
Save or commit your work before switching branches. Do not merge completed
exercise answers into the reference curriculum.

Every exercise has a sequential chapter-specific ID, a type, explicit
instructions, an expected outcome, and an immediately adjacent answer cell:

- **Conceptual, numerical, mathematical, and critical questions:** edit the next
  Markdown cell. Explain assumptions and reasoning; use Jupyter LaTeX for
  equations and show intermediate calculations.
- **Programming challenges:** implement your own code in the next code cell.
  TODO comments mark the place to work. Reference implementations are displayed
  as fenced code inside the collapsed solution, not executable notebook cells.
- **Experimental investigations:** use the next code cell for runs, tables,
  plots, and recorded settings. Add written interpretation in comments or in
  additional cells on your study branch.

After each answer cell, **Show hint** gives a nudge without the full answer.
**Show worked solution — try the exercise first** reveals a separate reference.
For open-ended questions, the reference is one justified response, not the
only acceptable answer. Experimental solutions include runnable example
investigations and explain how to interpret variable results.

Reference code stays displayed as text until you copy it into a code cell.
Run All does not execute reference snippets or define solution functions.
Some snippets build on named earlier solutions in the same chapter; no
snippet depends on another notebook. Hints and solutions use native HTML
disclosure blocks, with no additional dependency.

Demonstration cells are labeled in notebook metadata and contain working
illustrations, data generation, or visualizations. They run in order with blank
exercise cells and do not depend on learner-defined functions. Reference
notebooks may retain demonstration outputs; no filter, hook, or automation clears
notebook outputs.

## Conventions and reproducible experiments

Use English, standard Python naming, readable NumPy operations, and small
functions. Implement learning algorithms manually without PyTorch, TensorFlow,
scikit-learn, or automatic differentiation. Extract reusable code into
`src/ml_from_scratch/` only after a concrete need for reuse or testing.
Reference implementations stay in collapsed notebook Markdown; the package
remains reserved for independently justified reusable code.

Examples are rows of `X`, shape `(n, d)`. Linear/logistic regression use targets
`(n,)`; network chapters explicitly switch to `(n, 1)` targets/probabilities.
Network weights have shape `(previous_width, next_width)`. State shapes before
multiplying; distinguish matrix `@` from entrywise `*`.

Record seeds, splits, preprocessing, loss normalization, learning rates,
initialization, work budgets, and selection criteria. Fit preprocessing on
training data, use validation data for choices, and reserve test data for final
evaluation. Distinguish data loss from regularized objectives and computational
work from update counts. Report failures and variation across seeds.

## Repository organization and checks

~~~text
notebooks/01-foundations/        Linear regression, optimization, logistic regression
notebooks/02-neural-networks/    Perceptron, backpropagation, one/two hidden layers
src/ml_from_scratch/             Empty package reserved for justified reusable code
tests/                          Structure, execution, and numerical checks
pyproject.toml, uv.lock          Dependencies and reproducible environment
AGENTS.md                       Educational conventions for future editing
~~~

Run the repository checks with:

~~~bash
uv run pytest
~~~

Structural tests validate notebook schemas, chapter structure, exercise IDs,
adjacent answer cells, collapsed reference blocks, and local notebook links.
Execution tests run each chapter's demonstrations in a fresh kernel and a
temporary working directory, checking that plots render without running learner
answers or reference snippets. Notebook files and retained outputs are not
rewritten. Run either group separately with:

~~~bash
uv run pytest tests/test_curriculum.py
uv run pytest tests/test_execution.py
uv run pytest tests/test_numerics.py
~~~

Numerical checks compare displayed reference gradients to central differences
of independently written objectives, including mean-loss scaling, unpenalized
biases, one-example batches, both network depths, and smooth regions of tanh and
ReLU. They also check fixed worked values, extreme logits, plotted activation
slopes, and dataset shape/split conventions. Only reference function definitions
are loaded for these checks; the displayed experiments are not run as part of
ordinary notebook execution.

The checks allow completed answer cells on study branches; they do not grade
answers or replace mathematical review.
Before sharing a changed demonstration, restart its kernel and run the notebook
from top to bottom. Review mathematical reasoning and rendered equations as
well as execution.

Keep dependencies minimal and commit `pyproject.toml` with `uv.lock` when
changing them through `uv add`. Virtual environments, caches, notebook
checkpoints, and packaging output are excluded from Git.

## Optional further reading

The notebooks contain their own explanations and original worked examples;
external reading is optional. For alternative treatments, see
[Stanford CS229 notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
on linear models and neural-network derivatives, and the
[Deep Learning optimization chapter](https://www.deeplearningbook.org/contents/optimization.html)
on training objectives and stochastic optimization.
