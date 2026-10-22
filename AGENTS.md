# Educational editing agreements

## Preserve the learning material

- Keep learner answer cells blank in the reference curriculum. Provide hints
  and worked reference solutions only in separate, collapsed Markdown cells
  immediately after each learner answer. Render reference code as fenced text;
  do not execute it during Run All or move it into the package.
- Preserve student answers, outputs, and unrelated user work. Inspect the current
  branch and changes before editing. Students complete exercises on separate
  study branches; their completed answer cells are not merged into the reference.
  Author-provided collapsed reference solutions are part of the curriculum.
- Keep explanations substantial and self-contained. Define notation before use,
  justify derivation steps, give intuition and small worked examples, and state
  assumptions and limitations.
- Introduce concepts before requiring them. Use the README study order and all
  nine chapter sections consistently.

## Exercise contracts

- Number exercises sequentially within a chapter and keep their IDs stable
  unless a deliberate curriculum revision requires renumbering.
- Every exercise prompt must be followed immediately by its dedicated answer
  cell, with matching IDs in the existing curriculum metadata.
- Written, numerical, mathematical, and critical questions use editable Markdown
  answer cells. Programming and experimental tasks use dedicated code cells.
- State the task and expected outcome. Follow the learner answer with a
  collapsed hint, then a separately collapsed worked solution. Match all four
  cells by exercise ID; no open attribute should expose reference content by
  default. Code-answer placeholders contain comments only.
- Reference solutions explain reasoning. Open-ended tasks provide suggested
  responses and alternatives; experimental references do not invent results.
  Clearly name dependencies on earlier reference snippets in the same chapter.
- Demonstrations run independently of unfinished answers and do not supply
  completed implementations of the challenges.
- Distinguish explanation, demonstration, exercise, answer, hint, and solution
  in cell metadata. Tests must allow students to fill their answer cells.

## Mathematics and implementation

- Use row-wise examples, shape (n, d). The first three chapters use (n,) targets;
  the network chapters explicitly use (n, 1) targets/probabilities.
- Network weights have shape (previous_width, next_width) and biases are stored
  as one-dimensional arrays broadcast over observations.
- Define whether a loss is half-MSE, MSE, average cross-entropy, or a regularized
  objective. Put the observation average in the network output sensitivity once;
  do not average again during downstream matrix products or bias sums.
- Derive and differentiate at one parameter snapshot. Separate backpropagation
  from the optimizer's parameter updates.
- Use Python, NumPy, and Matplotlib. Avoid ML frameworks, automatic
  differentiation, premature abstractions, and unnecessary dependencies.
- Record seeds, splits, preprocessing, selection criteria, and work budgets.
  Keep held-out test observations out of fitting and repeated model selection.
- Do not configure notebook output clearing: no stripping filters, hooks, or
  automatic clearing steps. Preserve existing user outputs.

## Validation and scope

- Run uv sync --locked and uv run pytest for curriculum changes.
- Validate notebook JSON/metadata, adjacent answer-cell types, unsolved reference
  exercise cells, executable demonstrations, and rendered Markdown/LaTeX.
- Review formulas, numerical examples, array shapes, averaging, and sequencing.
  Passing structural tests does not establish mathematical or pedagogical quality.
- Keep the existing minimal environment. Ask before adding a production
  dependency unless it is highly consolidated, safe, and trusted.
- Do not change the GitHub remote, push, or merge without user authorization.
