# Research practice throughout the curriculum

## Purpose and prerequisites

Practice reading evidence as soon as an algorithm can be implemented and
checked. A paper study is bounded by concepts already introduced in the unit.
Stage 02 can investigate a small estimator or loss; Stage 08 can investigate
a larger controlled claim. Reading is not a substitute for a correct local
implementation, and implementing one component is not a full reproduction.

## One study cycle

1. **Explain:** identify the problem, data assumptions, baseline, proposed change,
   and claimed improvement. Classify the change as objective, architecture,
   data, optimization, inference procedure, or systems implementation.
2. **Derive:** translate notation and recover the essential equation from
   introduced assumptions. State dimensions, reductions, and differentiability
   or support conditions. Separate an exact identity from an approximation.
3. **Implement:** choose the smallest example that exposes the mechanism. Record
   which components and infrastructure are reused and which are built manually.
4. **Verify:** use an independent objective, exact enumeration, hand calculation,
   alternate algebraic path, or derivative check. Test failure cases before
   interpreting training curves.
5. **Evaluate:** choose metrics and matched budgets before examining outcomes.
   Record all selection effort, seeds, data boundaries, work, and resources.
6. **Critique:** identify uncertainty, omitted controls, possible confounders,
   scale limits, and departures from the original setup. Negative or ambiguous
   findings are valid outcomes when supported and reported clearly.

## Reading selection

Use primary sources: original papers and official course/tool documentation. Verify bibliographic
details and public access when a unit is authored. A reading task names the
specific section/equation being studied, prerequisites, an achievable local
question, and a reason the source is relevant. External assignments are scope
references; repository exercises and worked references are original.

The standing reference roles are [CS229](https://cs229.stanford.edu/index.html-spr26)
for statistical/classical breadth, [CS231n](https://cs231n.stanford.edu/schedule)
for representation, visual, and generative connections,
[CS224N](https://web.stanford.edu/class/cs224n/) for language representations,
and [CS336](https://cs336.stanford.edu/) for language-model construction, data,
training, systems, and evaluation. Required studies do not inherit the courses'
full-scale compute requirements.

Keep an evolving reading list alongside the stable core. Record paper version,
date accessed, repository commit if used, and the local study's scope. Revisit
an earlier idea after later chapters and explain how the new context changes
its interpretation.

## Evaluation and uncertainty

Match data/splits, preprocessing, selection criteria, and declared work budgets
when comparing methods. Change one factor per ablation where possible. Distinguish
fixed-model test-sample uncertainty from training/selection variability; use
paired comparisons when predictions share observations. Small repeated runs
support only the conditions they actually investigate. Do not invent results
or silently discard failed runs.

Default CI checks reference correctness and demonstrations. Learners run their
own implementation checks and experiments on study branches. Reports identify
mathematical correctness, implementation correctness, and the empirical claim
separately. Use the [template](paper-study-template.md) for evidence.

[Return to the foundations map](foundations-and-connections.md)
