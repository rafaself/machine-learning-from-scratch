# Manual regression and classification pipelines

Complete all five Stage 01 chapters, then open [the starter](ml-pipeline.ipynb)
on your study branch. Use the [report template](report-template.md). Required
data are generated locally; no downloads, GPU, framework, or new dependency
is needed. Both pipelines use NumPy and manual gradients.

## Protocol and deliverables

- Fix the protocol before measuring performance. The supplied presets generate
  300 rows per task (seeds 70/71) and split 180/60/60 (seed 72).
- Establish hand oracles, independent weight/bias gradient checks, shape and
  stability checks, and row-duplication invariance before any test access.
- Use five folds within the training partition (seed 73). Stratify binary
  labels; each class needs at least five rows. Fit scaling independently on
  each training fold. Select lambda from [0,.01,.1,1] using unpenalized held-out
  MSE or mean log loss. Record all fold scores, not only the winning mean.
- Use 800 full-batch steps at rate .1. Inspect histories and final gradients;
  investigate convergence using training-only checks before final evaluation.
- Refit the selected procedure on training rows only. Use validation rows to
  choose a classifier threshold from [.2,.3,.4,.5,.6,.7,.8] for FP cost 1,
  FN cost 2. Ties use grid order. Freeze the model, preprocessing, and threshold.
- Evaluate once on test data against training-mean/prevalence baselines. Report
  regression MSE/RMSE/MAE and classifier log loss, Brier, confusion counts,
  accuracy, precision, recall, F1, cost, and a reliability diagram. Undefined
  ratios are NaN. Use paired-row resampling for conditional test uncertainty.

Deliver your completed notebook, protocol/configuration, full score tables,
frozen numerical arrays, and a report containing actual results. Do not refit
on validation data after threshold selection in this protocol. If test outcomes
guide a change, disclose that the old test has become selection data.

## Self-check and mastery gate

- **Explain:** derive both mean objectives, unpenalized bias, L2 scaling,
  baselines, metric definitions, and the data-use boundaries.
- **Implement:** train both models and construct fold-local transforms and
  evaluation procedures manually.
- **Verify:** use independent objectives at nonzero parameters, n=1/n>1,
  extreme logits, malformed shapes, repeated rows, and held-out-outlier
  transform checks. A decreasing training objective is insufficient.
- **Evaluate:** compare fixed baselines and chosen models with a reproducible
  protocol, appropriate uncertainty, convergence diagnostics, resource
  measurements, and a conclusion limited to the actual experiment.

## Checkpoints and compute

Use separate resumable study sessions: (1) protocol and hand calculations;
(2) implemented/verified functions; (3) CV, validation decisions, and frozen
configuration; (4) final evaluation and report. Commit each checkpoint on your
study branch. Keep seed/version/work-budget records with the arrays. The
reference saves a numeric NPZ checkpoint (no pickle); reload with
`np.load(path, allow_pickle=False)` and regenerate the deterministic splits.

Two hours is a suggested maximum study session, not a required runtime.
The default numerical experiments are small CPU workloads. The Stage 01 guide
publishes measured reference time/memory; your implementation must be measured
separately. Default CI executes demonstrations and independent reference checks,
never unfinished learner answers or an automatic grade of model performance.
