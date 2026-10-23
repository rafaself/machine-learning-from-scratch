# Classical learning benchmark

Complete the seven Stage 02 chapters and Stage 01 evaluation, then open
[the starter](classical-benchmark.ipynb) on your study branch. Copy the
[report template](report-template.md). Required data are generated locally or
bundled under an attributed license. Required models are manual NumPy; no
download, cloud service, GPU, framework or new dependency is needed.

## Protocol and deliverables

- Use synthetic regression (180 rows, seed 240), balanced radial binary
  classification (160 rows, seed 241) and the bundled UCI Iris artifact.
  Preserve its source discrepancies and duplicate records. The
  [data notice](../../src/ml_from_scratch/data/NOTICE.md) records attribution,
  license and exact-version provenance.
- Group identical feature rows before splitting. Seed 242 allocates whole
  groups approximately 60/20/20 to training/validation/test, stratified by class
  for classification. Synthetic counts are 108/36/36 and 96/32/32; record actual
  Iris counts because unequal groups change row proportions.
- Use three common group-preserving folds within training only (seed 243).
  Every non-baseline family gets two candidates. Fit scaling on each fold's
  training rows; pool held-out row losses with the actual fold counts. Ties use
  candidate order. Keep every fold's score, time and solver diagnostics.
- Regression families: training mean, ridge, neighbors, tree, forest, boosting.
  Classification families: prevalence, softmax, neighbors, shared/diagonal
  Gaussian, tree, forest, one-versus-rest boosting and kernel SVM.
- Refit each family's CV winner on complete training rows only. Use validation
  error to select the champion, with baseline/family order as the tie rule.
  Freeze all per-family choices and the champion. Do not refit on validation.
- Evaluate every frozen family and baseline in one final test batch. The test
  panel does not select another winner. Primary metrics are MSE/error; secondary
  metrics include MAE/RMSE, confusion/macro recall and probability losses only
  for genuine probability models. The declared 5% uniform mixture for frequency
  forecasts changes their probability estimator explicitly; it is not a hidden
  clipping rule or a calibration guarantee.
- Report champion-minus-baseline paired primary loss using 400 complete-group
  resamples (seed 245). Retain all rows in each sampled group and recompute the
  pooled row mean. The interval is conditional resampling sensitivity, not
  guaranteed population coverage or uncertainty from new training/selection.

The notebook states the exact grids and work limits: 600 softmax steps, eight
depth-four forest trees, eight depth-two boosting stages and at most 1500
projected SVM steps per binary problem. Convert SVM mean penalty to canonical
`C` with the actual fitting-row count. The softmax gradient-norm diagnostic
target is 1e-4 after the fixed budget; it does not alter the iteration count.
Report unmet gradient/gap targets without
dropping candidates or secretly increasing work. Equal candidate counts control
selection opportunities; they do not make wall time or optimization equal.

Submit your completed notebook, independent correctness evidence, full score
records, selection/final numeric checkpoints, version/seed/workload record and a
report of actual results. Disclose test reuse and unsuccessful checks. Do not
infer a universal ranking from these small tasks.

## Self-check and mastery gate

- **Explain:** derive baseline/linear objectives, mean versus sum conventions,
  unpenalized biases, class-column alignment, group weighting, SVM constraints
  and the distinction between raw scores and probabilities.
- **Implement:** fit every required manual family, build fold-local transforms,
  preserve group boundaries, select on permitted data and save numeric evidence.
- **Verify:** use independent hand/gradient checks, exact small optima, scalar
  neighbor/likelihood checks, split/OOB audits, duplicate/split coverage, a
  held-out-outlier scaler test, pooled CV arithmetic, checkpoint round trips and
  an accidental-second-test-call check. A decreasing objective is insufficient.
- **Evaluate:** retain all candidates' errors, work budgets and diagnostics;
  freeze the validation decision, evaluate once, use the correct sampling unit,
  measure resources and state the actual supported claim and limitations.

Default CI executes the starter's demonstrations, structural contracts and
independent author-reference checks. It does not run unfinished learner
implementations, grade your model scores or declare personal mastery.

## Checkpoints and compute

Use resumable study sessions: (1) protocol, provenance and mathematical oracles;
(2) verified implementations and procedural isolation; (3) CV, validation and
frozen selection; (4) final evaluation, paired uncertainty and report. Commit
each checkpoint on your study branch. Roughly two hours is a study-session
target, not a required program runtime.

The reference saves split/fold/score arrays and JSON protocol text in numeric
NPZ checkpoints. Load with `np.load(path, allow_pickle=False)`; never serialize
model dictionaries with pickle. Resumption checks the canonical data digest,
then reconstructs models from the frozen training configuration without repeating
CV or validation selection. Completed final evaluation updates the checkpoint's
used flag; reuse its saved losses and predictions for analysis. If interrupted
during final evaluation, treat the test as used and disclose further access.

The [stage guide](../../notebooks/02-classical-ml/README.md) publishes measured
reference workload/time/memory. Measure your own implementation separately.
Required working sets target at most 6 GiB host memory; no GPU allocation or
multi-process/distributed work is required. Dense kernel training remains below
the teaching solver's 128-row limit. Do not expand grids or stage counts without
declaring a separate bounded study and preserving the original comparison.
