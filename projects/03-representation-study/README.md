# Representation study

Complete the five [Stage 03 chapters](../../notebooks/03-unsupervised-learning/README.md),
Stage 00 numerical/probability tools and Stage 01 evaluation. Open the
[starter notebook](representation-study.ipynb) on a study branch and copy the
[report template](report-template.md). Implement with NumPy and Matplotlib;
only the existing offline Iris loader is reused from the package. Required work
has no downloads, GPU, cloud service, new framework or additional dependency.

## Protocol and deliverables

The question is how representation and finite initialization affect a fixed
three-prototype compression task. Cluster count is a declared engineering
budget, not an estimated number of semantic categories. Compare these seven
procedures in this order: training mean, raw identity, raw PCA-1, raw PCA-2,
standardized identity, standardized PCA-1 and standardized PCA-2.

- Use the declared 360-row/four-feature mixture (seed 30600), 360-row Gaussian
  null (seed 30609) and unchanged 150-row offline Iris artifact. The null has
  no discrete generating label. Preserve Iris discrepancies and duplicate rows;
  its [notice](../../src/ml_from_scratch/data/NOTICE.md) supplies attribution,
  license, source/version and hash.
- Group identical feature rows by first occurrence, shuffle groups with seed 30601, and allocate
  approximately 50/20/15/15 percent to training, validation, calibration and
  final evaluation. Record actual counts. For distinct synthetic rows the IDs
  are row positions, so fixed seeds do not allocate predetermined feature ranks.
  Do not stratify or select using
  synthetic component labels or Iris species. Grouping identical rows does not
  establish independence of all remaining entities.
- Fit every coordinate mean, scale and PCA map on training only. Keep one
  training-fitted common reporting scale across all candidates. Primary loss
  is full-feature squared reconstruction error divided by the common squared
  scales, averaged once over features and then observations. Latent SSE and
  retained variance have different units/objectives and are not selector scores.
- Each nonbaseline procedure gets four random distinct-row starts and four
  D-squared starts, common seed positions derived from 30602, and at most 100
  Lloyd updates per start. Use computed first-index ties and keep/report empty
  centers. Select the first minimum training SSE within each representation,
  then the first minimum common validation loss across procedures. Preserve all
  144 starts across three tasks, including unmet stopping targets. Equal
  budgets do not prove equal optimization quality.
- Report every candidate's timing/loss and all 28 within-candidate restart
  pair ARIs on common validation rows. Stability does not select a new procedure.
  Freeze every model and the selected index before accessing calibration/final
  outcomes. The checkpoint retains all fitted arrays, starts, losses and splits.
- Calibrate reconstruction loss, not a density, at alpha .1 on separate rows.
  Evaluate all seven frozen procedures in one final panel; report the frozen
  selection without changing it after the panel. External labels enter only
  its descriptive ARI audit. Report selected/baseline nominal flag counts and
  the precise assumptions of any conformal or conditional Beta interpretation.
  Iris's convenience sampling and duplicates do not establish exchangeability
  or an IID Bernoulli model.
- Compute selected-minus-mean paired loss sensitivity using 400 complete-group
  resamples (seed 30605) of saved final losses. The percentile range conditions
  on fitted models/observed groups; it is not guaranteed population coverage or
  training/selection/calibration uncertainty.
- Perform the declared unit conversion and four training bootstraps (seed
  30606) using training/validation only. Retain all results and compare bootstrap
  reconstructions in the original common metric. These diagnostics do not
  replace the frozen primary decision or permit another final inference.

Submit a protocol, completed notebook, independent mathematical/procedural
checks, full score/start/sensitivity records, numeric checkpoints, actual resource
measurements and a technical report. Keep unsupported findings and unsuccessful
checks visible. Do not infer semantic categories from a null quantization or a
universal representation ranking from three small tasks.

## Self-check and mastery gate

- **Explain:** derive coordinate/reconstruction shapes, metric dependence,
  orthogonality conditions, sum versus mean scaling, fixed-assignment center
  gradients, hard versus marginal mixture likelihood, and the different
  probability spaces behind stability, rank calibration and Beta summaries.
- **Implement:** manual PCA/centroid routines, honest label-free group splits,
  complete bounded initialization/selection, numeric artifact round trips,
  fail-closed final-access bookkeeping and analysis of saved final arrays.
- **Verify:** independent scalar finite differences and hand optima, covariance
  projectors/full reconstruction, loss units, ARI pair arithmetic, exact tied
  ranks, rational Beta moments, group-resampling counts, held-out isolation,
  malformed/checkpoint identity checks and persisted consumption after an
  injected evaluation failure. A decreasing objective alone is insufficient.
- **Evaluate:** common held-out error and complete work/status records, null and
  external-label counterinterpretations, separate initialization/data/metric
  sensitivity, correctly conditioned uncertainty and the supported claim's
  limits. A reading checklist or passing structural test is not personal mastery.

Default CI executes demonstrations independently of learner answers, checks
notebook structure and separately validates displayed author definitions against
independent small oracles. It does not execute unfinished learner code, assign
an exercise grade or certify the learner's experimental conclusions.

## Checkpoints and compute

Use resumable sessions for (1) protocol/oracles, (2) verified implementations,
(3) selection and frozen checkpoint, and (4) final evaluation/reporting.
Roughly two hours is a study-session target, not required program runtime.

NPZ files contain numeric fitted/loss/split/result arrays plus JSON metadata;
load with `allow_pickle=False`. Never serialize model dictionaries with pickle.
Fresh saving refuses to overwrite a prior checkpoint. Resumption checks data
identity and versions and uses saved fitted arrays without new optimization.
Final evaluation atomically records `claimed` on disk before model calls, then
stores complete evidence and `complete` after success. Another final call from
either consumed phase is refused. If interrupted, preserve the claim and
disclose any further access. File bookkeeping cannot prevent deliberate copying
or retrospective protocol changes; it supports honest study practice.

The reference uses a unique explicit run directory and prints its path. Record
that path and resume it instead of silently starting over after final feedback.
All final metrics and uncertainty analysis can reuse saved arrays without new
predictions. Preserve the original reference notebook's blank answers and outputs.

The [stage guide](../../notebooks/03-unsupervised-learning/README.md) publishes
measured presets. Default selection uses 21 candidates / 144 bounded starts;
the units study repeats that selection, and four bootstrap refits per task
add at most 96 starts when the selected procedures are nonbaseline. Required
working sets target at most 6 GiB host memory with zero GPU allocation. Generic
helpers cap 2,000 rows, 16 features, eight centers, 200 updates, query blocks of
256, eight bootstrap refits and 800 paired-loss resamples. Do not expand these
budgets without a separate declared study and new resource measurements.
