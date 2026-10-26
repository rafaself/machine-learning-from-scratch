# Representation study report

Replace instructions with evidence from your actual run. This template contains
no invented results and does not imply that any mastery gate is satisfied.

## Explain: question, population, geometry and supported estimand

State the fixed three-prototype compression question, seven candidate order,
training/reporting coordinates, loss normalization, baseline and tie rules.
Derive reconstruction shapes and the conditions needed for an orthogonal error
decomposition. Distinguish training SSE, retained variance, common loss,
partition agreement, density and probability of a final flag.

| Task | Seed or artifact/version/license/hash | Generating/sampling assumptions | Limitations and audit labels |
| --- | --- | --- | --- |
| Mixture | 30600 | | |
| Gaussian null | 30609 | No discrete generating label | |
| Iris | UCI artifact notice | | Duplicates/source discrepancies preserved |

| Task | Training rows/groups | Validation rows/groups | Calibration rows/groups | Final rows/groups | Group convention |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Explain which information was available for each fitting/selection/calibration
decision. Disclose any use of external labels, test feedback or changed protocol.

## Implement: reproducible evidence and checkpoints

Link the study branch/commit, completed notebook, numeric checkpoint directory,
full candidate/start records and replay instructions. Record actual hardware,
Python/NumPy/Matplotlib versions, dtype, thread counts, seeds and work budgets.
Explain `frozen`, `claimed` and `complete` transitions and the outcome of the
injected-failure self-check. State what was saved and which analyses reuse saved
arrays without new final inference. File guards do not prove an honest protocol.

## Verify: independent mathematical and procedural checks

| Contract | Independent oracle | Observed result/discrepancy | Tolerance | Conclusion |
| --- | --- | --- | --- | --- |
| Fixed-assignment center gradients and sum scaling | Separate scalar objective/finite differences | | | |
| Tiny centroid optimum and stable/budget/empty cases | Enumeration/hand calculation | | | |
| PCA/projector/reconstruction/metric cross term | Covariance algebra and scalar errors | | | |
| Train-only transforms/selection | Held-out-outlier and copied-array audit | | | |
| ARI, trivial cases and label permutation | Pair counts/fixed-margin arithmetic | | | |
| Conservative calibration ties/rank resolution | Exhaustive held-out ranks | | | |
| Conditional Beta moments | Rational posterior calculation | | | |
| Group partitions and paired resampling | Coverage/disjointness/explicit row counts | | | |
| Checkpoint identity, shapes, selection and consumption | Numeric round trip/malformed payload/failure injection | | | |

Separate mathematical claims, implementation correctness and experimental
conclusions. Record unsuccessful checks and remaining unverified items.

## Evaluate: validation and optimization before the frozen final panel

Include all seven candidate rows per task, all eight nonbaseline starts,
training-coordinate SSE traces, stable/budget/empty status, gradient diagnostic,
common validation losses and actual work. Never remove an unfinished start.

| Task/candidate | Common validation mean loss | Retained variance and its coordinates | Chosen start | Stable/total starts, updates, empty cases | Seconds |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Record the selected procedure and when its choice was frozen. Include every
restart-pair ARI plus summary/range and explain the common validation domain.
State why initialization stability was not a hidden selection criterion.

## Evaluate: one frozen final panel

| Task/frozen candidate | Common final mean loss | External ARI if available | Interpretation and unsupported alternatives |
| --- | --- | --- | --- |
| | | | |

| Task | Frozen selection | Selected-minus-mean paired loss | Complete-group percentile sensitivity | Groups/repeats/seed | Excluded uncertainty |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

| Task/selected or baseline score | Alpha, calibration size and resolution | Flags/final rows | Conditional Beta prior/posterior if justified | Exchangeability/IID limitations |
| --- | --- | --- | --- | --- |
| | | | | |

Do not select a new winner using this panel. Explain why lower null compression
does not discover generating classes, species agreement answers another
question, rank validity averages over calibration/new nominal rows, and a Beta
posterior is model-based and conditional on a realized detector. For Iris,
identify which probabilistic interpretations are only hypothetical assumptions.

## Evaluate: declared sensitivities and resources

| Task/candidate | Original/converted common validation loss | Unit-conversion ARI | Assumptions about scaling and ties |
| --- | --- | --- | --- |
| | | | |

| Task/bootstrap repeat | Sampled training indices | Original-metric validation loss | ARI versus original fit | Actual stopping/work status |
| --- | --- | --- | --- | --- |
| | | | | |

Keep every training bootstrap and unit result. These diagnostics use no new
final predictions and do not change the original selection. Distinguish
initialization randomness, empirical training resampling, fixed-model final
loss resampling and a conditional flag-probability posterior.

| Phase/configuration | Seconds | Memory value and named metric | Sizes/work/budgets and status |
| --- | --- | --- | --- |
| Selection | | | |
| Checkpoint/final evaluation | | | |
| Units and training bootstrap | | | |

Name included processes, sampling interval, shared-page convention, browser
exclusion, threads and GPU use. Approximate aggregate RSS is not an exact peak
physical-memory measurement. Record failures and interruptions.

## Primary-source study, supported claim and limitations

Translate the selected PCA and stability passages into the implemented
coordinates, perturbation protocol and finite optimizer. State exactly which
source equations were checked and what was not reproduced. Avoid interpreting
a finite diagnostic as the paper's full population/asymptotic theorem.

Write the narrow claim supported by actual evidence and at least one competing
explanation. Discuss sample/artifact assumptions, duplicate entities, units,
finite restarts, optimization error, dimension, objective mismatch, null
structure, selection noise, uncertainty and deployment shift. Propose one
bounded follow-up with new held-out data, a declared alternative explanation
and compute budget; do not invent its results.

Conclude with separate Explain / Implement / Verify / Evaluate evidence and
unresolved checks. Advancing the curriculum requires those abilities, not only
executing every cell or reading the primary sources.
