# Classical benchmark report

Replace instructions with evidence from your own study run. This blank template
does not contain invented scores or imply that any mastery gate has passed.

## Explain: protocol and supported question

State the question, target risks, representation assumptions, objective formulas,
penalty/loss normalization, bias treatment, class/score/probability conventions,
candidate/tie rules, optimizer budgets and the exact final-test boundary.

| Dataset | Artifact/seed | Feature/target/group assumptions | Train/validation/test rows and groups | Limitations |
| --- | --- | --- | --- | --- |
| Synthetic regression | seed 240 | | | |
| Radial binary | seed 241 | | | |
| Iris | UCI DOI/hash/license | unchanged discrepancies and duplicate handling | | |

Explain why equal candidates and common splits do not imply equal compute,
capacity or optimization quality. Define the primary paired comparison before
test use. Cite the source artifact and the bounded theoretical argument used.

## Implement: reproducible artifacts and checkpoints

Link your study commit, completed notebook/models, fixed configuration,
selection/final NPZ checkpoints and replay instructions. Record Python/NumPy
versions, hardware, dtype, thread counts, seeds and any declared protocol changes.
Check that no object/pickle payload is required. Reuse saved final predictions
for arithmetic and analysis; disclose any repeated test inference or feedback.

## Verify: mathematical and procedural correctness

| Contract | Independent oracle | Observed result/discrepancy | Tolerance | Conclusion |
| --- | --- | --- | --- | --- |
| Ridge/softmax weights and biases | Separate objective/finite differences | | | |
| Loss reduction/penalty/replicated rows | Explicit normalization calculation | | | |
| Neighbors/Gaussians | Scalar distances/densities and tiny hand values | | | |
| Trees/forests/boosting | Split/OOB/index/state checks | | | |
| SVM kernel/projection/optimum | Feature map, constraints and tiny dual oracle | | | |
| Group partitions/folds | Exhaustive coverage/disjointness | | | |
| Scaling isolation | Held-out-outlier check and fitting-row audit | | | |
| CV selection | Pooled loss sums/counts and deterministic ties | | | |
| Checkpoint/final boundary | Numeric round trip and second-call rejection | | | |

Passing structural or numerical checks does not establish predictive quality.
State mathematical correctness, implementation correctness and experimental
conclusions separately. Retain failures and unresolved checks.

## Evaluate: selection evidence before final results

Include every candidate's three fold sizes/loss sums, pooled score, time and
diagnostics. Do not replace fold statuses with the final refit's status.

| Dataset/family | Two candidates and CV scores | Selected candidate | Validation score | Actual diagnostic/work status |
| --- | --- | --- | --- | --- |
| | | | | |

Record the frozen validation champion and baseline/family tie rule. SVM gaps,
KKT residuals and gradient norms are not interchangeable with error rate. Do not
describe budget-exhausted models as objective optima or silently exclude them.

## Evaluate: one frozen test panel and conditional comparison

| Dataset/frozen family | Primary test metric | Secondary metrics available | Probability convention | Interpretation |
| --- | --- | --- | --- | --- |
| | | | | |

Supply confusion counts and macro recall for classifiers. Mark probability
metrics unavailable for raw-score models. Document any zero forecasts or
numerical underflow and preserve stable log probabilities when supplied.

| Dataset | Validation-selected champion | Paired champion-minus-baseline mean | Group sensitivity interval | Groups/repeats/seed | Excluded uncertainty |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Explain the row-weighted estimand, complete-group resampling and fixed-model
conditioning. State that these intervals do not include selection/training
uncertainty or establish a universal population ranking. Do not choose a new
champion after viewing this panel.

## Evaluate: measured resources

| Workload/configuration | Seconds | Memory value and named metric | Work performed and status |
| --- | --- | --- | --- |
| Training-only CV | | | |
| Refit and validation selection | | | |
| Frozen final prediction and resampling | | | |

State which processes are included, sampling interval, array sizes, kernel/query
blocks, thread count and hardware. Approximate aggregate RSS is not exact peak
physical memory. Record unsuccessful or interrupted runs as well.

## Supported claim and limitations

Write the narrow claim supported by actual evidence and give competing
explanations. Discuss artifact quirks, group/iid assumptions, sample size, model
misspecification, uncertainty, selection effort, numerical/optimization error,
hardware dependence and distribution shift. Disclose any test reuse and say
which evidence would be needed for a stronger claim or another study.

Conclude with separate Explain / Implement / Verify / Evaluate mastery evidence
and the remaining unverified items. No reading checklist substitutes for it.
