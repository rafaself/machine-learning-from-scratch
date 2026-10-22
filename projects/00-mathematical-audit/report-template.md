# Mathematical audit report

Replace prompts with your own evidence. Empty fields are not passing results.

## Explain: model and derivation

State the data-generating assumptions, dimensions, objective reduction, both
covariance conventions, derivative, and fixed parameters during differentiation.

## Implement: artifacts and reproducibility

Record your study-branch commit, Python/NumPy versions, hardware, dtype, seeds,
data order, and exact replay steps. List estimator/likelihood/checker functions.

## Verify: independent evidence

| Check | Independently expected result | Observed result | Tolerance | Pass/failure and interpretation |
| --- | --- | --- | --- | --- |
| Four-row summary and NLL | Derive before calling implementation | | | |
| Mean gradient at two non-optimal points | Separate objective and central differences | | | |
| Covariance denominators and d=1 shape | Independent centered calculation | | | |
| Invalid covariance and input mutation | Explicit failure/no mutation | | | |

## Evaluate: measured experiments

Report train/held-out NLL and what that comparison supports. Compare float32
and float64 with the same intended inputs. Include condition numbers, residuals
or forward errors where applicable, and all intentional/unexpected failures.

| Configuration | Elapsed seconds | Memory value and metric | Numerical result | Interpretation |
| --- | --- | --- | --- | --- |
| Standard audit | | | | |
| Precision/conditioning investigation | | | | |

## Limitations and conclusion

Discuss model misspecification, dependence, small samples, conditioning,
finite-difference limitations, and held-out uncertainty. Identify which of
the four mastery requirements your evidence establishes and what remains open.
