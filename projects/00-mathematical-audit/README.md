# Mathematical audit project

**Prerequisites:** the Stage 00 diagnostics and implementation checks.
Open [the starter notebook](mathematical-audit.ipynb) on a study branch.
No external datasets, GPU, or additional dependencies are required.

## Deliverables

1. A model/notation and audit protocol, written before running experiments.
2. Gaussian summaries with both covariance conventions, a stable mean NLL,
   and a fixed-covariance mean gradient.
3. Independent hand-value and finite-difference checks, shape/domain failures,
   and a comparison of precision and conditioning.
4. A [report](report-template.md) with actual measurements and limitations.

The supplied demonstration generates 512 training and 256 held-out rows with
three features, seeds 2026 and 2027. Fit summaries only on training rows.
Keep the held-out assessment separate from implementation verification.

## Self-check and mastery gate

- **Explain:** derive the objective and mean derivative; state iid, Gaussian,
  positive-definiteness, and covariance-denominator assumptions.
- **Implement:** use centered statistics, Cholesky, and linear solves, preserving
  input arrays and the (n,d)/(d,)/(d,d) shape contracts.
- **Verify:** reproduce the notebook's four-row oracle, compare covariance with
  an independent calculation, and check nonzero gradients at multiple means.
  Reject nonsymmetric, singular, and indefinite covariance deliberately.
- **Evaluate:** report held-out NLL, precision/conditioning sensitivity, and
  cases where correct numerical code does not justify the statistical model.

Do not certify a gradient solely at the fitted mean. Do not add covariance
jitter or change the objective without documenting the change. A numerical
check and a supported experimental conclusion are separate requirements.

## Working sessions and compute

Suggested checkpoints: commit the protocol and hand oracle; commit the tested
implementations; save measurements and finish the report. Each session can
stop independently and resume from recorded seeds/configuration. Two hours
is a study-session budget, not a required training duration.

Required data are tiny CPU arrays. The published validation measurements in
the Stage 00 guide cover the author references, not unfinished learner code.
Default CI executes data/plot demonstrations and checks references; it does
not grade your answer cells. Run your own self-checks on your study branch.
