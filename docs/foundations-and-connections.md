# Foundations for following AI research

## Learning objective

Given a new AI paper, explain its problem and assumptions, derive its essential
equations, implement a bounded version, independently verify it, and critique
the experimental claim. Competence is demonstrated by evidence, not a count
of completed notebooks or familiarity with model names.

The nine stages remain the main progression. Stages 00–01 and the existing
Stage 04 chapters are retained. New paths below describe planned coverage;
the registry alone identifies published notebooks. Stage 00 stays a compact
applied review for learners who already know calculus, algebra, and programming.

## Concepts, connections, and entry points

| Foundation | Development in the path | Connection to later work |
| --- | --- | --- |
| Statistical inference | Stage 00 probability; Stage 01 evaluation; Stage 02 likelihood/MAP and statistical learning; Stage 03 Bayesian and latent-variable inference | Separate a fitted objective, predictive uncertainty, and an experimental claim |
| Information theory | Stage 02 entropy, cross-entropy, KL, mutual information, and decision risk; revisit categorical losses in Stage 04 | Understand token prediction, contrastive objectives, variational bounds, and regularized post-training |
| Geometry and linear algebra | Shapes/conditioning in Stage 00; margins/kernels in Stage 02; PCA/SVD in Stage 03; curvature in Stage 04 | Analyze representations, ill-conditioned training, low-rank updates, and computational approximations |
| Learning dynamics | Stage 01 optimization; Stage 04 initialization, curvature, optimizer state, normalization, residual gradient paths, and diagnostics | Explain convergence failures and compare training under controlled work budgets |
| Representation learning | Stage 03 reconstruction/projection; Stage 05 embeddings, contrastive/masked learning, and transfer | Analyze how supervision and invariances produce features, including paired modalities |
| Generative modeling | Stage 03 latent inference; Stage 06 common overview of autoregression, latent variables, denoising, and vector fields | Compare mathematical objectives and sampling procedures across text, image, and other modalities |
| Sequential decisions | Stage 07 essential MDP/return/policy-gradient concepts before reward/preference post-training; advanced RL in Stage 08 | Distinguish prediction, sequential control, reward optimization, search, and model-assisted tool use |
| Systems and data | Stage 05 device/training infrastructure; Stage 06 tokenizer and document boundaries; Stage 07 profiling, mixtures, scaling, memory, and inference | Identify whether a result changes an algorithm, data distribution, implementation, or compute budget |
| Scientific evaluation | Stage 01 onward: independent checks, baselines, declared selection, ablations, uncertainty, and limitations | Read benchmark improvements as bounded evidence rather than universal capability claims |

These are conceptual dependencies, not claims that every chapter of an earlier
stage is required for every later unit. For example, Stage 03 inference needs
Stage 00 probability and the Stage 02 information/decision chapter, but not a
completed random-forest implementation. The retained Stage 04 chapters remain
usable after Stage 01; new chapters introduce their additional prerequisites
explicitly.

## Deeper coverage to author

| Stage | Additional or strengthened units | Completion evidence |
| --- | --- | --- |
| 02 | Information/decision foundations before neighbors and generative classifiers; impurity as a loss; margins and finite-sample generalization | Derive losses/decision rules; implement manual algorithms; compare assumptions, computation, and uncertainty on matched data |
| 03 | Bayesian prediction and latent-variable inference; explicit ELBO/KL reasoning; SVD geometry | Compare posterior versus point prediction, verify a lower bound, and analyze representation assumptions |
| 04 | Multiclass objectives; optimization geometry and training diagnostics alongside optimizers, normalization/dropout, and learner-built autodiff | Independently check gradients/curvature, optimizer state, and train/evaluation behavior; diagnose failed training |
| 05 | Self-supervised representations alongside image/sequence models and transfer | Construct contrastive or masked targets, prevent evaluation leakage, and evaluate frozen versus adapted representations |
| 06 | Required generative-modeling overview and sequence-architecture comparisons; optional full VAE/diffusion/flow/multimodal branches | Explain each objective and sampling mechanism; train a small decoder LM; compare quality and computational constraints |
| 07 | Essential RL before preference/reward optimization; inference-time computation and search | Check a tiny policy-gradient example, distinguish SFT/preference/RL objectives, and evaluate extra inference work under matched budgets |
| 08 | Independent bounded reproductions, including planning/tool systems as an optional track | Establish baselines, verify a component, run ablations/repeats, and report the supported claim and reproduction departures |

## Authoring sequence and status

1. Publish this dependency map and the research-study convention.
2. Develop Stage 02 in ordered units, beginning with information and decision
   foundations. Each unit passes structural, numerical, fresh-kernel, rendering,
   and resource gates before a focused commit and push.
3. Develop Stages 03–04 with the additional inference, multiclass, geometry,
   and differentiation foundations. Keep all existing IDs, answers, and outputs.
4. Develop Stages 05–07, introducing PyTorch only with the authored Stage 05
   bridge. Compare small, measurable models rather than assuming frontier
   hardware or datasets.
5. Author Stage 08 as independent research pathways. A learner chooses a track;
   completing every specialization is not a prerequisite for a reproduction.

There is no fixed exercise quota. Tasks progress through prediction/derivation,
implementation, independent checks, diagnosis, transfer, and experimental
critique as the learning objective requires. Keep optional depth distinguishable
from the main route and preserve stable exercise identifiers.

## Publication and compute contract

Required experiments run on the specified Ryzen 7 6800H / 6 GiB laptop GPU.
Targets remain at most 6 GiB host working memory and 5 GiB GPU memory, short
correctness checks, about 30 minutes for ordinary experiments, and resumable
study sessions of roughly two hours. These are budgets; measurements are
published per developed unit. Large-scale paper results are reading material,
not required local reproductions.

Manual NumPy continues through Stage 04. Frameworks supply tensor/autograd/GPU
infrastructure from Stage 05; learners still implement architectures and
training loops. Dependencies are introduced only when an authored unit needs
them. Package reuse follows demonstrated project needs.

Use the [research practice](research-practice.md) and
[paper-study template](paper-study-template.md) throughout the progression.

[Return to the curriculum](../README.md)
