"""Check displayed implementations and worked examples against independent math."""

import ast
import copy
import math
import re
from functools import lru_cache

import nbformat
import numpy as np
import pytest
from curriculum_support import ROOT, notebook_path



@lru_cache
def reference_functions(name):
    """Load definitions only; never run experiments or learner answer cells."""
    path = notebook_path(name)
    notebook = nbformat.read(path, as_version=4)
    namespace = {"np": np}
    for cell in notebook.cells:
        if cell.metadata.get("curriculum", {}).get("role") != "solution":
            continue
        for source in re.findall(r"(?ms)^~~~python[ \t]*\n(.*?)^~~~[ \t]*$", cell.source):
            tree = ast.parse(source)
            tree.body = [
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            ]
            exec(compile(tree, f"{path.name}:{cell.id}", "exec"), namespace)
    return namespace


def central_difference(objective, value, step=1e-5):
    """Perturb a private array and restore it even if objective evaluation fails."""
    value = np.array(value, dtype=float, copy=True)
    gradient = np.empty_like(value)
    for coordinate in np.ndindex(value.shape):
        original = value[coordinate]
        try:
            value[coordinate] = original + step
            plus = objective(value)
            value[coordinate] = original - step
            minus = objective(value)
            gradient[coordinate] = (plus - minus) / (2 * step)
        finally:
            value[coordinate] = original
    return gradient


@pytest.mark.parametrize("batch_size", [1, 5])
def test_linear_gradients_against_independent_half_mse(batch_size):
    functions = reference_functions("linear-regression")
    X = np.linspace(-0.8, 1.1, batch_size * 2).reshape(batch_size, 2)
    y = np.linspace(0.2, 0.9, batch_size)
    theta = np.array([0.3, -0.4, 0.2])

    def objective(candidate):
        residual = X @ candidate[:2] + candidate[2] - y
        return float(np.sum(residual**2) / (2 * batch_size))

    loss, dw, db = functions["linear_loss_and_gradient"](X, y, theta[:2], theta[2])
    np.testing.assert_allclose(loss, objective(theta), rtol=1e-12)
    np.testing.assert_allclose(
        np.r_[dw, db], central_difference(objective, theta), rtol=1e-5, atol=1e-8
    )
    assert dw.shape == (2,) and np.ndim(db) == 0


def test_linear_worked_values_and_closed_form():
    functions = reference_functions("linear-regression")
    predictions = functions["predict_linear"]([[2, 1], [-1, 4], [0, 2]], [1, 2], -1)
    np.testing.assert_array_equal(predictions, [3, 6, 3])
    loss, _, _ = functions["linear_loss_and_gradient"](
        [[-1], [0], [2]], [0, 1, 4], [1], 1
    )
    assert loss == pytest.approx(1 / 6)
    loss, dw, db = functions["linear_loss_and_gradient"](
        [[1, 0], [0, 2]], [2, 1], [0, 1], 0
    )
    np.testing.assert_allclose([loss, *dw, db], [1.25, -1, 1, -0.5])
    np.testing.assert_allclose(
        functions["fit_one_feature"]([0, 1, 2], [1, 2, 2]), [0.5, 7 / 6]
    )
    with pytest.raises(ValueError):
        functions["fit_one_feature"]([2, 2], [1, 3])


def test_quadratic_optimization_states_follow_exact_recurrence():
    descent = reference_functions("gradient-descent")["batch_descent"]
    curvature, optimum, initial, rate = 4.0, -1.0, 2.0, 0.1
    result = descent(
        lambda theta: (
            float(curvature * np.sum((theta - optimum)**2) / 2),
            curvature * (theta - optimum),
        ),
        [initial], rate, 8,
    )
    expected = optimum + (initial - optimum) * (1 - rate * curvature) ** np.arange(9)
    np.testing.assert_allclose(result["states"][:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(result["losses"], curvature * (expected - optimum)**2 / 2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("penalty", [0.0, 0.13])
def test_logistic_gradients_against_independent_mean_loss(batch_size, penalty):
    functions = reference_functions("logistic-regression")
    X = np.linspace(-0.7, 0.9, batch_size * 2).reshape(batch_size, 2)
    y = (np.arange(batch_size) % 2).astype(float)
    theta = np.array([0.2, -0.3, 0.4])

    def objective(candidate):
        z = X @ candidate[:2] + candidate[2]
        return float(np.mean(np.logaddexp(0, z) - y * z)
                     + penalty * np.sum(candidate[:2]**2) / 2)

    loss, dw, db = functions["logistic_loss_and_gradient"](
        X, y, theta[:2], theta[2], penalty
    )
    np.testing.assert_allclose(loss, objective(theta), rtol=1e-12)
    np.testing.assert_allclose(
        np.r_[dw, db], central_difference(objective, theta), rtol=1e-5, atol=1e-8
    )


def test_logistic_worked_losses_and_extreme_logits():
    functions = reference_functions("logistic-regression")
    loss_function = functions["logistic_loss_and_gradient"]
    for probabilities, expected in (
        ([0.6, 0.3, 0.9], 0.3242870277875165),
        ([0.6, 0.3, 0.01], 1.8242235845642717),
    ):
        p = np.array(probabilities)
        logits = np.log(p / (1 - p))
        loss, _, _ = loss_function(np.eye(3), [1, 0, 1], logits, 0)
        assert loss == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(functions["stable_sigmoid"]([-1000, 0, 1000]), [0, .5, 1])
    loss, dw, db = loss_function([[1], [-1]], [0, 1], [1000], 0)
    assert loss == pytest.approx(1000)
    assert np.isfinite(dw).all() and np.isfinite(db)


def test_perceptron_worked_update_and_boundary_tie():
    update = reference_functions("perceptron")["perceptron_pass"]
    w, b, updates = update(
        [[2, 1], [1, 2]], [1, -1], [0, 0], 0, [0, 1, 0]
    )
    np.testing.assert_array_equal(w, [1, -1])
    assert b == 0 and updates == 2
    w, b, updates = update([[0, 0]], [-1], [0, 0], 0, [0])
    np.testing.assert_array_equal(w, [0, 0])
    assert b == -1 and updates == 1


@pytest.mark.parametrize("batch_size", [1, 4])
def test_backpropagation_all_parameters_and_mean_loss(batch_size):
    functions = reference_functions("backpropagation")
    X = np.linspace(-0.5, .7, batch_size * 2).reshape(batch_size, 2)
    Y = (np.arange(batch_size) % 2).reshape(-1, 1).astype(float)
    parameters = {
        "W1": np.array([[.2, -.3], [.4, .1]]), "b1": np.array([.05, -.1]),
        "W2": np.array([[.3], [-.2]]), "b2": np.array([.1]),
    }
    before = copy.deepcopy(parameters)

    def objective(state):
        hidden = np.tanh(X @ state["W1"] + state["b1"])
        logits = hidden @ state["W2"] + state["b2"]
        return float(np.sum(np.logaddexp(0, logits) - Y * logits) / batch_size)

    _, cache = functions["network_forward"](X, parameters)
    gradients = functions["network_backward"](cache, Y)
    assert functions["network_loss"](parameters, X, Y) == pytest.approx(objective(parameters))
    for name, value in parameters.items():
        numeric = central_difference(lambda candidate: objective({**parameters, name: candidate}), value)
        assert gradients[name].shape == value.shape
        np.testing.assert_allclose(gradients[name], numeric, rtol=1e-5, atol=1e-8)
        np.testing.assert_array_equal(parameters[name], before[name])


def test_scalar_network_independently_calculated_values():
    functions = reference_functions("backpropagation")
    parameters = {
        "W1": np.array([[1.]]), "b1": np.array([0.]),
        "W2": np.array([[2.]]), "b2": np.array([-1.]),
    }
    P, cache = functions["network_forward"]([[1.]], parameters)
    gradients = functions["network_backward"](cache, [[0.]])
    np.testing.assert_allclose(P, [[0.6278929975347187]], atol=1e-12)
    assert functions["network_loss"](parameters, [[1.]], [[0.]]) == pytest.approx(
        0.9885738250340395, abs=1e-12
    )
    for name, expected in {
        "W1": .5273978964874017, "b1": .5273978964874017,
        "W2": .47819963748798927, "b2": .6278929975347187,
    }.items():
        np.testing.assert_allclose(gradients[name], expected, atol=1e-12)


@pytest.mark.parametrize("hidden", [(3,), (3, 2)])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("penalty", [0.0, .07])
def test_mlp_all_parameters_against_independent_objective(
    hidden, activation, batch_size, penalty
):
    functions = reference_functions("multilayer-perceptron")
    X = np.linspace(-.4, .6, batch_size * 2).reshape(batch_size, 2)
    Y = (np.arange(batch_size) % 2).reshape(-1, 1).astype(float)
    parameters = functions["initialize_mlp"](2, hidden, activation, seed=71)
    for W in parameters["weights"]:
        W *= .2
    # Keep ReLU coordinates on both active and inactive smooth branches.
    for b in parameters["biases"][:-1]:
        b[:] = .8 * (-1.) ** np.arange(len(b))
    parameters["biases"][-1][:] = .1
    before = copy.deepcopy(parameters)

    def objective(state):
        A = X
        for W, b in zip(state["weights"][:-1], state["biases"][:-1]):
            Z = A @ W + b
            A = np.tanh(Z) if activation == "tanh" else np.maximum(Z, 0)
        logits = A @ state["weights"][-1] + state["biases"][-1]
        data = np.sum(np.logaddexp(0, logits) - Y * logits) / batch_size
        return float(data + penalty / 2 * sum(np.sum(W**2) for W in state["weights"]))

    _, P, cache = functions["mlp_forward"](X, parameters)
    gradients = functions["mlp_backward"](cache, Y, penalty)
    assert P.shape == (batch_size, 1)
    assert functions["mlp_objective"](parameters, X, Y, penalty)[1] == pytest.approx(
        objective(parameters), abs=1e-12
    )
    for key in ("weights", "biases"):
        for layer, value in enumerate(parameters[key]):
            def changed_objective(candidate):
                state = copy.deepcopy(parameters)
                state[key][layer] = candidate
                return objective(state)

            numeric = central_difference(changed_objective, value)
            assert gradients[key][layer].shape == value.shape
            np.testing.assert_allclose(
                gradients[key][layer], numeric, rtol=1e-5, atol=1e-8
            )
            np.testing.assert_array_equal(value, before[key][layer])


@pytest.mark.parametrize("name", ["backpropagation", "multilayer-perceptron"])
def test_network_labels_cannot_broadcast_or_contain_nonbinary_values(name):
    functions = reference_functions(name)
    X = np.array([[.2, -.3], [.4, .5]])
    if name == "backpropagation":
        parameters = {
            "W1": np.ones((2, 2)), "b1": np.zeros(2),
            "W2": np.ones((2, 1)), "b2": np.zeros(1),
        }
        objective = functions["network_loss"]
    else:
        parameters = functions["initialize_mlp"](2, (2,), seed=5)
        objective = functions["mlp_objective"]
    for labels in ([0, 1], [[0], [2]], [[0], [np.nan]]):
        with pytest.raises(ValueError):
            objective(parameters, X, labels)


@pytest.fixture
def demonstration_values(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    def load(name):
        path = notebook_path(name)
        namespace = {}
        for cell in nbformat.read(path, as_version=4).cells:
            if cell.metadata.get("curriculum", {}).get("role") == "demonstration":
                exec(compile(cell.source, f"{path.name}:{cell.id}", "exec"), namespace)
        return namespace

    yield load
    plt.close("all")


def test_linear_demonstration_residuals_and_split(demonstration_values):
    values = demonstration_values("linear-regression")
    residuals = values["demo_reference"] - np.array([1., 2., 2.])
    np.testing.assert_allclose(residuals, [.5, -.5, -.5], atol=1e-12)
    assert np.mean(residuals**2) / 2 == pytest.approx(.125)
    train, test = values["demo_train"], values["demo_test"]
    assert len(train) == 35 and len(test) == 15
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(50))


def test_gradient_demonstration_sampling_and_feature_scale(demonstration_values):
    values = demonstration_values("gradient-descent")
    assert np.mean(values["demo_gradients"]) == 0
    assert np.var(values["demo_gradients"]) == 5
    train, validation = values["demo_train_ids"], values["demo_validation_ids"]
    np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(90))
    scales = values["demo_X"][train].std(axis=0)
    assert scales[1] > 100 * scales[0]


def test_logistic_demonstration_probability_and_loss_curves(demonstration_values):
    values = demonstration_values("logistic-regression")
    z, p, axes = values["demo_z"], values["demo_p"], values["axes"]
    np.testing.assert_allclose(
        axes[0].lines[0].get_ydata(), np.exp(-np.logaddexp(0, -z)), rtol=1e-12
    )
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), -np.log(p), rtol=1e-12)
    np.testing.assert_allclose(axes[1].lines[1].get_ydata(), -np.log1p(-p), rtol=1e-12)
    assert set(np.unique(values["demo_y"])) <= {0, 1}
    all_ids = np.r_[
        values["demo_train_ids"], values["demo_validation_ids"], values["demo_test_ids"]
    ]
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(180))


def test_perceptron_demonstration_margin_and_labels(demonstration_values):
    values = demonstration_values("perceptron")
    X, y = values["demo_X"], values["demo_y"]
    score = X[:, 0] - .6 * X[:, 1] + .2
    assert X.shape == (80, 2) and y.shape == (80,)
    assert np.all(y * score > .25)
    all_ids = np.r_[values["demo_train_ids"], values["demo_test_ids"]]
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(80))


def test_backpropagation_demonstration_slopes_match_finite_differences(demonstration_values):
    values = demonstration_values("backpropagation")
    z, axes = values["demo_z"], values["axes"]
    for curve, derivative in zip(axes[0].lines[:2], axes[1].lines[:2]):
        numeric = np.gradient(curve.get_ydata(), z)
        np.testing.assert_allclose(
            derivative.get_ydata()[1:-1], numeric[1:-1], rtol=1e-3, atol=1e-4
        )
    middle = int(np.argmin(np.abs(z)))
    assert axes[1].lines[0].get_ydata()[middle] == pytest.approx(.25)
    assert axes[1].lines[1].get_ydata()[middle] == pytest.approx(1.)
    np.testing.assert_array_equal(axes[1].lines[2].get_ydata(), (z > 0).astype(float))


def test_mlp_demonstration_target_shape_noise_and_split(demonstration_values):
    values = demonstration_values("multilayer-perceptron")
    X, Y = values["demo_X"], values["demo_Y"]
    expected = (X[:, 0] * X[:, 1] < 0).astype(float)
    expected = np.where(values["demo_flips"], 1 - expected, expected)[:, None]
    assert Y.shape == (360, 1)
    np.testing.assert_array_equal(Y, expected)
    partitions = [values[f"demo_{name}_ids"] for name in ("train", "validation", "test")]
    assert list(map(len, partitions)) == [210, 75, 75]
    np.testing.assert_array_equal(np.sort(np.concatenate(partitions)), np.arange(360))


def test_regression_parameter_visualization_uses_the_displayed_loss(demonstration_values):
    values = demonstration_values("linear-regression")
    np.testing.assert_allclose(
        values["intuition_candidate_losses"], [.6725, .045, .9225], atol=1e-12
    )
    surface = values["intuition_loss_surface"]
    assert surface.shape == (100, 100) and np.isfinite(surface).all()
    i, j = 23, 61
    slope = values["intuition_slopes"][i, j]
    bias = values["intuition_biases"][i, j]
    independent = sum(
        (slope * x + bias - y)**2
        for x, y in zip(values["intuition_x"], values["intuition_y"])
    ) / 8
    assert surface[i, j] == pytest.approx(independent, abs=1e-12)


def test_quadratic_visualization_states_and_loss_are_consistent(demonstration_values):
    values = demonstration_values("gradient-descent")
    states, losses = values["intuition_trajectory"], values["intuition_trajectory_losses"]
    assert states.shape == (13, 2)
    np.testing.assert_allclose(states[0], [2.5, .9], atol=1e-12)
    np.testing.assert_allclose(states[1:], states[:-1] * [.8, -.6], atol=1e-12)
    np.testing.assert_allclose(losses, np.sum(states**2 * [1, 8], axis=1) / 2)
    assert np.all(np.diff(losses) < 0)


def test_batch_comparison_matches_updates_full_loss_and_data_budgets(demonstration_values):
    values = demonstration_values("gradient-descent")
    targets = np.array([-2., -1., 1., 2.])
    orders = values["intuition_orders"]
    np.testing.assert_array_equal(np.sort(orders, axis=1), np.tile(targets, (20, 1)))
    for name, (means, size) in values["intuition_schedules"].items():
        result = values["intuition_comparison"][name]
        states = result["states"]
        assert states[0] == 3
        np.testing.assert_allclose(states[1:], .85 * states[:-1] + .15 * means, atol=1e-12)
        independent_full_loss = np.mean((states[:, None] - targets)**2, axis=1) / 2
        np.testing.assert_allclose(result["excess_loss"], independent_full_loss - 1.25, atol=1e-12)
        np.testing.assert_array_equal(result["examples"], result["updates"] * size)
        assert result["examples"][-1] == 80
        assert result["updates"][-1] == {"Batch GD": 20, "SGD": 80, "Mini-batch SGD": 40}[name]


@pytest.mark.parametrize("n, d, m", [(1, 2, 3), (4, 3, 2)])
def test_affine_vjp_against_independent_scalarization(n, d, m):
    f = reference_functions("arrays-and-matrix-calculus")
    rng = np.random.default_rng(90)
    X, W, b, G = rng.normal(size=(n, d)), rng.normal(size=(d, m)), rng.normal(size=m), rng.normal(size=(n, m))
    originals = [a.copy() for a in (X, W, b, G)]
    z, dx, dw, db = f["affine_vjp"](X, W, b, G)
    np.testing.assert_allclose(z, X @ W + b)
    for observed, parameter, objective in (
        (dx, X, lambda v: np.sum((v @ W + b) * G)),
        (dw, W, lambda v: np.sum((X @ v + b) * G)),
        (db, b, lambda v: np.sum((X @ W + v) * G)),
    ):
        assert observed.shape == parameter.shape
        np.testing.assert_allclose(observed, central_difference(objective, parameter), rtol=1e-6, atol=1e-8)
    for actual, original in zip((X, W, b, G), originals):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(ValueError):
        f["affine_vjp"](X, W, b, G.ravel())


def test_jacobian_direction_curvature_and_diagnostic_values():
    f = reference_functions("arrays-and-matrix-calculus")
    function = lambda t: np.array([t[0]**2 + t[1], t[0] * t[1]])
    np.testing.assert_allclose(f["finite_jacobian"](function, [2, 3]), [[4, 1], [3, 2]], atol=1e-8)
    np.testing.assert_allclose(f["directional_jvp"](function, [2, 3], [1, -1]), [3, 1], atol=1e-8)
    np.testing.assert_allclose(np.array([[4, 1], [3, 2]]).T @ [2, -1], [5, 0])
    Q, r, point = np.array([[2., 1.], [1., 3.]]), np.array([1., -1.]), np.array([1., 2.])
    assert .5 * point @ Q @ point - r @ point == 10
    np.testing.assert_allclose(Q @ point - r, [3, 8])
    np.testing.assert_allclose(f["hessian_vector"](lambda t: Q @ t - r, point, [1, -1]), [1, -2], atol=1e-8)
    with pytest.raises(ValueError):
        f["finite_jacobian"](function, [2, 3], step=0)
    with pytest.raises(ValueError):
        f["hessian_vector"](lambda t: np.ones(3), [1, 2], [1, 1])


def test_bayes_covariance_and_prior_examples():
    f = reference_functions("probability-and-statistics")
    posterior, evidence = f["discrete_posterior"]([.1, .9], [.8, .1])
    np.testing.assert_allclose(posterior, [8/17, 9/17])
    assert evidence == pytest.approx(.17)
    with pytest.raises(ValueError):
        f["discrete_posterior"]([.5, .5], [0, 0])
    X = np.array([[0., 1.], [2., 1.], [4., 4.]])
    for ddof, expected in [(0, [[8/3, 2], [2, 2]]), (1, [[4, 3], [3, 3]])]:
        mean, covariance = f["sample_covariance"](X, ddof)
        np.testing.assert_allclose(mean, [2, 2])
        np.testing.assert_allclose(covariance, expected)
        np.testing.assert_allclose(covariance, np.cov(X, rowvar=False, ddof=ddof))
    assert f["sample_covariance"](X[:, :1])[1].shape == (1, 1)
    with pytest.raises(ValueError):
        f["sample_covariance"]([[1]], ddof=1)
    # Beta(3,5): posterior mode 1/3, mean 3/8, versus MLE 1/4.
    assert (3 - 1) / (3 + 5 - 2) == pytest.approx(1/3)
    assert 3 / (3 + 5) == .375


@pytest.mark.parametrize("p", [.25, .55, .8])
def test_bernoulli_gradient_with_independent_product_likelihood(p):
    f = reference_functions("probability-and-statistics")
    y = np.array([0., 1., 1., 0.])
    loss, derivative = f["bernoulli_nll"](p, y)
    independent = lambda t: -np.log(np.prod(t[0]**y * (1 - t[0])**(1-y))) / len(y)
    assert loss == pytest.approx(independent([p]))
    np.testing.assert_allclose([derivative], central_difference(independent, [p]), atol=1e-8)
    with pytest.raises(ValueError):
        f["bernoulli_nll"](0, y)


def test_bootstrap_is_reproducible_and_uses_observation_resampling():
    bootstrap = reference_functions("probability-and-statistics")["bootstrap_mean_interval"]
    values = np.array([1., 2., 8., 10.])
    mean, interval, estimates = bootstrap(values, repeats=30, level=.8, seed=51)
    rng = np.random.default_rng(51)
    independent = [np.mean(values[rng.choice(len(values), len(values), replace=True)]) for _ in range(30)]
    assert mean == 5.25
    np.testing.assert_array_equal(estimates, independent)
    np.testing.assert_allclose(interval, np.quantile(independent, [.1, .9]))
    np.testing.assert_array_equal(bootstrap(values, 30, .8, 51)[2], estimates)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_stable_logsumexp_reduction_values_and_shapes(axis, keepdims):
    f = reference_functions("numerical-computing")
    values = np.array([[1000., 999.], [-1000., -998.]])
    expected = np.logaddexp.reduce(values, axis=axis, keepdims=keepdims)
    result = f["stable_logsumexp"](values, axis, keepdims)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=1e-12)
    with pytest.raises(ValueError):
        f["stable_logsumexp"]([-np.inf, -np.inf])


def test_softplus_solve_and_streaming_against_independent_oracles():
    f = reference_functions("numerical-computing")
    with np.errstate(over="raise", invalid="raise"):
        value, derivative = f["softplus_sigmoid"]([-1000, 0, 1000])
    np.testing.assert_allclose(value, [0, np.log(2), 1000])
    np.testing.assert_allclose(derivative, [0, .5, 1])
    assert np.ndim(f["softplus_sigmoid"](.2)[0]) == 0
    x, residual, condition = f["solve_checked"]([[2, 0], [0, 4]], [6, 8])
    np.testing.assert_allclose(x, [3, 2])
    assert residual == 0 and condition == 2
    with pytest.raises(np.linalg.LinAlgError):
        f["solve_checked"]([[1, 1], [1, 1]], [1, 1])
    values = 1e6 + np.random.default_rng(5).normal(0, .1, 500)
    for ddof in (0, 1):
        n, mean, variance = f["streaming_moments"]((values[i:i+17] for i in range(0, 500, 17)), ddof)
        assert n == 500
        assert mean == pytest.approx(values.mean(), abs=1e-8)
        assert variance == pytest.approx(np.var(values, ddof=ddof), rel=1e-7)
    with pytest.raises(ValueError):
        f["streaming_moments"]([[]])


@pytest.mark.parametrize("dimension", [1, 3])
def test_gaussian_project_gradient_and_loglikelihood_independently(dimension):
    f = reference_functions("mathematical-audit")
    X = np.arange(5 * dimension).reshape(5, dimension) / 7
    mu = np.linspace(-.2, .3, dimension)
    covariance = np.eye(dimension) * 1.3 + .1
    loss, gradient = f["gaussian_nll"](X, mu, covariance)
    inverse = np.linalg.inv(covariance)  # Deliberately independent test oracle only.
    def objective(candidate):
        total = sum((row-candidate) @ inverse @ (row-candidate) for row in X)
        return dimension / 2 * np.log(2*np.pi) + .5*np.log(np.linalg.det(covariance)) + total / (2*len(X))
    assert loss == pytest.approx(objective(mu))
    np.testing.assert_allclose(gradient, central_difference(objective, mu), rtol=1e-6, atol=1e-8)
    analytic, numeric, absolute, relative = f["audit_mean_gradient"](X, mu, covariance)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
    assert absolute < 1e-8 and relative < 1e-8


def test_gaussian_project_hand_values_and_covariance_domain():
    f = reference_functions("mathematical-audit")
    X = np.array([[1., 0.], [3., 2.], [2., 4.], [4., 2.]])
    before = X.copy()
    mean, mle, unbiased = f["summary_estimates"](X)
    np.testing.assert_allclose(mean, [2.5, 2])
    np.testing.assert_allclose(mle, [[1.25, .5], [.5, 2]])
    np.testing.assert_allclose(unbiased, [[5/3, 2/3], [2/3, 8/3]])
    loss, gradient = f["gaussian_nll"](X, [0, 0], np.eye(2))
    assert loss == pytest.approx(np.log(2*np.pi) + 6.75)
    np.testing.assert_allclose(gradient, [-2.5, -2])
    for covariance in (np.diag([-1., -1.]), np.diag([1., 0.])):
        with pytest.raises(np.linalg.LinAlgError):
            f["gaussian_nll"](X, [0, 0], covariance)
    with pytest.raises(ValueError):
        f["gaussian_nll"](X, [0, 0], [[1, .3], [.1, 1]])
    np.testing.assert_array_equal(X, before)


OBJECTIVES = [
    ("regularization-and-generalization", "ridge_loss_gradient", "regression"),
    ("regularization-and-generalization", "regularized_logistic_loss_gradient", "classification"),
    ("ml-pipeline", "pipeline_objective", "regression"),
    ("ml-pipeline", "pipeline_objective", "classification"),
]


@pytest.mark.parametrize("chapter,function,task", OBJECTIVES)
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("penalty", [0., .17])
def test_new_supervised_objectives_independent_gradients_and_scaling(chapter, function, task, n, penalty):
    f = reference_functions(chapter)[function]
    X = np.linspace(-.8, 1.1, n*2).reshape(n, 2)
    y = (np.arange(n) % 2).astype(float) if task == "classification" else np.linspace(.1, .9, n)
    theta = np.array([.3, -.4, .2]); before = [a.copy() for a in (X, y, theta)]
    def independent(candidate):
        z = X @ candidate[:2] + candidate[2]
        if task == "regression":
            data = sum(float(a-b)**2 for a, b in zip(z, y)) / (2*n)
        else:
            p = 1 / (1 + np.exp(-z))
            data = -sum(float(t)*np.log(q) + (1-float(t))*np.log(1-q) for t, q in zip(y, p)) / n
        return data + penalty * sum(float(v)**2 for v in candidate[:2]) / 2
    args = (task,) if function == "pipeline_objective" else ()
    loss, dw, db = f(X, y, theta[:2], theta[2], penalty, *args)
    assert loss == pytest.approx(independent(theta), abs=1e-12)
    assert dw.shape == (2,) and np.ndim(db) == 0
    np.testing.assert_allclose(np.r_[dw, db], central_difference(independent, theta), rtol=1e-5, atol=1e-8)
    doubled = f(np.tile(X, (2, 1)), np.tile(y, 2), theta[:2], theta[2], penalty, *args)
    np.testing.assert_allclose(np.r_[doubled[0], doubled[1], doubled[2]], np.r_[loss, dw, db], atol=1e-12)
    zero = f(X, y, theta[:2], theta[2], 0., *args)
    np.testing.assert_allclose(dw-zero[1], penalty*theta[:2], atol=1e-12)
    assert db == zero[2]  # Intercept is never penalized.
    for actual, original in zip((X, y, theta), before):
        np.testing.assert_array_equal(actual, original)


def test_ridge_hand_values_closed_form_and_rank_deficient_policy():
    f = reference_functions("regularization-and-generalization")
    loss, dw, db = f["ridge_loss_gradient"]([[1, 0], [0, 2]], [2, 1], [0, 1], 0, .2)
    np.testing.assert_allclose([loss, *dw, db], [1.35, -1, 1.2, -.5])
    w, b = f["fit_ridge"]([[-1], [1]], [-3, 3], 1.)
    np.testing.assert_allclose(w, [1.5]); assert b == 0
    X = np.array([[1., 2.], [2., 4.], [3., 6.]])
    for penalty in (0., .2):
        w, b = f["fit_ridge"](X, [1, 2, 3], penalty)
        _, dw, db = f["ridge_loss_gradient"](X, [1, 2, 3], w, b, penalty)
        np.testing.assert_allclose(np.r_[dw, db], 0., atol=1e-12)
    with pytest.raises(ValueError):
        f["fit_ridge"](X, [1, 2, 3], -.1)


@pytest.mark.parametrize("chapter,function,task", OBJECTIVES)
def test_supervised_domain_rejects_broadcasting_and_invalid_data(chapter, function, task):
    f = reference_functions(chapter)[function]
    args = (task,) if function == "pipeline_objective" else ()
    for labels in ([[0], [1]], [0, np.nan]):
        with pytest.raises(ValueError):
            f([[1, 0], [0, 1]], labels, [.2, -.3], .1, .2, *args)
    with pytest.raises(ValueError):
        f([[1, 0], [0, 1]], [0, 1], [.2, -.3], .1, -.2, *args)
    if task == "classification":
        with pytest.raises(ValueError):
            f([[1, 0], [0, 1]], [0, 2], [.2, -.3], .1, .2, *args)


@pytest.mark.parametrize("chapter,function", [
    ("regularization-and-generalization", "regularized_logistic_loss_gradient"),
    ("ml-pipeline", "pipeline_objective"),
])
def test_new_classification_extreme_logits_are_finite(chapter, function):
    args = ("classification",) if function == "pipeline_objective" else ()
    with np.errstate(over="raise", invalid="raise"):
        loss, dw, db = reference_functions(chapter)[function]([[1], [-1]], [0, 1], [1000], 0., 0., *args)
    assert loss == 1000 and np.isfinite(dw).all() and np.isfinite(db)


def test_manual_trainers_record_parameter_snapshots_and_converge_on_small_case():
    f = reference_functions("regularization-and-generalization")
    X, y = np.array([[-1.], [-.5], [.5], [1.]]), np.array([0., 0., 1., 1.])
    w, b, history = f["fit_regularized_logistic"](X, y, penalty=.2, steps=300)
    assert history.shape == (301,) and history[0] == pytest.approx(np.log(2))
    assert np.all(np.diff(history) <= 1e-12)
    assert history[-1] == pytest.approx(f["regularized_logistic_loss_gradient"](X, y, w, b, .2)[0])
    with pytest.raises(ValueError):
        f["fit_regularized_logistic"](X, np.zeros(4))
    p = reference_functions("ml-pipeline")
    model = p["pipeline_fit"](X, 2*X[:, 0]+.4, .2, "regression", steps=400)
    expected_w = np.mean(X[:, 0] * (2*X[:, 0])) / (np.mean(X[:, 0]**2)+.2)
    np.testing.assert_allclose(model['w'], [expected_w], atol=1e-9)
    assert model['b'] == pytest.approx(.4, abs=1e-9)
    assert len(model['history']) == 401 and np.all(np.diff(model['history']) <= 1e-12)
    for task in ("regression", "classification"):
        analytic, numeric = p["pipeline_gradient_check"](X, y, [.3], .2, .15, task)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)


def test_regression_metrics_and_constant_target_convention():
    f = reference_functions("evaluation-and-model-selection")
    result = f["regression_metrics"]([1, 2, 4], [0, 2, 5])
    assert result == pytest.approx({"mse": 2/3, "rmse": np.sqrt(2/3), "mae": 2/3, "r2": 4/7})
    assert f["regression_metrics"]([1, 2], [0, 0])["r2"] == -9
    assert np.isnan(f["regression_metrics"]([1, 1], [1, 1])["r2"])
    with pytest.raises(ValueError):
        f["regression_metrics"]([1, 2], [[1], [2]])


def test_classification_probability_hand_values_ties_and_empty_bins():
    f = reference_functions("evaluation-and-model-selection")
    y, p = [0, 0, 1, 1], [.1, .6, .4, .9]
    metrics = f["classification_metrics"](y, p)
    assert metrics == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "accuracy": .5, "precision": .5, "recall": .5, "f1": .5}
    result = f["probability_metrics"](y, p)
    assert result['brier'] == pytest.approx(.185)
    assert result['clipped_log_loss'] == pytest.approx(.5108256237659907)
    baseline = f["probability_metrics"](y, [.25]*4)
    assert baseline['brier'] == .3125
    assert baseline['clipped_log_loss'] == pytest.approx(.8369882167858358)
    assert result['calibration'][:, 0].sum() == 4
    boundaries = f["probability_metrics"]([0, 1], [0, 1], bins=5)
    np.testing.assert_array_equal(boundaries['calibration'][:, 0], [1, 0, 0, 0, 1])
    assert np.isnan(boundaries['calibration'][1:4, 1:]).all()
    assert boundaries['brier'] == 0 and np.isfinite(boundaries['clipped_log_loss'])
    tied = f["classification_metrics"]([0, 1], [.5, .5], .5)
    assert tied['tp'] == tied['fp'] == 1 and tied['fn'] == tied['tn'] == 0


def test_one_class_metric_conventions_are_explicit():
    f = reference_functions("evaluation-and-model-selection")
    result = f["classification_metrics"]([0, 0], [.1, .2])
    assert result['accuracy'] == 1
    assert all(np.isnan(result[key]) for key in ('precision', 'recall', 'f1'))
    assert np.isfinite(f["probability_metrics"]([0, 0], [.1, .2])['brier'])
    with pytest.raises(ValueError):
        f["binary_curves"]([0, 0], [.1, .2])
    with pytest.raises(ValueError):
        f["classification_metrics"]([0, 1], [1.1, .2])


@pytest.mark.parametrize("p", [[.1, .6, .4, .9], [.5, .5, .5, .5], [.8, .7, .2, .1], [.2, .5, .5, .8]])
def test_roc_auc_against_independent_positive_negative_pair_comparison(p):
    y = np.array([0, 0, 1, 1]); p = np.array(p)
    expected = np.mean([float(a > b) + .5 * float(a == b) for a in p[y == 1] for b in p[y == 0]])
    thresholds, fpr, tpr, precision, auc = reference_functions("evaluation-and-model-selection")["binary_curves"](y, p)
    assert auc == pytest.approx(expected)
    assert np.isinf(thresholds[0]) and precision[0] == 1
    assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


@pytest.mark.parametrize("n,folds", [(11, 3), (2, 2)])
def test_fold_coverage_and_standardizer_boundary(n, folds):
    f = reference_functions("evaluation-and-model-selection")
    splits = f["kfold_indices"](n, folds, seed=13)
    np.testing.assert_array_equal(np.sort(np.concatenate([held for _, held in splits])), np.arange(n))
    for training, held in splits:
        assert not set(training) & set(held)
        assert set(training) | set(held) == set(range(n))
    state = f["fit_standardizer"]([[0, 5], [2, 5]])
    np.testing.assert_array_equal(state, [[1, 5], [1, 1]])
    np.testing.assert_array_equal(f["transform_standardizer"]([[1000, 7]], state), [[999, 2]])
    np.testing.assert_array_equal(state, [[1, 5], [1, 1]])
    with pytest.raises(ValueError):
        f["kfold_indices"](n, n+1)


def test_ridge_cv_independent_augmented_solve_and_fold_local_fits(monkeypatch):
    f = reference_functions("evaluation-and-model-selection")
    X = np.column_stack([np.arange(12, dtype=float), np.ones(12)])
    X[-1, 0] = 1000
    y, grid = np.sin(np.arange(12)), [0., .1]
    fitted_rows, original = [], f['fit_standardizer']
    def tracked(rows):
        fitted_rows.append(rows.copy())
        return original(rows)
    monkeypatch.setitem(f, 'fit_standardizer', tracked)
    chosen, scores = f['ridge_cv'](X, y, grid, folds=3, seed=14)
    splits = f['kfold_indices'](12, 3, 14)
    assert len(fitted_rows) == 3
    expected = np.empty((2, 3))
    for j, (training, held) in enumerate(splits):
        np.testing.assert_array_equal(fitted_rows[j], X[training])
        mean, scale = X[training].mean(axis=0), X[training].std(axis=0)
        scale[scale == 0] = 1
        A = np.column_stack([(X[training]-mean)/scale, np.ones(len(training))])
        B = np.column_stack([(X[held]-mean)/scale, np.ones(len(held))])
        for i, penalty in enumerate(grid):
            regularizer = np.diag([np.sqrt(len(training)*penalty)]*2 + [0.])
            coefficients = np.linalg.lstsq(np.vstack([A, regularizer]), np.r_[y[training], np.zeros(3)], rcond=None)[0]
            expected[i, j] = np.mean((B @ coefficients-y[held])**2)
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)
    assert chosen == grid[np.argmin(expected.mean(axis=1))]


def test_validation_threshold_cost_and_ordered_tie_rule():
    f = reference_functions("evaluation-and-model-selection")
    threshold, costs = f['choose_threshold']([0, 0, 1, 1], [.1, .6, .4, .9], [.3, .5, .7], fp_cost=1, fn_cost=2)
    np.testing.assert_allclose(costs, [.25, .75, .5]); assert threshold == .3
    threshold, costs = f['choose_threshold']([0, 1], [.1, .9], [.6, .4])
    assert threshold == .6 and np.all(costs == 0)


def test_project_stratified_folds_and_preserved_split_data(demonstration_values):
    f = reference_functions("ml-pipeline")
    labels = np.array([0]*9+[1]*9)
    splits = f['pipeline_folds'](18, 3, labels=labels)
    np.testing.assert_array_equal(np.sort(np.concatenate([held for _, held in splits])), np.arange(18))
    for training, held in splits:
        assert not set(training) & set(held)
        assert len(np.unique(labels[held])) == len(np.unique(labels[training])) == 2
    with pytest.raises(ValueError):
        f['pipeline_folds'](8, 5, labels=[0]*4+[1]*4)
    values = demonstration_values('ml-pipeline')
    np.testing.assert_array_equal(np.sort(np.concatenate(list(values['split_ids'].values()))), np.arange(300))
    for task in ('regression', 'classification'):
        assert [len(values['data_sets'][task][part][0]) for part in ('train', 'validation', 'test')] == [180, 60, 60]
        assert all(values['data_sets'][task][part][0].shape[1] == 5 for part in ('train', 'validation', 'test'))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_project_cv_against_independent_fixed_work_training_and_isolation(task, monkeypatch):
    f = reference_functions('ml-pipeline')
    X = np.column_stack([np.linspace(-1, 1, 18), np.cos(np.arange(18))])
    y = (np.arange(18) % 2).astype(float) if task == 'classification' else np.sin(np.arange(18))
    grid, steps = [0., .2], 30
    observed_rows, standardizer = [], f['pipeline_standardizer']
    def tracked(rows):
        observed_rows.append(rows.copy())
        return standardizer(rows)
    monkeypatch.setitem(f, 'pipeline_standardizer', tracked)
    chosen, scores = f['pipeline_select'](X, y, grid, task, folds=3, steps=steps)
    splits = f['pipeline_folds'](18, 3, 73, y if task == 'classification' else None)
    expected = np.empty_like(scores)
    assert len(observed_rows) == 3
    for j, (training, held) in enumerate(splits):
        np.testing.assert_array_equal(observed_rows[j], X[training])
        mean, scale = X[training].mean(axis=0), X[training].std(axis=0)
        A, B = (X[training]-mean)/scale, (X[held]-mean)/scale
        for i, penalty in enumerate(grid):
            w, b = np.zeros(2), 0.
            for _ in range(steps):
                z = A @ w + b
                residual = z-y[training] if task == 'regression' else 1/(1+np.exp(-z))-y[training]
                weight_gradient = np.array([sum(A[k, q]*residual[k] for k in range(len(A)))/len(A) for q in range(2)]) + penalty*w
                w, b = w-.1*weight_gradient, b-.1*float(sum(residual)/len(A))
            z = B @ w + b
            if task == 'regression':
                expected[i, j] = np.mean((z-y[held])**2)
            else:
                probability = 1/(1+np.exp(-z))
                expected[i, j] = -np.mean(y[held]*np.log(probability)+(1-y[held])*np.log1p(-probability))
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    assert chosen == grid[np.argmin(expected.mean(axis=1))]


def test_new_visualizations_match_regularized_stationarity_and_roc_area(demonstration_values):
    values = demonstration_values('regularization-and-generalization')
    np.testing.assert_allclose(values['weights']-3+values['penalties']*values['weights'], 0., atol=1e-14)
    np.testing.assert_allclose(values['data_loss'], (values['weights']-3)**2/2)
    values = demonstration_values('evaluation-and-model-selection')
    curve = values['axes'][1].lines[0]
    assert np.trapezoid(curve.get_ydata(), curve.get_xdata()) == .75


def test_information_losses_hand_values_support_and_domain():
    f = reference_functions('probability-information-and-decisions')
    with np.errstate(divide='raise', invalid='raise'):
        entropy, cross, divergence = f['information_losses']([.25, .75], [.5, .5])
        np.testing.assert_allclose([entropy, cross, divergence], [.5623351446188083, .6931471805599453, .13081203594113697], atol=1e-14)
        assert entropy / np.log(2) == pytest.approx(.8112781244591328)
        np.testing.assert_array_equal(f['information_losses']([1., 0.], [1., 0.]), [0., 0., 0.])
        result = f['information_losses']([0., 1.], [1., 0.])
        assert result[0] == 0 and np.isinf(result[1:]).all()
    for p, q in [([.2, .2], [.5, .5]), ([-.1, 1.1], [.5, .5]), ([np.nan, 0], [.5, .5]), ([1.], [.5, .5])]:
        with pytest.raises(ValueError):
            f['information_losses'](p, q)


@pytest.mark.parametrize('categories', [2, 3, 7])
def test_information_identity_nonnegativity_and_permutation(categories):
    import math
    f = reference_functions('probability-information-and-decisions')
    rng = np.random.default_rng(601)
    p, q = rng.dirichlet(np.ones(categories), size=2)
    h, cross, divergence = f['information_losses'](p, q)
    independent = sum(float(a)*math.log(float(a)/float(b)) for a, b in zip(p, q))
    assert divergence == pytest.approx(independent, abs=1e-14) and divergence >= -1e-14
    assert cross == pytest.approx(h + divergence, abs=1e-14)
    assert h <= np.log(categories) + 1e-14
    order = rng.permutation(categories)
    np.testing.assert_allclose(f['information_losses'](p[order], q[order]), [h, cross, divergence], atol=1e-14)


def test_joint_information_independent_deterministic_and_zero_categories():
    f = reference_functions('probability-information-and-decisions')
    np.testing.assert_allclose(f['joint_information']([[.5, 0], [0, .5]]), [np.log(2)]*4)
    np.testing.assert_allclose(f['joint_information'](np.full((2, 2), .25)), [np.log(2), np.log(2), np.log(4), 0], atol=1e-14)
    joint = np.outer([.3, .7, 0.], [.1, .4, .5])
    hx, hy, hxy, mutual = f['joint_information'](joint)
    assert mutual == pytest.approx(0, abs=1e-14)
    assert hx + hy - hxy == pytest.approx(mutual, abs=1e-14)
    # The product of the two tiny marginals would underflow; log marginals do not.
    with np.errstate(divide='raise', invalid='raise'):
        values = f['joint_information']([[1e-200, 0], [0, 1.]])
    assert np.isfinite(values).all() and values[3] > 0
    with pytest.raises(ValueError):
        f['joint_information']([[.2, .2]])


@pytest.mark.parametrize('n,categories', [(1, 2), (1, 4), (5, 3)])
def test_categorical_logit_gradients_against_scalar_objective(n, categories):
    f = reference_functions('probability-information-and-decisions')
    logits = np.linspace(-.7, .9, n*categories).reshape(n, categories)
    labels = np.arange(n) % categories
    before = logits.copy()
    def independent(candidate):
        return sum(np.log(sum(np.exp(float(v)) for v in row))-row[target] for row, target in zip(candidate, labels)) / n
    loss, probability, gradient = f['categorical_nll'](logits, labels)
    assert loss == pytest.approx(independent(logits), abs=1e-12)
    np.testing.assert_allclose(gradient, central_difference(independent, logits), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-15)
    shifted = f['categorical_nll'](logits+1000, labels)
    assert shifted[0] == pytest.approx(loss, abs=1e-12)
    np.testing.assert_allclose(shifted[1], probability, atol=1e-13)
    np.testing.assert_array_equal(logits, before)


def test_categorical_extremes_shapes_and_labels():
    f = reference_functions('probability-information-and-decisions')
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        loss, p, gradient = f['categorical_nll']([[1000., -1000.]], [1])
    assert loss == 2000
    np.testing.assert_array_equal(p, [[1., 0.]])
    np.testing.assert_array_equal(gradient, [[1., -1.]])
    for labels in ([1.], [[1]], [2], [-1]):
        with pytest.raises(ValueError):
            f['categorical_nll']([[1., 2.]], labels)
    with pytest.raises(ValueError):
        f['categorical_nll']([[np.nan, 0]], [0])
    with pytest.raises(FloatingPointError):
        f['categorical_nll']([[1e308, -1e308]], [1])


@pytest.mark.parametrize('n', [1, 4])
@pytest.mark.parametrize('penalty', [0., .13])
def test_categorical_affine_all_parameters_and_mean_penalty_scaling(n, penalty):
    f = reference_functions('probability-information-and-decisions')
    X = np.linspace(-.6, .8, n*2).reshape(n, 2)
    labels = np.arange(n) % 3
    W, b = np.linspace(-.3, .4, 6).reshape(2, 3), np.array([.1, -.2, .3])
    W_before, b_before = W.copy(), b.copy()
    def independent(weight, bias):
        z = X @ weight + bias
        data = sum(np.log(sum(np.exp(float(v)) for v in row))-row[target] for row, target in zip(z, labels)) / n
        return data + penalty * sum(float(v)**2 for v in weight.ravel()) / 2
    loss, dw, db = f['categorical_affine_loss'](X, labels, W, b, penalty)
    assert loss == pytest.approx(independent(W, b))
    np.testing.assert_allclose(dw, central_difference(lambda v: independent(v, b), W), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(db, central_difference(lambda v: independent(W, v), b), rtol=1e-6, atol=1e-8)
    duplicated = f['categorical_affine_loss'](np.tile(X, (2, 1)), np.tile(labels, 2), W, b, penalty)
    for actual, expected in zip(duplicated, (loss, dw, db)):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(W, W_before); np.testing.assert_array_equal(b, b_before)
    with pytest.raises(ValueError):
        f['categorical_affine_loss'](X, labels, W, b[:, None], penalty)


def test_categorical_curvature_variance_nullspace_and_jacobian():
    f = reference_functions('probability-information-and-decisions')
    point = np.array([.2, -.5, .7])
    _, p, _ = f['categorical_nll'](point[None, :], [1]); p = p[0]
    hessian = np.diag(p) - np.outer(p, p)
    def probability(candidate):
        e = np.exp(candidate)
        return e/e.sum()
    numeric = np.column_stack([central_difference(lambda v: probability(v)[j], point) for j in range(3)]).T
    np.testing.assert_allclose(hessian, numeric, atol=1e-10)
    np.testing.assert_allclose(hessian @ np.ones(3), 0, atol=1e-15)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-14
    direction = np.array([1., -2., .5])
    assert direction @ hessian @ direction == pytest.approx(p @ direction**2 - (p @ direction)**2)


def test_bayes_actions_costs_abstention_and_ties():
    f = reference_functions('probability-information-and-decisions')
    posterior = np.array([[.8, .2], [.6, .4], [.75, .25]])
    action, risk = f['bayes_actions'](posterior, [[0., 3.], [1., 0.]])
    np.testing.assert_array_equal(action, [0, 1, 0])
    np.testing.assert_allclose(risk, [[.6, .8], [1.2, .6], [.75, .75]])
    abstain, risk = f['bayes_actions']([[.5, .5]], [[0, 1], [1, 0], [.3, .3]])
    np.testing.assert_array_equal(abstain, [2]); np.testing.assert_allclose(risk, [[.5, .5, .3]])
    with pytest.raises(ValueError):
        f['bayes_actions']([[.2, .2]], [[0, 1]])
    with pytest.raises(ValueError):
        f['bayes_actions']([[.5, .5]], [[-1, 0]])


@pytest.mark.parametrize('n,noise,prior', [(1, 1., 1.), (5, .7, 2.), (20, 2., .25)])
def test_conjugate_normal_posterior_matches_completed_square_and_mode(n, noise, prior):
    f = reference_functions('probability-information-and-decisions')
    x, m0 = np.linspace(-.2, 1.1, n), .3
    mean, variance, predictive = f['normal_mean_posterior'](x, noise, m0, prior)
    expected_variance = noise*prior/(noise+n*prior)
    expected_mean = (noise*m0+prior*x.sum())/(noise+n*prior)
    np.testing.assert_allclose([mean, variance, predictive], [expected_mean, expected_variance, noise+expected_variance])
    objective = lambda value: sum((float(v)-value[0])**2 for v in x)/(2*noise)+(value[0]-m0)**2/(2*prior)
    np.testing.assert_allclose(central_difference(objective, [mean]), [0], atol=1e-8)
    assert variance < prior and predictive > noise
    np.testing.assert_allclose(f['normal_mean_posterior']([1., 3.], 1.), [4/3, 1/3, 4/3])


def test_map_to_mean_ridge_relationship_and_fixed_intercept():
    f = reference_functions('regularization-and-generalization')
    X, y, w, bias = np.array([[1., 2.], [-.5, .3], [.2, -.4]]), np.array([.4, -.2, .1]), np.array([.3, -.1]), .2
    noise, prior = 1.7, .9
    penalty = noise/(len(X)*prior)
    loss, dw, _ = f['ridge_loss_gradient'](X, y, w, bias, penalty)
    def independent(candidate):
        residual = X @ candidate + bias-y
        return float(residual @ residual/(2*noise)+candidate @ candidate/(2*prior))
    scale = noise/len(X)
    assert loss == pytest.approx(scale*independent(w))
    np.testing.assert_allclose(dw, scale*central_difference(independent, w), atol=1e-8)


def test_information_decision_demonstrations_use_declared_population(demonstration_values):
    values = demonstration_values('probability-information-and-decisions')
    grid = values['forecast']
    independent = -.25*np.log(grid)-.75*np.log1p(-grid)
    np.testing.assert_allclose(values['expected_logloss'], independent)
    assert abs(grid[np.argmin(independent)]-.25) <= np.diff(grid).max()
    np.testing.assert_allclose(values['axes'][1].lines[0].get_ydata(), 3*grid)
    np.testing.assert_allclose(values['axes'][1].lines[1].get_ydata(), 1-grid)


@pytest.mark.parametrize('k', [1, 3, 6])
@pytest.mark.parametrize('query_block,train_block', [(1, 1), (2, 2), (32, 128)])
def test_neighbor_blocks_match_independent_scalar_sort(k, query_block, train_block):
    f = reference_functions('nearest-neighbors')
    train = np.array([[0., 2.], [1., 2.], [1., 2.], [3., 0.], [-1., -2.], [0., 0.]])
    query = np.array([[1., 2.], [.5, 2.], [1e8, 0.]])
    originals = train.copy(), query.copy()
    expected_indices, expected_distances = [], []
    for row in query:
        ordered = sorted((sum((float(a)-float(b))**2 for a, b in zip(row, stored)), index) for index, stored in enumerate(train))[:k]
        expected_indices.append([index for _, index in ordered])
        expected_distances.append([distance for distance, _ in ordered])
    indices, distances = f['neighbor_search'](train, query, k, query_block, train_block)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, expected_distances, atol=1e-14)
    np.testing.assert_array_equal(train, originals[0]); np.testing.assert_array_equal(query, originals[1])


def test_neighbor_large_offsets_empty_queries_and_range_failure():
    f = reference_functions('nearest-neighbors')
    train = np.array([[1e8], [1e8+1], [1e8+2]])
    indices, squared = f['neighbor_search'](train, [[1e8+.9]], 3)
    np.testing.assert_array_equal(indices, [[1, 0, 2]])
    expected = [(float(1e8+.9)-float(train[j, 0]))**2 for j in (1, 0, 2)]
    np.testing.assert_allclose(squared[0], expected, atol=1e-14)
    indices, squared = f['neighbor_search'](train, np.empty((0, 1)), 2)
    assert indices.shape == squared.shape == (0, 2)
    with pytest.raises(FloatingPointError):
        f['neighbor_search']([[1e308]], [[-1e308]], 1)
    for kwargs in ({'k': 0}, {'k': 4}, {'k': True}, {'k': 1, 'train_block': 0}, {'k': 1, 'query_block': 1.5}):
        with pytest.raises(ValueError):
            f['neighbor_search'](train, [[0]], **kwargs)
    with pytest.raises(ValueError):
        f['neighbor_search'](train, [[np.nan]], 1)


def test_neighbor_weight_policies_exact_matches_and_domain():
    f = reference_functions('nearest-neighbors')
    np.testing.assert_allclose(f['neighbor_weights']([[1., 1., 9.]], 'distance'), [[3/7, 3/7, 1/7]])
    np.testing.assert_allclose(f['neighbor_weights']([[1., 1., 9.]]), [[1/3]*3])
    with np.errstate(divide='raise', invalid='raise'):
        weights = f['neighbor_weights']([[0., 0., 4.], [0., 4., 9.]], 'distance')
    np.testing.assert_allclose(weights, [[.5, .5, 0.], [1., 0., 0.]])
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert f['neighbor_weights'](np.empty((0, 3)), 'distance').shape == (0, 3)
    for distance in ([[-1., 0.]], [[np.nan, 0.]], np.empty((2, 0))):
        with pytest.raises(ValueError):
            f['neighbor_weights'](distance)
    with pytest.raises(ValueError):
        f['neighbor_weights']([[1., 2.]], 'squared-inverse')


def test_neighbor_classification_regression_and_noncontiguous_classes():
    f = reference_functions('nearest-neighbors')
    X = np.array([[0.], [2.], [4.], [6.]])
    for mode, expected_scores, expected_regression in [('uniform', [1/3, 2/3], 2.), ('distance', [1/7, 6/7], 18/7)]:
        prediction, scores, classes = f['knn_classify'](X, [0, 1, 1, 0], [[3.]], 3, mode)
        np.testing.assert_array_equal(prediction, [1]); np.testing.assert_array_equal(classes, [0, 1])
        np.testing.assert_allclose(scores, [expected_scores])
        np.testing.assert_allclose(f['knn_regress'](X, [0, 2, 4, 6], [[3.]], 3, mode), [expected_regression])
    prediction, scores, classes = f['knn_classify']([[0.], [1.], [2.]], [7, 9, -3], [[1.]], 3)
    np.testing.assert_array_equal(classes, [-3, 7, 9]); np.testing.assert_array_equal(prediction, [-3])
    np.testing.assert_allclose(scores, [[1/3]*3])
    np.testing.assert_allclose(f['knn_regress'](X, [0, 2, 4, 6], [[2.]], 3, 'distance'), [2.])


def test_conflicting_duplicate_neighbor_policy_and_empty_predictions():
    f = reference_functions('nearest-neighbors')
    X, y = np.array([[1.], [1.], [3.]]), np.array([7, -3, 9])
    prediction, score, classes = f['knn_classify'](X, y, [[1.]], 2, 'distance')
    np.testing.assert_array_equal(prediction, [-3]); np.testing.assert_allclose(score, [[.5, .5, 0.]])
    assert f['knn_classify'](X, y, [[1.]], 1, 'distance')[0][0] == 7
    empty = np.empty((0, 1))
    prediction, score, classes = f['knn_classify'](X, y, empty, 2)
    assert prediction.shape == (0,) and score.shape == (0, 3)
    assert f['knn_regress'](X, y, empty, 2).shape == (0,)
    with pytest.raises(ValueError):
        f['knn_classify'](X, y[:, None], [[1.]], 1)
    with pytest.raises(ValueError):
        f['knn_classify'](X, y.astype(float), [[1.]], 1)
    with pytest.raises(ValueError):
        f['knn_regress'](X, [0, np.nan, 1], [[1.]], 1)


def test_neighbor_regression_is_weighted_squared_loss_minimizer():
    f = reference_functions('nearest-neighbors')
    X, y, query = np.array([[0.], [2.], [4.], [6.]]), np.array([1., -2., 4., 3.]), np.array([[3.]])
    prediction = f['knn_regress'](X, y, query, 3, 'distance')[0]
    # Independent hand-derived neighborhood and inverse Euclidean weights.
    targets, weights = y[[1, 2, 0]], np.array([3/7, 3/7, 1/7])
    objective = lambda c: float(np.sum(weights*(targets-c[0])**2))
    np.testing.assert_allclose(central_difference(objective, [prediction]), [0.], atol=1e-8)
    assert objective([prediction]) < objective([prediction+.5])


def test_neighbor_cv_independent_predictions_pooling_and_fold_local_geometry(monkeypatch):
    f = reference_functions('nearest-neighbors')
    X = np.column_stack([np.arange(16, dtype=float), np.sin(np.arange(16)), np.ones(16)])
    X[-1, 0] = 10000.
    labels, grid, seed = np.array([2]*7+[5]*5+[-1]*4), [1, 3, 5], 209
    calls, classify = [], f['knn_classify']
    def tracked(A, y, B, k, mode='uniform'):
        calls.append((A.copy(), y.copy(), B.copy(), k))
        return classify(A, y, B, k, mode)
    monkeypatch.setitem(f, 'knn_classify', tracked)
    selected, scores = f['select_knn_k'](X, labels, grid, folds=3, seed=seed)
    rng = np.random.default_rng(seed)
    pieces = [np.array_split(rng.permutation(np.flatnonzero(labels == c)), 3) for c in np.unique(labels)]
    expected, sizes = np.empty((3, 3)), []
    assert len(calls) == 9
    for j in range(3):
        held = np.concatenate([p[j] for p in pieces]); training = np.setdiff1d(np.arange(16), held)
        sizes.append(len(held))
        mean, scale = X[training].mean(axis=0), X[training].std(axis=0); scale[scale == 0] = 1.
        A, B = (X[training]-mean)/scale, (X[held]-mean)/scale
        for i, k in enumerate(grid):
            recorded = calls[j*len(grid)+i]
            np.testing.assert_allclose(recorded[0], A); np.testing.assert_array_equal(recorded[1], labels[training])
            np.testing.assert_allclose(recorded[2], B); assert recorded[3] == k
            predictions = []
            for row in B:
                nearest = sorted((sum((float(a)-float(b))**2 for a, b in zip(row, stored)), index) for index, stored in enumerate(A))[:k]
                neighbor_labels = [labels[training[index]] for _, index in nearest]
                choices = sorted(set(neighbor_labels))
                predictions.append(min(choices, key=lambda c: (-neighbor_labels.count(c), c)))
            expected[i, j] = np.mean(np.array(predictions) != labels[held])
    np.testing.assert_array_equal(scores, expected)
    assert selected == grid[np.argmin(np.average(expected, axis=1, weights=sizes))]


def test_neighbor_cv_candidate_order_and_invalid_configuration():
    f = reference_functions('nearest-neighbors')
    X, y = np.arange(9.)[:, None], np.zeros(9, dtype=int)
    selected, scores = f['select_knn_k'](X, y, [3, 1], folds=3)
    assert selected == 3 and np.all(scores == 0)
    for candidates in ([1.], [], [7]):
        with pytest.raises(ValueError):
            f['select_knn_k'](X, y, candidates, folds=3)
    with pytest.raises(ValueError):
        f['select_knn_k'](X, [0]*7+[1]*2, [1], folds=3)


def test_neighbor_geometry_demonstration_uses_uniform_cube_assumption(demonstration_values):
    values = demonstration_values('nearest-neighbors')
    dimension, radius = values['demo_dimensions'], values['demo_halfwidth']
    np.testing.assert_allclose(radius**dimension, .1, rtol=1e-13)
    assert np.all(np.diff(radius) > 0)
    np.testing.assert_allclose(.1**(1/np.array([2., 10., 50.])), [.31622776601683794, .7943282347242815, .954992586021436], atol=1e-14)


def test_generative_log_scores_shift_invariance_and_extreme_contrasts():
    normalize = reference_functions('generative-classifiers')['posterior_from_scores']
    scores = np.array([[1000., 999., 998.], [-900., -900., -900.]])
    logp, p = normalize(scores)
    expected = np.array([math.exp(2), math.exp(1), 1.])
    expected /= expected.sum()
    np.testing.assert_allclose(p, [expected, [1/3]*3], rtol=1e-13)
    np.testing.assert_allclose(normalize(scores + np.array([[21.], [-31.]]))[0], logp)
    np.testing.assert_allclose(np.exp(logp), p)
    extreme_logp, extreme_p = normalize([[10000., -10000.]])
    np.testing.assert_allclose(extreme_logp, [[0., -20000.]])
    np.testing.assert_array_equal(extreme_p, [[1., 0.]])
    assert normalize(np.empty((0, 3)))[0].shape == (0, 3)
    with pytest.raises(FloatingPointError):
        normalize([[np.finfo(float).max, -np.finfo(float).max]])


@pytest.mark.parametrize('scores', [[1., 2.], np.empty((2, 0)), [[np.nan, 0]], [[np.inf, 0]]])
def test_generative_log_scores_reject_invalid_arrays(scores):
    with pytest.raises(ValueError):
        reference_functions('generative-classifiers')['posterior_from_scores'](scores)


def test_gaussian_density_and_mean_gradient_against_scalar_2d_objective():
    density = reference_functions('generative-classifiers')['gaussian_logpdf']
    X = np.array([[-1., .2], [.3, 1.1], [1.4, -.6]])
    mean = np.array([.4, -.2])
    covariance = np.array([[2., .6], [.6, 1.5]])
    a, b, c = 2., .6, 1.5
    determinant = a*c - b*b

    def scalar_nll(candidate):
        total = 0.
        for x0, x1 in X:
            r0, r1 = x0-candidate[0], x1-candidate[1]
            quadratic = (c*r0*r0 - 2*b*r0*r1 + a*r1*r1) / determinant
            total += math.log(2*math.pi) + .5*math.log(determinant) + .5*quadratic
        return total / len(X)

    actual = density(X, mean, covariance)
    expected = []
    for row in X:
        r0, r1 = row-mean
        expected.append(-math.log(2*math.pi)-.5*math.log(determinant)
                        -.5*(c*r0*r0 - 2*b*r0*r1 + a*r1*r1)/determinant)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    delta = mean-X.mean(axis=0)
    analytical = np.array([c*delta[0]-b*delta[1], a*delta[1]-b*delta[0]]) / determinant
    np.testing.assert_allclose(central_difference(scalar_nll, mean), analytical, rtol=1e-6, atol=1e-9)
    assert -actual.mean() == pytest.approx(scalar_nll(mean))
    assert density(np.empty((0, 2)), mean, covariance).shape == (0,)
    fixed = -.5*density([[0.], [2.]], [1.], [[4.]]).sum()
    assert fixed == pytest.approx(.5*math.log(8*math.pi)+.125)


@pytest.mark.parametrize('covariance', [np.zeros((2, 2)), [[1., 2.], [2., 1.]], [[1., .3], [.1, 1.]], [[1., 0.], [0., np.nan]]])
def test_gaussian_log_density_rejects_invalid_covariances(covariance):
    with pytest.raises(ValueError):
        reference_functions('generative-classifiers')['gaussian_logpdf']([[0., 0.]], [0., 0.], covariance)


@pytest.mark.parametrize('kind', ['shared', 'full', 'diagonal'])
def test_gaussian_fit_matches_independent_within_class_scatter_and_replication(kind):
    fit = reference_functions('generative-classifiers')['fit_gaussian_classifier']
    X = np.array([[-2,0], [-1,1], [0,-1], [1,2], [2,1], [3,0], [4,3], [6,1], [5,5]], dtype=float)
    labels = np.array([-4]*4 + [9]*5)
    before = X.copy()
    model = fit(X, labels, kind, jitter=.02)
    means = np.array([[-.5, .5], [4., 2.]])
    scatters = []
    for c, mean in zip([-4, 9], means):
        scatter = np.zeros((2, 2))
        for row in X[labels == c]:
            r = row-mean
            for j in range(2):
                for k in range(2):
                    scatter[j, k] += r[j]*r[k]
        scatters.append(scatter)
    if kind == 'shared':
        expected = np.repeat(((scatters[0]+scatters[1])/9)[None], 2, axis=0)
    else:
        expected = np.array([scatters[0]/4, scatters[1]/5])
        if kind == 'diagonal':
            expected[:, 0, 1] = expected[:, 1, 0] = 0.
    expected += .02*np.eye(2)[None]
    np.testing.assert_array_equal(model['classes'], [-4, 9])
    np.testing.assert_allclose(model['means'], means)
    np.testing.assert_allclose(np.exp(model['log_prior']), [4/9, 5/9])
    np.testing.assert_allclose(model['covariance'], expected)
    duplicate = fit(np.repeat(X, 2, axis=0), np.repeat(labels, 2), kind, jitter=.02)
    for key in ('means', 'covariance', 'log_prior'):
        np.testing.assert_allclose(duplicate[key], model[key])
    np.testing.assert_array_equal(X, before)


def test_gaussian_classifier_hand_odds_empty_queries_and_explicit_singularity():
    f = reference_functions('generative-classifiers')
    X, labels = np.array([[0.], [2.], [4.], [6.]]), np.array([-4, -4, 9, 9])
    model = f['fit_gaussian_classifier'](X, labels)
    predicted, logp, p = f['predict_gaussian_classifier'](model, [[2.], [3.], [4.]])
    np.testing.assert_array_equal(predicted, [-4, -4, 9])
    np.testing.assert_allclose(logp[:, 1]-logp[:, 0], [-4., 0., 4.])
    assert p[0, 1] == pytest.approx(1/(1+math.exp(4)))
    assert np.mean((X-X.mean())**2) == 5.
    assert f['predict_gaussian_classifier'](model, np.empty((0, 1)))[2].shape == (0, 2)
    singular = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    with pytest.raises(ValueError):
        f['fit_gaussian_classifier'](singular, labels)
    regularized = f['fit_gaussian_classifier'](singular, labels, jitter=.1)
    assert np.all(np.linalg.eigvalsh(regularized['covariance']) > 0)
    with pytest.raises(ValueError):
        f['fit_gaussian_classifier'](X, [1., 1., 2., 2.])
    with pytest.raises(ValueError):
        f['fit_gaussian_classifier'](X, [1, 1, 1, 1])


def test_gaussian_shared_and_unequal_variance_odds_against_closed_forms():
    predict = reference_functions('generative-classifiers')['predict_gaussian_classifier']
    x = np.linspace(-3., 3., 11)
    base = dict(classes=np.array([0, 1]), means=np.array([[-1.], [1.]]),
                log_prior=np.log([.5, .5]), covariance=np.array([[[1.]], [[1.]]]))
    _, logp, _ = predict(base, x[:, None])
    np.testing.assert_allclose(logp[:, 1]-logp[:, 0], 2*x, atol=1e-14)
    base['covariance'][1, 0, 0] = 2.
    _, logp, _ = predict(base, x[:, None])
    expected = -.5*math.log(2)-(x-1)**2/4+(x+1)**2/2
    np.testing.assert_allclose(logp[:, 1]-logp[:, 0], expected, atol=1e-14)


@pytest.mark.parametrize('kind', ['bernoulli', 'multinomial'])
def test_discrete_nb_matches_scalar_likelihoods_priors_and_count_denominators(kind):
    f = reference_functions('generative-classifiers')
    if kind == 'bernoulli':
        X = np.array([[1,0], [1,1], [0,1], [0,1]])
        queries = np.array([[1,0], [0,0], [1,1]])
        theta = [[.75, .5], [.25, .75]]
    else:
        X = np.array([[2,0], [1,1], [0,2], [1,1]])
        queries = np.array([[2,0], [0,0], [0,3]])
        theta = [[2/3, 1/3], [1/3, 2/3]]
    model = f['fit_discrete_nb'](X, [-7, -7, 13, 13], kind)
    np.testing.assert_allclose(np.exp(model['log_feature']), theta)
    predicted, logp, probabilities = f['predict_discrete_nb'](model, queries)
    expected = []
    for row in queries:
        joint = []
        for c in (0, 1):
            likelihood = .5
            for j, value in enumerate(row):
                likelihood *= theta[c][j]**value
                if kind == 'bernoulli':
                    likelihood *= (1-theta[c][j])**(1-value)
            joint.append(likelihood)
        expected.append(np.array(joint)/sum(joint))
    np.testing.assert_allclose(probabilities, expected, rtol=1e-13)
    np.testing.assert_allclose(np.exp(logp), expected)
    np.testing.assert_array_equal(predicted, np.array([-7, 13])[np.argmax(expected, axis=1)])
    assert probabilities[0, 0] == pytest.approx(6/7 if kind == 'bernoulli' else .8)
    assert f['predict_discrete_nb'](model, np.empty((0, 2)))[2].shape == (0, 2)
    uneven = f['fit_discrete_nb']([[0,0], [1,0], [0,1]], [1,1,2], kind, alpha=3.)
    np.testing.assert_allclose(np.exp(uneven['log_prior']), [2/3, 1/3])
    if kind == 'multinomial':
        np.testing.assert_allclose(f['predict_discrete_nb'](uneven, [[0,0]])[2][0], [2/3, 1/3])


@pytest.mark.parametrize('features, kind, alpha', [
    ([[2, 0]], 'bernoulli', 1.), ([[.5, 0]], 'multinomial', 1.),
    ([[-1, 0]], 'multinomial', 1.), ([[np.nan, 0]], 'bernoulli', 1.),
    ([[0, 0]], 'bernoulli', 0.), ([[0, 0]], 'bernoulli', -1.),
    ([[0, 0]], 'multinomial', np.inf),
])
def test_discrete_nb_rejects_invalid_inputs(features, kind, alpha):
    with pytest.raises(ValueError):
        reference_functions('generative-classifiers')['fit_discrete_nb'](features, [0], kind, alpha)


def test_discrete_nb_log_space_survives_long_documents_and_uses_absences():
    f = reference_functions('generative-classifiers')
    model = f['fit_discrete_nb']([[2,0], [1,1], [0,2], [1,1]], [0,0,1,1], kind='multinomial')
    _, logp, p = f['predict_discrete_nb'](model, [[10000, 0]])
    assert np.isfinite(logp).all() and p[0, 0] == 1.
    assert logp[0, 0]-logp[0, 1] == pytest.approx(10000*math.log(2))
    bernoulli = dict(kind='bernoulli', classes=np.array([0, 1]), log_prior=np.log([.5, .5]),
                     log_feature=np.log([[.8, .8], [.2, .2]]), log_absent=np.log([[.2, .2], [.8, .8]]))
    assert f['predict_discrete_nb'](bernoulli, [[0, 0]])[2][0, 0] == pytest.approx(1/17)
    assert .5**2000 == 0.
    assert 2000*math.log(.5) == pytest.approx(-1386.2943611198905)


def test_frozen_vocabulary_unknowns_empty_documents_and_presence_features():
    f = reference_functions('generative-classifiers')
    training = ['Blue blue sky', 'red sky']
    vocabulary = f['training_vocabulary'](training)
    assert vocabulary == ('blue', 'red', 'sky')
    queries = ['blue new new', '', 'SKY! blue']
    counts = f['document_counts'](queries, vocabulary)
    np.testing.assert_array_equal(counts, [[1,0,0,2], [0,0,0,0], [1,0,0,1]])
    assert vocabulary == ('blue', 'red', 'sky') and 'new' not in vocabulary
    assert f['document_counts']([], vocabulary).shape == (0, 4)
    np.testing.assert_array_equal(f['document_counts'](['A a'], ()), [[2]])
    np.testing.assert_array_equal(counts > 0, [[1,0,0,1], [0,0,0,0], [1,0,0,1]])
    with pytest.raises(ValueError):
        f['document_counts'](['a'], ('a', 'a'))
    with pytest.raises(ValueError):
        f['document_counts'](['a'], ('A',))


def test_generative_demonstrations_against_enumerated_expected_loss(demonstration_values):
    values = demonstration_values('generative-classifiers')
    np.testing.assert_allclose(values['demo_linear_odds'], 2*values['demo_x'])
    expected = []
    for copies in range(1, 7):
        q = 1 / (1 + 4.**(-copies))
        expected.append(-.8*math.log(q)-.2*math.log1p(-q))
    np.testing.assert_allclose(values['demo_factorized_loss'], expected, rtol=1e-12)
    assert values['demo_true_loss'] == pytest.approx(-.8*math.log(.8)-.2*math.log(.2))
    assert np.all(np.diff(expected) > 0)


def test_poisson_feature_models_give_affine_binary_log_odds_without_gaussians():
    rate0, rate1, prior = 2., 5., .3
    for x in range(7):
        p0 = math.exp(-rate0)*rate0**x / math.factorial(x)
        p1 = math.exp(-rate1)*rate1**x / math.factorial(x)
        ratio = math.log(prior*p1 / ((1-prior)*p0))
        expected = x*math.log(rate1/rate0) - (rate1-rate0) + math.log(prior/(1-prior))
        assert ratio == pytest.approx(expected)


def scalar_tree_risk(targets, criterion):
    """Independent small-vector oracle, deliberately avoiding reference helpers."""
    values = list(targets)
    if criterion == 'mse':
        mean = sum(values)/len(values)
        return sum((value-mean)**2 for value in values)/len(values)
    frequencies = [values.count(c)/len(values) for c in sorted(set(values))]
    if criterion == 'gini':
        return 1-sum(p*p for p in frequencies)
    if criterion == 'entropy':
        return -sum(p*math.log(p) for p in frequencies)
    return 1-max(frequencies)


@pytest.mark.parametrize('criterion', ['mse', 'gini', 'entropy', 'zero_one'])
def test_tree_impurity_against_scalar_loss_and_pure_nodes(criterion):
    impurity = reference_functions('decision-trees')['tree_impurity']
    y = [0,0,0,1,1,2]
    assert impurity(y, criterion) == pytest.approx(scalar_tree_risk(y, criterion))
    assert impurity([5,5,5], criterion) == 0.
    if criterion != 'mse':
        assert impurity([-9,-9,-9,2,2,100], criterion) == pytest.approx(impurity(y, criterion))
        with pytest.raises(ValueError):
            impurity([0., 1.], criterion)
    with pytest.raises(ValueError):
        impurity([], criterion)
    with pytest.raises(ValueError):
        impurity([0, np.nan], criterion)


@pytest.mark.parametrize('criterion', ['mse', 'gini', 'entropy', 'zero_one'])
@pytest.mark.parametrize('min_leaf', [1, 2, 3])
def test_tree_split_matches_exhaustive_scalar_partition_oracle(criterion, min_leaf):
    best_split = reference_functions('decision-trees')['best_tree_split']
    X = np.array([[0,0], [0,1], [1,1], [2,2], [3,2], [3,3]], dtype=float)
    y = [0,0,1,2,2,1]
    candidates = []
    parent = scalar_tree_risk(y, criterion)
    for j in range(2):
        values = sorted(set(float(row[j]) for row in X))
        for a, b in zip(values[:-1], values[1:]):
            threshold = (a+b)/2
            left = [y[i] for i in range(len(y)) if X[i, j] <= threshold]
            right = [y[i] for i in range(len(y)) if X[i, j] > threshold]
            if min(len(left), len(right)) < min_leaf:
                continue
            gain = parent-(len(left)*scalar_tree_risk(left, criterion)
                           + len(right)*scalar_tree_risk(right, criterion))/len(y)
            candidates.append((gain, j, threshold))
    actual = best_split(X, y, criterion, min_leaf)
    assert actual is not None
    assert actual['gain'] == pytest.approx(max(v[0] for v in candidates), abs=1e-14)
    matching = [v for v in candidates if v[1:] == (actual['feature'], actual['threshold'])]
    assert len(matching) == 1 and matching[0][0] == pytest.approx(actual['gain'])
    left = X[:, actual['feature']] <= actual['threshold']
    assert min(left.sum(), (~left).sum()) >= min_leaf
    for value in np.unique(X[:, actual['feature']]):
        assert len(set(left[X[:, actual['feature']] == value])) == 1


def test_tree_hand_gains_weight_children_and_use_global_node_mass():
    f = reference_functions('decision-trees')
    assert f['tree_impurity']([0.,0.,2.,2.], 'mse') == 1.
    assert scalar_tree_risk([0,2,2], 'mse') == pytest.approx(8/9)
    assert 1-.75*f['tree_impurity']([0.,2.,2.], 'mse') == pytest.approx(1/3)
    assert .5-.75*f['tree_impurity']([0,1,1], 'gini') == pytest.approx(1/6)
    entropy_gain = math.log(2)-.75*f['tree_impurity']([0,1,1], 'entropy')
    assert entropy_gain == pytest.approx(.21576155433883565)
    model = f['fit_tree']([[0.], [1.], [2.], [3.]], [0,0,1,1])
    assert model['root']['risk'] == .5
    assert model['root']['left']['risk'] == 0.
    assert (4/20)*.5 == .1


def test_tree_split_ties_constants_repeated_values_and_float_midpoints():
    f = reference_functions('decision-trees')
    X = np.repeat(np.arange(4.)[:, None], 2, axis=1)
    split = f['best_tree_split'](X, [0,0,1,1])
    assert split == dict(feature=0, threshold=1.5, gain=.5)
    tied_threshold = f['best_tree_split']([[0.], [1.], [2.]], [0,1,0])
    assert tied_threshold['threshold'] == .5
    assert f['best_tree_split'](np.ones((4, 2)), [0,1,0,1]) is None
    assert f['best_tree_split'](X, [0,0,1,1], min_leaf=3) is None
    low = 1e308
    high = np.nextafter(low, np.inf)
    adjacent = f['best_tree_split']([[low], [high]], [0,1])
    assert low <= adjacent['threshold'] < high
    opposite = f['best_tree_split']([[-1e308], [1e308]], [0,1])
    assert opposite['threshold'] == 0.
    with pytest.raises(ValueError):
        f['best_tree_split']([[0.], [np.inf]], [0,1])


@pytest.mark.parametrize('criterion', ['mse', 'gini', 'entropy'])
def test_tree_fit_predictions_sizes_stopping_and_empty_queries(criterion):
    f = reference_functions('decision-trees')
    X = np.arange(4.)[:, None]
    y = np.array([0.,0.,2.,2.]) if criterion == 'mse' else np.array([-3,-3,7,7])
    before = X.copy(), y.copy()
    model = f['fit_tree'](X, y, criterion, max_depth=4)
    prediction, probabilities = f['predict_tree'](model, [[1.5], [2.]])
    np.testing.assert_array_equal(prediction, [0.,2.] if criterion == 'mse' else [-3,7])
    assert f['tree_size'](model) == (3,2,1)
    if criterion == 'mse':
        assert probabilities is None
    else:
        np.testing.assert_array_equal(probabilities, [[1,0], [0,1]])
    empty = f['predict_tree'](model, np.empty((0, 1)))
    assert empty[0].shape == (0,)
    if criterion != 'mse':
        assert empty[1].shape == (0, 2)
    root_only = f['fit_tree'](X, y, criterion, max_depth=0)
    assert f['tree_size'](root_only) == (1,1,0)
    assert f['tree_size'](f['fit_tree'](X, y, criterion, min_leaf=3)) == (1,1,0)
    np.testing.assert_array_equal(X, before[0]); np.testing.assert_array_equal(y, before[1])
    with pytest.raises(ValueError):
        f['predict_tree'](model, [[0., 1.]])


def test_tree_positive_gain_contract_keeps_xor_and_conflicting_duplicates_unsplit():
    f = reference_functions('decision-trees')
    X, y = [[0,0], [0,1], [1,0], [1,1]], [0,1,1,0]
    assert f['best_tree_split'](X, y)['gain'] == 0.
    assert f['tree_size'](f['fit_tree'](X, y, max_depth=2)) == (1,1,0)
    duplicate = f['fit_tree']([[1.], [1.]], [-3,7])
    prediction, probability = f['predict_tree'](duplicate, [[1.]])
    np.testing.assert_array_equal(prediction, [-3])
    np.testing.assert_array_equal(probability, [[.5,.5]])
    positive = f['fit_tree']([[0.], [1.]], [0,1], min_gain=.5)
    assert f['tree_size'](positive) == (1,1,0)


@pytest.mark.parametrize('kwargs', [dict(max_depth=-1), dict(max_depth=17), dict(max_depth=True),
                                   dict(min_leaf=0), dict(min_leaf=3), dict(min_leaf=1.5),
                                   dict(min_gain=-.1), dict(min_gain=np.nan)])
def test_tree_growth_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        reference_functions('decision-trees')['fit_tree']([[0.], [1.]], [0,1], **kwargs)


def test_fixed_tree_pruning_matches_exhaustive_subtree_costs_and_preserves_model():
    f = reference_functions('decision-trees')
    X = np.arange(8.)[:, None]
    y = np.array([0.,0.,1.,1.,3.,3.,2.,2.])
    model = f['fit_tree'](X, y, 'mse', max_depth=4)
    original = copy.deepcopy(model)

    def all_cost_pairs(node):
        choices = [(node['risk'], 1)]
        if 'left' in node:
            for lr, ll in all_cost_pairs(node['left']):
                for rr, rl in all_cost_pairs(node['right']):
                    choices.append((lr+rr, ll+rl))
        return choices

    def leaf_risk(node):
        return node['risk'] if 'left' not in node else leaf_risk(node['left'])+leaf_risk(node['right'])

    choices = all_cost_pairs(model['root'])
    previous_leaves = math.inf
    for alpha in (0., .01, .1, .3, .5, 1., 2.):
        pruned = f['prune_tree'](model, alpha)
        leaves = f['tree_size'](pruned)[1]
        objective = leaf_risk(pruned['root']) + alpha*leaves
        assert objective == pytest.approx(min(risk+alpha*count for risk, count in choices))
        assert leaves <= previous_leaves
        previous_leaves = leaves
    assert f['tree_size'](model) == f['tree_size'](original)
    np.testing.assert_array_equal(f['predict_tree'](model, X)[0], f['predict_tree'](original, X)[0])
    assert model['root'] is not f['prune_tree'](model, 0.)['root']
    for bad in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            f['prune_tree'](model, bad)


def test_tree_pruning_exact_breakpoint_and_nested_global_weights():
    f = reference_functions('decision-trees')
    model = f['fit_tree'](np.arange(4.)[:, None], [0,0,1,1])
    assert f['tree_size'](f['prune_tree'](model, .49))[1] == 2
    assert f['tree_size'](f['prune_tree'](model, .5))[1] == 1
    scaled = copy.deepcopy(model)
    scaled['root']['risk'] *= .2
    assert f['tree_size'](f['prune_tree'](scaled, .099))[1] == 2
    assert f['tree_size'](f['prune_tree'](scaled, .1))[1] == 1


def test_nonlinear_monotone_transform_preserves_training_partition_not_midpoint_query():
    f = reference_functions('decision-trees')
    X, query = np.array([[0.], [2.]]), np.array([[1.2]])
    raw = f['fit_tree'](X, [0,1], max_depth=1)
    transformed = f['fit_tree'](X**3, [0,1], max_depth=1)
    np.testing.assert_array_equal(f['predict_tree'](raw, X)[0], f['predict_tree'](transformed, X**3)[0])
    assert raw['root']['threshold'] == 1. and transformed['root']['threshold'] == 4.
    assert f['predict_tree'](raw, query)[0][0] == 1
    assert f['predict_tree'](transformed, query**3)[0][0] == 0


def test_tree_demonstration_costs_and_multiclass_risks(demonstration_values):
    values = demonstration_values('decision-trees')
    np.testing.assert_allclose(values['demo_split_cost'], 2*values['demo_alpha'])
    np.testing.assert_allclose(values['demo_leaf_cost'], 1+values['demo_alpha'])
    assert 2*1. == 1+1.
    labels = [0]*6+[1]*3+[2]
    impurity = reference_functions('decision-trees')['tree_impurity']
    assert impurity(labels, 'zero_one') == pytest.approx(.4)
    assert impurity(labels, 'gini') == pytest.approx(.54)
    assert impurity(labels, 'entropy') == pytest.approx(.8979457248567798)


@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_ensemble_base_learner_hand_values_and_global_class_columns(task):
    f = reference_functions('tree-ensembles')
    X = np.arange(4.)[:, None]
    y = [0.,0.,2.,2.] if task == 'regression' else [0,0,2,2]
    model = f['ensemble_tree'](X, y, task, max_depth=1, min_leaf=1,
                                n_classes=None if task == 'regression' else 3)
    assert model['root']['threshold'] == 1.5
    outputs = f['ensemble_tree_predict'](model, [[1.5], [2.]])
    if task == 'regression':
        np.testing.assert_array_equal(outputs, [0.,2.])
    else:
        np.testing.assert_array_equal(outputs, [[1,0,0], [0,0,1]])
    expected = (0,) if task == 'regression' else (0, 3)
    assert f['ensemble_tree_predict'](model, np.empty((0, 1))).shape == expected
    with pytest.raises(ValueError):
        f['ensemble_tree_predict'](model, [[0., 1.]])


def test_forest_feature_subsets_are_sampled_at_each_eligible_node():
    f = reference_functions('tree-ensembles')

    class RecordingGenerator(np.random.Generator):
        def __init__(self):
            super().__init__(np.random.PCG64(226))
            self.chosen = []

        def choice(self, *args, **kwargs):
            result = super().choice(*args, **kwargs)
            self.chosen.append(tuple(np.asarray(result).tolist()))
            return result

    rng = RecordingGenerator()
    X = np.repeat(np.arange(8.)[:, None], 4, axis=1)
    tree = f['ensemble_tree'](X, [0.,0.,1.,1.,2.,2.,3.,3.], max_depth=3,
                               min_leaf=1, max_features=1, rng=rng)
    assert len(rng.chosen) >= 3
    assert all(len(group) == 1 for group in rng.chosen)
    assert len(set(rng.chosen)) > 1
    assert tree['root']['feature'] == rng.chosen[0][0]
    with pytest.raises(ValueError):
        f['ensemble_tree'](X, np.arange(8.), max_features=1)


@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_forest_bootstrap_stream_is_reproducible_and_independent_of_node_draws(task):
    f = reference_functions('tree-ensembles')
    X = np.random.default_rng(227).normal(size=(12, 4))
    y = np.arange(12.) if task == 'regression' else np.array([-7,9,100]*4)
    bagged = f['fit_forest'](X, y, task, n_trees=5, min_leaf=1, max_features=4, seed=228)
    sampled = f['fit_forest'](X, y, task, n_trees=5, min_leaf=1, max_features=1, seed=228)
    duplicate = f['fit_forest'](X, y, task, n_trees=5, min_leaf=1, max_features=1, seed=228)
    row_seed = np.random.SeedSequence(228).spawn(2)[0]
    oracle_rng = np.random.default_rng(row_seed)
    for a, b, c in zip(bagged['bags'], sampled['bags'], duplicate['bags']):
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(b, c)
        np.testing.assert_array_equal(a, oracle_rng.integers(0, 12, size=12))
    np.testing.assert_array_equal(f['predict_forest'](sampled, X)[0], f['predict_forest'](duplicate, X)[0])
    assert any(len(np.unique(bag)) < len(bag) for bag in bagged['bags'])


@pytest.mark.parametrize('task', ['regression', 'classification'])
@pytest.mark.parametrize('voting', ['soft', 'hard'])
def test_forest_oob_values_match_independent_index_contributor_audit(task, voting):
    f = reference_functions('tree-ensembles')
    X = np.array([[0,0], [1,0], [2,1], [3,1], [4,2], [5,2]], dtype=float)
    y = np.array([0.,1.,2.,4.,2.,0.]) if task == 'regression' else np.array([-7,-7,9,9,100,100])
    model = f['fit_forest'](X, y, task, n_trees=3, max_depth=2, min_leaf=1, max_features=1, seed=229)
    values, counts = f['oob_predictions'](model, X, voting)
    for i in range(len(X)):
        eligible = [b for b, bag in enumerate(model['bags']) if i not in list(bag)]
        assert counts[i] == len(eligible)
        if not eligible:
            assert np.isnan(values[i]).all()
            continue
        outputs = []
        for b in eligible:
            node = model['trees'][b]['root']
            while 'left' in node:
                node = node['left'] if X[i, node['feature']] <= node['threshold'] else node['right']
            value = np.array(node['value'], copy=True)
            if task == 'classification' and voting == 'hard':
                value = np.eye(3)[int(np.argmax(value))]
            outputs.append(value)
        expected = sum(outputs)/len(outputs)
        np.testing.assert_allclose(values[i], expected)
    with pytest.raises(ValueError):
        f['oob_predictions'](model, X[:-1], voting)


def test_forest_missing_class_bags_keep_global_labels_and_missing_oob_is_nan():
    f = reference_functions('tree-ensembles')
    X, y = np.array([[0.], [1.]]), np.array([-7, 9])
    model = f['fit_forest'](X, y, 'classification', n_trees=12, min_leaf=1, seed=225)
    missing = [b for b in model['bags'] if len(np.unique(y[b])) == 1]
    assert missing
    np.testing.assert_array_equal(model['classes'], [-7,9])
    for tree in model['trees']:
        assert f['ensemble_tree_predict'](tree, X).shape == (2, 2)
    assert f['predict_forest'](model, np.empty((0, 1)))[1].shape == (0, 2)
    uncovered = copy.deepcopy(model)
    uncovered['bags'] = [np.array([0,1]) for _ in model['trees']]
    values, counts = f['oob_predictions'](uncovered, X)
    assert np.isnan(values).all() and np.all(counts == 0)


def test_forest_hard_and_soft_voting_are_distinct_and_ties_choose_smallest_label():
    f = reference_functions('tree-ensembles')
    trees = [dict(task='classification', n_features=1, n_classes=2, root=dict(value=np.array([1-p, p])))
             for p in (.51, .51, 0.)]
    model = dict(trees=trees, classes=np.array([-7,9]), task='classification')
    soft, p = f['predict_forest'](model, [[0.]])
    hard, votes = f['predict_forest'](model, [[0.]], 'hard')
    assert soft[0] == -7 and hard[0] == 9
    np.testing.assert_allclose(p, [[.66,.34]])
    np.testing.assert_allclose(votes, [[1/3,2/3]])
    model['trees'] = [trees[0], trees[2]]
    assert f['predict_forest'](model, [[0.]], 'hard')[0][0] == -7
    with pytest.raises(ValueError):
        f['predict_forest'](model, [[0.]], 'unrecognized')


@pytest.mark.parametrize('loss', ['squared', 'logistic'])
@pytest.mark.parametrize('n', [1, 5])
def test_boosting_gradients_against_independent_mean_objective(loss, n):
    evaluate = reference_functions('tree-ensembles')['boost_objective']
    y = np.linspace(-.3, 1.2, n) if loss == 'squared' else (np.arange(n) % 2).astype(float)
    scores = np.linspace(-.7, 1.1, n)

    def independent(candidate):
        if loss == 'squared':
            return sum((float(a)-float(b))**2/2 for a, b in zip(y, candidate))/n
        return sum(math.log1p(math.exp(-float(s) if label else float(s)))
                   for label, s in zip(y, candidate))/n

    value, residual = evaluate(y, scores, loss)
    assert value == pytest.approx(independent(scores))
    np.testing.assert_allclose(-residual/n, central_difference(independent, scores), rtol=1e-5, atol=1e-9)
    repeated_value, repeated_residual = evaluate(np.repeat(y, 3), np.repeat(scores, 3), loss)
    assert repeated_value == pytest.approx(value)
    np.testing.assert_allclose(repeated_residual, np.repeat(residual, 3))


def test_logistic_boost_objective_extreme_logits_and_directional_derivative():
    evaluate = reference_functions('tree-ensembles')['boost_objective']
    value, residual = evaluate([0,1,0,1], [1000.,-1000.,-1000.,1000.], 'logistic')
    assert value == 500.
    np.testing.assert_array_equal(residual, [-1,1,0,0])
    value, residual = evaluate([0,1], [0.,0.], 'logistic')
    assert value == pytest.approx(math.log(2))
    gradient = -residual/2
    np.testing.assert_array_equal(gradient, [.25,-.25])
    h, step = np.array([-1.,1.]), 1e-5
    independent = lambda score: (math.log1p(math.exp(score[0]))+math.log1p(math.exp(-score[1])))/2
    derivative = (independent(step*h)-independent(-step*h))/(2*step)
    assert derivative == pytest.approx(-.5, abs=1e-9)
    assert gradient @ h == -.5


@pytest.mark.parametrize('loss', ['squared', 'logistic'])
def test_gradient_boosting_one_step_against_hand_states(loss):
    f = reference_functions('tree-ensembles')
    X = np.arange(4.)[:, None]
    y, rate = ([0.,0.,2.,2.], .5) if loss == 'squared' else ([0,0,1,1], 1.)
    model = f['fit_gradient_boost'](X, y, loss, n_stages=1, learning_rate=rate, max_depth=1, min_leaf=1)
    scores = f['staged_boost_scores'](model, X)
    if loss == 'squared':
        np.testing.assert_allclose(scores, [[1.,.5], [1.,.5], [1.,1.5], [1.,1.5]])
        np.testing.assert_allclose(model['training_loss'], [.5,.125])
    else:
        np.testing.assert_allclose(scores, [[0.,-.5], [0.,-.5], [0.,.5], [0.,.5]])
        np.testing.assert_allclose(model['training_loss'], [math.log(2), math.log1p(math.exp(-.5))])
    assert all(tree['task'] == 'regression' for tree in model['trees'])
    assert f['staged_boost_scores'](model, np.empty((0, 1))).shape == (0, 2)


@pytest.mark.parametrize('loss', ['squared', 'logistic'])
def test_boosting_stump_trace_against_independent_residual_means_and_candidate_costs(loss):
    f = reference_functions('tree-ensembles')
    X = np.arange(6.)[:, None]
    y = np.array([0.,0.,2.,1.,3.,3.]) if loss == 'squared' else np.array([0,0,1,0,1,1])
    rate = .3
    model = f['fit_gradient_boost'](X, y, loss, n_stages=3, learning_rate=rate, max_depth=1, min_leaf=1)
    actual = f['staged_boost_scores'](model, X)
    initial = y.mean() if loss == 'squared' else math.log(y.mean()/(1-y.mean()))
    expected = np.full(len(y), initial)
    np.testing.assert_allclose(actual[:, 0], expected)
    for m, tree in enumerate(model['trees'], start=1):
        residual = y-expected if loss == 'squared' else np.array([label-1/(1+math.exp(-score)) for label, score in zip(y, expected)])
        costs = []
        for threshold in np.arange(.5, 5.5):
            left = X[:, 0] <= threshold
            guess = np.where(left, residual[left].mean(), residual[~left].mean())
            costs.append(sum((float(a)-float(b))**2 for a,b in zip(residual, guess))/len(y))
        root = tree['root']
        if 'left' in root:
            left = X[:, 0] <= root['threshold']
            direction = np.where(left, residual[left].mean(), residual[~left].mean())
            np.testing.assert_allclose(root['left']['value'], residual[left].mean())
            np.testing.assert_allclose(root['right']['value'], residual[~left].mean())
            assert np.mean((residual-direction)**2) == pytest.approx(min(costs), abs=1e-14)
        else:
            direction = np.full(len(y), residual.mean())
        expected += rate*direction
        np.testing.assert_allclose(actual[:, m], expected, atol=1e-14)
        if loss == 'squared':
            objective = sum((float(a)-float(b))**2/2 for a,b in zip(y, expected))/len(y)
        else:
            objective = sum(math.log1p(math.exp(-float(s) if label else float(s))) for label,s in zip(y,expected))/len(y)
        assert model['training_loss'][m] == pytest.approx(objective)


@pytest.mark.parametrize('kwargs', [dict(n_stages=0), dict(n_stages=65), dict(n_stages=True),
                                   dict(learning_rate=0.), dict(learning_rate=1.1),
                                   dict(learning_rate=np.nan), dict(max_depth=17), dict(min_leaf=0)])
def test_boosting_rejects_invalid_work_controls(kwargs):
    with pytest.raises(ValueError):
        reference_functions('tree-ensembles')['fit_gradient_boost']([[0.], [1.]], [0.,1.], **kwargs)


def test_ensemble_fitting_rejects_invalid_labels_and_limits():
    f = reference_functions('tree-ensembles')
    for count in (0,65,True):
        with pytest.raises(ValueError):
            f['fit_forest']([[0.], [1.]], [0.,1.], n_trees=count)
    with pytest.raises(ValueError):
        f['fit_forest']([[0.], [1.]], [0.,1.], 'classification')
    with pytest.raises(ValueError):
        f['fit_gradient_boost']([[0.], [1.]], [1,1], 'logistic')
    with pytest.raises(ValueError):
        f['fit_gradient_boost']([[0.], [1.]], [0,2], 'logistic')
    with pytest.raises(ValueError):
        f['boost_objective']([0,2], [0.,0.], 'logistic')


def test_ensemble_analytical_demo_variance_and_oob_formula(demonstration_values):
    values = demonstration_values('tree-ensembles')
    B = values['demo_trees']
    for rho, variance in values['demo_variance'].items():
        np.testing.assert_allclose(variance, (B*4+B*(B-1)*rho*4)/B**2)
        assert variance[4] == pytest.approx(4*(rho+(1-rho)/5))
    q = (79/80)**80
    assert values['demo_absent'] == pytest.approx(q)
    np.testing.assert_allclose(values['demo_coverage'], 1-(1-q)**B)
    assert 4*(.25+.75/5) == pytest.approx(1.6)
    assert 4*(1-(3/4)**4) == 175/64
    assert (175/256)**3 == pytest.approx(.31944364309310913)


def test_exponential_loss_stage_against_independent_derivative_and_reweighting():
    error = .25
    beta = .5*math.log((1-error)/error)
    objective = lambda b: (1-error)*math.exp(-float(b[0]))+error*math.exp(float(b[0]))
    assert beta == pytest.approx(.5*math.log(3))
    assert objective([beta]) == pytest.approx(math.sqrt(3)/2)
    assert error*math.exp(beta)/objective([beta]) == pytest.approx(.5)
    np.testing.assert_allclose(central_difference(objective, [beta]), [0.], atol=1e-9)


@pytest.mark.parametrize('C', [.1, 1.3])
@pytest.mark.parametrize('n', [1, 5])
def test_hinge_primal_subgradient_against_independent_smooth_snapshot(C, n):
    evaluate = reference_functions('support-vector-machines-and-kernels')['hinge_primal']
    X = np.linspace(-.6, .8, n*2).reshape(n, 2)
    y = np.where(np.arange(n) % 2, 1., -1.)
    theta = np.array([.2, -.3, .1])
    assert np.all(np.abs(y*(X@theta[:2]+theta[2])-1) > .1)

    def independent(candidate):
        squared = sum(float(w)**2 for w in candidate[:2])/2
        hinge = sum(max(0., 1-float(label)*(float(row[0])*candidate[0]+float(row[1])*candidate[1]+candidate[2]))
                    for label, row in zip(y, X))
        return squared+C*hinge

    objective, dw, db = evaluate(X, y, theta[:2], theta[2], C)
    assert objective == pytest.approx(independent(theta))
    np.testing.assert_allclose(np.r_[dw, db], central_difference(independent, theta), rtol=1e-6, atol=1e-9)
    repeated = evaluate(np.repeat(X, 2, axis=0), np.repeat(y, 2), theta[:2], theta[2], C/2)
    np.testing.assert_allclose([objective, *dw, db], [repeated[0], *repeated[1], repeated[2]])


def test_hinge_kink_selected_subgradient_differs_from_symmetric_difference():
    evaluate = reference_functions('support-vector-machines-and-kernels')['hinge_primal']
    value, dw, db = evaluate([[1.]], [1], [1.], 0., 1.)
    assert value == .5 and dw[0] == 1. and db == 0.
    independent = lambda candidate: .5*float(candidate[0])**2+max(0., 1-float(candidate[0]))
    assert central_difference(independent, [1.])[0] == pytest.approx(.5, abs=1e-8)
    for query in (-2., .3, 1., 1.4, 3.):
        assert independent([query]) >= value+dw[0]*(query-1)-1e-14


@pytest.mark.parametrize('kind', ['linear', 'polynomial', 'rbf'])
def test_kernel_matrices_against_independent_features_and_scalar_distances(kind):
    kernel = reference_functions('support-vector-machines-and-kernels')['kernel_matrix']
    X = np.array([[-1., .2], [.3, .7], [1.1, -.4]])
    Z = np.array([[.5, -.6], [.2, .9]])
    actual = kernel(X, Z, kind)
    expected = np.empty((3, 2))
    for i, x in enumerate(X):
        for j, z in enumerate(Z):
            dot = sum(float(a)*float(b) for a,b in zip(x,z))
            expected[i,j] = dot if kind == 'linear' else ((1+dot)**2 if kind == 'polynomial'
                                                          else math.exp(-sum((float(a)-float(b))**2 for a,b in zip(x,z))))
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    gram = kernel(X, X, kind)
    np.testing.assert_allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12
    assert kernel(np.empty((0, 2)), Z, kind).shape == (0, 2)
    if kind == 'polynomial':
        features = lambda A: np.column_stack([np.ones(len(A)), np.sqrt(2)*A[:,0], np.sqrt(2)*A[:,1],
                                              A[:,0]**2, np.sqrt(2)*A[:,0]*A[:,1], A[:,1]**2])
        np.testing.assert_allclose(actual, features(X)@features(Z).T, rtol=1e-13)


@pytest.mark.parametrize('kwargs', [dict(gamma=0.), dict(gamma=np.nan), dict(offset=-1.),
                                   dict(degree=True), dict(degree=1.5), dict(degree=5), dict(kind='unknown')])
def test_kernel_invalid_constructions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        reference_functions('support-vector-machines-and-kernels')['kernel_matrix']([[0.]], [[1.]], **kwargs)


def test_rbf_direct_distances_preserve_large_offset_and_indefinite_similarity_is_rejected():
    f = reference_functions('support-vector-machines-and-kernels')
    X = np.array([[1e12], [1e12+1]])
    np.testing.assert_allclose(f['kernel_matrix'](X, X, 'rbf'), [[1, math.exp(-1)], [math.exp(-1), 1]])
    invalid = np.array([[1., 2.], [2., 1.]])
    assert np.array([1.,-1.])@invalid@np.array([1.,-1.]) == -2.
    with pytest.raises(ValueError):
        f['fit_svm_dual'](invalid, [-1,1])


@pytest.mark.parametrize('C', [.1, 1., 3.])
@pytest.mark.parametrize('values', [[.8, .2], [-2., 4.], [-2., -3.]])
def test_dual_projection_two_coordinates_matches_exact_scalar_solution(C, values):
    project = reference_functions('support-vector-machines-and-kernels')['project_svm_dual']
    actual = project(values, [-1,1], C)
    expected = min(C, max(0., sum(values)/2))
    np.testing.assert_allclose(actual, [expected, expected], atol=1e-10)
    assert np.all(actual >= 0) and np.all(actual <= C)
    assert abs(np.array([-1.,1.])@actual) <= 1e-9


def test_dual_projection_equality_active_bounds_and_interior_plane_projection():
    project = reference_functions('support-vector-machines-and-kernels')['project_svm_dual']
    np.testing.assert_allclose(project([.7,.2,.8],[-1,-1,1],1.), [2/3,1/6,5/6], atol=1e-10)
    np.testing.assert_allclose(project([2.,-1.,.3],[-1,-1,1],1.), [1,0,1], atol=1e-10)
    np.testing.assert_allclose(project([1.,0.,0.],[1,1,-1],1.), [.5,0,.5], atol=1e-10)
    assert np.array([1,1,-1])@np.clip([1.,0.,0.],0,1) == 1.
    for labels in ([1,1], [0,1]):
        with pytest.raises(ValueError):
            project([.3,.4], labels, 1.)


@pytest.mark.parametrize('C', [.1, .5, 1., 10.])
def test_svm_two_point_optimum_and_primal_dual_certificate(C):
    f = reference_functions('support-vector-machines-and-kernels')
    X, y = np.array([[-1.], [1.]]), np.array([-1,1])
    model = f['fit_kernel_svm'](X, y, C, tolerance=1e-9)
    a = min(C, .5); w = 2*a
    expected = w*w/2+C*2*max(0., 1-w)
    assert model['converged']
    np.testing.assert_allclose(model['alpha'], a, atol=1e-9)
    assert model['bias'] == pytest.approx(0., abs=1e-10)
    assert model['primal'] == pytest.approx(expected)
    assert model['dual'] == pytest.approx(2*a-2*a*a)
    assert abs(model['gap']) < 1e-9 and model['kkt_violation'] < 1e-9
    assert np.all(np.diff(model['history'][:, 2]) >= -1e-12)
    np.testing.assert_allclose(model['margins'], w)
    np.testing.assert_allclose(model['train_X'][:,0]@model['coefficients'], w)
    predicted, scores = f['predict_kernel_svm'](model, [[-2.], [0.], [2.]], query_block=1)
    np.testing.assert_array_equal(predicted, [-1,1,1])
    np.testing.assert_allclose(scores, [-2*w,0,2*w])


def test_svm_budget_exhaustion_is_explicit_and_more_work_checks_kkt():
    f = reference_functions('support-vector-machines-and-kernels')
    X = np.array([[-2.],[-1.],[1.],[2.]])
    y = np.array([-1,-1,1,1])
    short = f['fit_kernel_svm'](X, y, max_steps=1, tolerance=1e-8)
    assert short['iterations'] == 1 and not short['converged'] and short['gap'] > .1
    model = f['fit_kernel_svm'](X, y, max_steps=2000, tolerance=1e-7)
    assert model['converged'] and model['gap'] <= 1e-7*max(1.,model['primal'])
    assert abs(y@model['alpha']) <= 1e-9
    w = X.T@model['coefficients']
    margins = y*(X@w+model['bias'])
    primal = float(w@w)/2+sum(max(0.,1-float(m)) for m in margins)
    assert model['primal'] == pytest.approx(primal)
    for a, margin in zip(model['alpha'], margins):
        if a <= 1e-7:
            assert margin >= 1-1e-4
        elif a >= 1-1e-7:
            assert margin <= 1+1e-4
        else:
            assert margin == pytest.approx(1., abs=1e-4)
    assert np.all(np.diff(model['history'][:, 2]) >= -1e-10)


def test_svm_zero_gram_recovers_majority_bias_and_bound_interval_diagnostic():
    f = reference_functions('support-vector-machines-and-kernels')
    model = f['fit_svm_dual'](np.zeros((3,3)), [-1,-1,1], C=1.)
    assert model['converged']
    np.testing.assert_allclose(model['alpha'], [.5,.5,1.], atol=1e-9)
    assert model['bias'] == pytest.approx(-1.)
    assert model['primal'] == pytest.approx(2.) and model['dual'] == pytest.approx(2.)
    unfinished = f['svm_diagnostics'](np.eye(2), [-1,1], np.zeros(2), 1.)
    assert not unfinished['bias_interval_valid'] and unfinished['gap'] == 2.


def test_polynomial_svm_exact_xor_feature_solution_and_query_blocks():
    f = reference_functions('support-vector-machines-and-kernels')
    X = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    y = np.array([1,-1,-1,1])
    original = X.copy()
    model = f['fit_kernel_svm'](X, y, kind='polynomial', tolerance=1e-9)
    assert model['converged']
    np.testing.assert_allclose(model['alpha'], .125, atol=1e-9)
    assert model['primal'] == pytest.approx(.25) and model['dual'] == pytest.approx(.25)
    query = np.array([[-2.,.3], [.7,.9], [0.,0.]])
    labels, scores = f['predict_kernel_svm'](model, query, query_block=1)
    np.testing.assert_allclose(scores, query[:,0]*query[:,1], atol=1e-9)
    np.testing.assert_allclose(f['predict_kernel_svm'](model, query, query_block=10)[1], scores)
    assert f['predict_kernel_svm'](model, np.empty((0, 2)))[1].shape == (0,)
    X[:] = 99.
    np.testing.assert_array_equal(model['train_X'], original)
    with pytest.raises(ValueError):
        f['predict_kernel_svm'](model, query, query_block=0)


def test_svm_observation_duplication_requires_adjusting_summed_C():
    f = reference_functions('support-vector-machines-and-kernels')
    X, y = np.array([[-1.],[1.]]), np.array([-1,1])
    base = f['fit_kernel_svm'](X, y, C=.1)
    duplicated_X, duplicated_y = np.repeat(X, 2, axis=0), np.repeat(y, 2)
    same_objective = f['fit_kernel_svm'](duplicated_X, duplicated_y, C=.05)
    changed_objective = f['fit_kernel_svm'](duplicated_X, duplicated_y, C=.1)
    slopes = [float(m['train_X'][:,0]@m['coefficients']) for m in (base,same_objective,changed_objective)]
    np.testing.assert_allclose(slopes, [.2,.2,.4], atol=1e-9)
    assert same_objective['primal'] == pytest.approx(base['primal'])
    assert changed_objective['primal'] == pytest.approx(.32)


@pytest.mark.parametrize('kwargs', [dict(C=0.),dict(C=np.inf),dict(tolerance=0.),dict(max_steps=0),
                                   dict(max_steps=True),dict(max_steps=10001)])
def test_svm_rejects_invalid_optimization_controls(kwargs):
    with pytest.raises(ValueError):
        reference_functions('support-vector-machines-and-kernels')['fit_svm_dual'](np.eye(2), [-1,1], **kwargs)


def test_svm_rejects_wrong_gram_shapes_asymmetry_and_unsigned_labels():
    solve = reference_functions('support-vector-machines-and-kernels')['fit_svm_dual']
    for gram, labels in ((np.ones((2,3)),[-1,1]), ([[1.,.3],[.1,1.]],[-1,1]),
                         (np.eye(2),[0,1]),(np.eye(2),[1,1]),(np.eye(129),[-1]*64+[1]*65)):
        with pytest.raises(ValueError):
            solve(gram, labels)


def test_svm_demo_hinge_upper_bound_and_analytic_geometry(demonstration_values):
    values = demonstration_values('support-vector-machines-and-kernels')
    np.testing.assert_allclose(values['demo_hinge'], np.maximum(0,1-values['demo_margin']))
    assert np.all(values['demo_hinge'] >= values['demo_zero_one'])
    assert np.dot([3,4],[1,1])-1 == 6
    assert 6/np.linalg.norm([3,4]) == 12/np.linalg.norm([6,8])
