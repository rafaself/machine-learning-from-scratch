"""Check displayed implementations and worked examples against independent math."""

import ast
import copy
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
