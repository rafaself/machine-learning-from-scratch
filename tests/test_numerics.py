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
