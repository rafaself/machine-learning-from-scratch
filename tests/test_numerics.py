"""Check displayed implementations and worked examples against independent math."""

import ast
import copy
import re
from functools import lru_cache
from pathlib import Path

import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def reference_functions(name):
    """Load definitions only; never run experiments or learner answer cells."""
    paths = list((ROOT / "notebooks").glob(f"*/{name}.ipynb"))
    assert len(paths) == 1
    notebook = nbformat.read(paths[0], as_version=4)
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
            exec(compile(tree, f"{paths[0].name}:{cell.id}", "exec"), namespace)
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
        path, = (ROOT / "notebooks").glob(f"*/{name}.ipynb")
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
