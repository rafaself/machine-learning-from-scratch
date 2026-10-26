"""Independent categorical calculus, manual network and evaluation checks."""

import copy
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("multiclass-learning-objectives")


def scalar_log_probabilities(row, temperature=1.):
    shifted = [(float(value) - max(float(v) for v in row)) / temperature for value in row]
    normalizer = math.log(sum(math.exp(v) for v in shifted))
    return [v - normalizer for v in shifted]


def scalar_categorical(logits, targets, weights, temperature):
    return -sum(float(w) * sum(float(q) * lp for q, lp in zip(target, scalar_log_probabilities(row, temperature)))
                for row, target, w in zip(logits, targets, weights)) / sum(float(w) for w in weights)


def scalar_network(parameters, x, q, weights, penalty, temperature):
    logits = []
    for row in x:
        if parameters["kind"] == "linear":
            logits.append([sum(float(row[j]) * float(parameters["W"][j, k]) for j in range(len(row)))
                           + float(parameters["b"][k]) for k in range(q.shape[1])])
        else:
            hidden = [math.tanh(sum(float(row[j]) * float(parameters["W1"][j, h]) for j in range(len(row)))
                                + float(parameters["b1"][h])) for h in range(len(parameters["b1"]))]
            logits.append([sum(hidden[h] * float(parameters["W2"][h, k]) for h in range(len(hidden)))
                           + float(parameters["b2"][k]) for k in range(q.shape[1])])
    squared = sum(float(value) ** 2 for key, array in parameters.items() if key.startswith("W") for value in array.ravel())
    return scalar_categorical(logits, q, weights, temperature) + penalty * squared / 2


@pytest.mark.parametrize("temperature", [.5, 1., 3.])
@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (5, 4)])
def test_log_softmax_and_reverse_product_against_independent_scalar_difference(temperature, shape):
    f = functions(); rng = np.random.default_rng(470)
    logits, upstream = rng.normal(size=shape), rng.normal(size=shape)
    before = logits.copy(), upstream.copy()
    expected = np.array([scalar_log_probabilities(row, temperature) for row in logits])
    np.testing.assert_allclose(f["mc_log_probabilities"](logits, temperature), expected, atol=2e-14)

    def objective(candidate):
        matrix = candidate.reshape(shape)
        return sum(float(upstream[i, j]) * math.exp(scalar_log_probabilities(matrix[i], temperature)[j])
                   for i in range(shape[0]) for j in range(shape[1]))

    actual = f["mc_softmax_vjp"](logits, upstream, temperature)
    np.testing.assert_allclose(actual.ravel(), central_difference(objective, logits.ravel()), atol=2e-9, rtol=3e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 0., atol=2e-14)
    np.testing.assert_array_equal(logits, before[0]); np.testing.assert_array_equal(upstream, before[1])


def test_hand_rational_probabilities_targets_gradients_and_jacobian():
    f = functions(); z = np.array([[0., math.log(2), math.log(3)]])
    p = np.array([1 / 6, 1 / 3, 1 / 2])
    jacobian = np.array([[5 / 36, -1 / 18, -1 / 12], [-1 / 18, 2 / 9, -1 / 6], [-1 / 12, -1 / 6, 1 / 4]])
    for epsilon, expected_gradient in [(0., [1 / 6, -2 / 3, 1 / 2]), (.3, [1 / 15, -7 / 15, 2 / 5])]:
        target = f["mc_targets"]([20], [10, 20, 40], epsilon)
        result = f["mc_categorical_objective"](z, target)
        np.testing.assert_allclose(result["probability"][0], p, atol=2e-15)
        np.testing.assert_allclose(result["gradient"][0], expected_gradient, atol=2e-15)
        assert result["loss"] == pytest.approx(result["entropy"] + result["kl"], abs=2e-14)
        for j in range(3):
            direction = np.eye(3)[j][None, :]
            np.testing.assert_allclose(f["mc_logit_hvp"](z, direction)[0], jacobian[:, j], atol=2e-15)
    assert f["mc_categorical_objective"](z, [[0., 1., 0.]])["loss"] == pytest.approx(math.log(3))


@pytest.mark.parametrize("temperature", [.7, 1., 2.3])
def test_weighted_soft_target_logits_log_temperature_and_hessian(temperature):
    f = functions(); z = np.array([[-.2, .4, .1], [.8, -.3, .2], [.4, .5, -.1]])
    q = np.array([[.2, .5, .3], [.7, .1, .2], [0., 0., 0.]])
    weights = np.array([1., 3., 0.]); direction = np.array([[.3, -.2, .7], [-.8, .1, .2], [1., 2., 3.]])
    result = f["mc_categorical_objective"](z, q, weights, temperature)
    objective = lambda candidate: scalar_categorical(candidate.reshape(z.shape), q, weights, temperature)
    assert result["loss"] == pytest.approx(objective(z.ravel()), abs=1e-14)
    np.testing.assert_allclose(result["gradient"].ravel(), central_difference(objective, z.ravel()), rtol=2e-7, atol=2e-9)
    temperature_objective = lambda log_t: scalar_categorical(z, q, weights, math.exp(float(log_t[0])))
    assert result["log_temperature_gradient"] == pytest.approx(central_difference(temperature_objective, [math.log(temperature)])[0], rel=2e-7, abs=2e-9)
    expected = []
    for row, vector, weight in zip(z, direction, weights):
        p = np.exp(scalar_log_probabilities(row, temperature))
        dense = (np.diag(p) - np.outer(p, p)) * weight / sum(weights) / temperature ** 2
        assert np.linalg.eigvalsh(dense).min() >= -1e-14
        expected.append(dense @ vector)
    actual = f["mc_logit_hvp"](z, direction, weights, temperature)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_array_equal(actual[-1], [0., 0., 0.])
    np.testing.assert_allclose(f["mc_logit_hvp"](z, np.ones_like(z), weights, temperature), 0., atol=2e-14)
    assert float(np.sum(actual * direction)) >= -1e-13


def test_common_offsets_temperature_ordering_and_probability_underflow():
    f = functions(); z = np.array([[0., 1., -1.], [4., -.5, .2]])
    np.testing.assert_allclose(f["mc_log_probabilities"](z + 1000), f["mc_log_probabilities"](z), atol=2e-13)
    np.testing.assert_allclose(f["mc_log_probabilities"]([[1e308, 1e308]], .1), [[-math.log(2)] * 2])
    result = f["mc_categorical_objective"]([[-1000., 1000.]], [[1., 0.]])
    assert result["probability"][0, 0] == 0. and result["loss"] == 2000.
    np.testing.assert_array_equal(result["gradient"], [[-1., 1.]])
    assert np.isfinite(result["log_probability"]).all()
    with pytest.raises(FloatingPointError):
        f["mc_log_probabilities"]([[-1e308, 1e308]])


def test_one_class_empty_query_scoring_and_admitted_target_roundoff_null_direction():
    f = functions()
    assert f["mc_log_probabilities"](np.empty((0, 3))).shape == (0, 3)
    assert f["mc_softmax_vjp"](np.empty((0, 3)), np.empty((0, 3))).shape == (0, 3)
    for q in ([[1.]], [[1. + 5e-13]]):
        result = f["mc_categorical_objective"]([[1234.]], q)
        assert result["loss"] == 0. and result["log_temperature_gradient"] == 0.
        np.testing.assert_array_equal(result["gradient"], [[0.]])
        np.testing.assert_array_equal(f["mc_logit_hvp"]([[1234.]], [[9.]]), [[0.]])


@pytest.mark.parametrize("temperature", [0., -1., np.inf, np.nan, True, [1.]])
def test_invalid_temperature_is_not_clipped(temperature):
    with pytest.raises(ValueError):
        functions()["mc_log_probabilities"]([[0., 1.]], temperature)


@pytest.mark.parametrize("bad", [np.empty((0, 0)), np.zeros((2, 33)), np.zeros((2001, 2)), [[np.nan]], [0., 1.]])
def test_logit_shape_and_work_limits(bad):
    with pytest.raises(ValueError):
        functions()["mc_log_probabilities"](bad)


def test_vocabulary_alignment_masked_unknown_labels_and_column_permutation():
    f = functions(); labels = np.array([40, 10, -1]); classes = np.array([10, 20, 40]); weights = [2., 1., 0.]
    q = f["mc_targets"](labels, classes, .3, weights)
    np.testing.assert_allclose(q, [[.1, .1, .8], [.8, .1, .1], [0., 0., 0.]])
    z = np.array([[.3, -.2, .8], [.1, .5, -.1], [99., -99., 0.]])
    first = f["mc_categorical_objective"](z, q, weights)
    permutation = [2, 0, 1]
    changed_targets = f["mc_targets"](labels, classes[permutation], .3, weights)
    changed = f["mc_categorical_objective"](z[:, permutation], changed_targets, weights)
    assert first["loss"] == pytest.approx(changed["loss"], abs=1e-14)
    np.testing.assert_allclose(first["gradient"][:, permutation], changed["gradient"], atol=2e-15)
    with pytest.raises(ValueError, match="active label"):
        f["mc_targets"]([40, 10, -1], classes)


@pytest.mark.parametrize("labels,classes", [([[10]], [10, 20]), ([10.], [10, 20]), ([10], [10, 10]), ([10], [[10, 20]]), ([10], []), ([10], list(range(33)))])
def test_class_and_label_contract(labels, classes):
    with pytest.raises(ValueError):
        functions()["mc_targets"](labels, classes)


@pytest.mark.parametrize("epsilon", [-.1, 1.1, np.nan, True, [0.]])
def test_smoothing_contract(epsilon):
    with pytest.raises(ValueError):
        functions()["mc_targets"]([0], [0, 1], epsilon)


def test_weight_scale_invariance_large_totals_and_relative_underflow():
    f = functions(); z = np.array([[.1, .3], [.5, -.2]]); q = np.array([[1., 0.], [0., 1.]])
    ordinary = f["mc_categorical_objective"](z, q, [1., 1.])
    large = f["mc_categorical_objective"](z, q, [1e308, 1e308])
    assert large["loss"] == ordinary["loss"]
    np.testing.assert_array_equal(large["gradient"], ordinary["gradient"])
    with pytest.raises(FloatingPointError):
        f["mc_weights"](2, [1e308, 1e-308])


@pytest.mark.parametrize("weights", [[0., 0.], [-1., 2.], [np.nan, 1.], [np.inf, 1.], [[1., 2.]], [1.]])
def test_weight_contract_and_undefined_all_masked_mean(weights):
    with pytest.raises(ValueError):
        functions()["mc_weights"](2, weights)


@pytest.mark.parametrize("q,weights", [([[.2, .2]], None), ([[-.1, 1.1]], None), ([[1., 0.], [1., 0.]], [1., 0.]), ([[1., 0.], [np.nan, 0.]], [1., 0.])])
def test_invalid_target_mass_and_ignored_nan_targets(q, weights):
    with pytest.raises(ValueError):
        functions()["mc_categorical_objective"](np.zeros(np.shape(q)), q, weights)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("epsilon", [0., .2])
@pytest.mark.parametrize("penalty", [0., .09])
def test_every_parameter_and_input_against_separate_scalar_network(kind, n, epsilon, penalty):
    f = functions(); x = np.linspace(-.5, .7, n * 2).reshape(n, 2)
    parameters = f["mc_initialize"](kind, 2, 3, seed=477, width=3)
    weights = np.arange(1., n + 1)
    if n > 1:
        weights[-1] = 0.
    q = np.zeros((n, 3))
    for i in range(n):
        if weights[i]:
            q[i] = epsilon / 3; q[i, i % 3] += 1 - epsilon
    before = copy.deepcopy(parameters); before_x = x.copy(); temperature = .7
    result = f["mc_network_objective"](parameters, x, q, weights, penalty, temperature)
    assert result["loss"] == pytest.approx(scalar_network(parameters, x, q, weights, penalty, temperature), abs=2e-14)
    for key in f["mc_parameter_fields"](parameters):
        def objective(candidate):
            trial = copy.deepcopy(parameters); trial[key] = candidate.reshape(parameters[key].shape)
            return scalar_network(trial, x, q, weights, penalty, temperature)
        expected = central_difference(objective, parameters[key].ravel()).reshape(parameters[key].shape)
        np.testing.assert_allclose(result["gradients"][key], expected, rtol=3e-6, atol=3e-8)
        np.testing.assert_array_equal(parameters[key], before[key])
    input_objective = lambda candidate: scalar_network(parameters, candidate.reshape(x.shape), q, weights, penalty, temperature)
    np.testing.assert_allclose(result["input_gradient"].ravel(), central_difference(input_objective, x.ravel()), rtol=3e-6, atol=3e-8)
    np.testing.assert_array_equal(x, before_x)
    if n > 1:
        np.testing.assert_array_equal(result["input_gradient"][-1], np.zeros(2))
    repeated = f["mc_network_objective"](parameters, np.tile(x, (3, 1)), np.tile(q, (3, 1)), np.tile(weights, 3), penalty, temperature)
    assert repeated["loss"] == pytest.approx(result["loss"], abs=2e-14)
    for key in result["gradients"]:
        np.testing.assert_allclose(repeated["gradients"][key], result["gradients"][key], atol=2e-14)


def test_bias_regularization_and_input_sensitivity_do_not_gain_extra_penalties():
    f = functions(); p = f["mc_initialize"]("mlp", 2, 3, width=3)
    x = [[.3, -.2], [.1, .7]]; q = [[1., 0., 0.], [0., 1., 0.]]
    base = f["mc_network_objective"](p, x, q)
    penalized = f["mc_network_objective"](p, x, q, penalty=.3)
    for key in base["gradients"]:
        expected = .3 * p[key] if key.startswith("W") else np.zeros_like(p[key])
        np.testing.assert_allclose(penalized["gradients"][key] - base["gradients"][key], expected, atol=2e-14)
    np.testing.assert_array_equal(penalized["input_gradient"], base["input_gradient"])


def test_two_logit_binary_likelihood_and_minimum_norm_penalty_mapping():
    f = functions(); x = np.array([[-1.], [.2], [.8]]); labels = np.array([0, 1, 1]); weight = .7; bias = -.2; penalty = .4
    p = dict(kind="linear", W=np.array([[-weight / 2, weight / 2]]), b=np.array([-bias / 2, bias / 2]))
    q = np.eye(2)[labels]
    result = f["mc_network_objective"](p, x, q, penalty=penalty)
    scalar_logits = [float(row[0]) * weight + bias for row in x]
    expected_data = sum(math.log1p(math.exp(z)) - int(y) * z for z, y in zip(scalar_logits, labels)) / 3
    assert result["loss"] == pytest.approx(expected_data + penalty * weight ** 2 / 4)
    expected_probability = [1 / (1 + math.exp(-z)) for z in scalar_logits]
    np.testing.assert_allclose(result["probability"][:, 1], expected_probability)
    effective_gradient = (result["gradients"]["W"][0, 1] - result["gradients"]["W"][0, 0]) / 2
    expected_gradient = sum(float(row[0]) * (prob - int(y)) for row, prob, y in zip(x, expected_probability, labels)) / 3 + penalty * weight / 2
    assert effective_gradient == pytest.approx(expected_gradient)


def test_zero_hidden_symmetry_and_visible_forward_overflow_without_zero_penalty_failure():
    f = functions(); parameters = f["mc_initialize"]("mlp", 2, 3, width=3)
    for key in f["mc_parameter_fields"](parameters):
        parameters[key][:] = 0.
    result = f["mc_network_objective"](parameters, [[1., 2.], [2., 1.]], [[1., 0., 0.], [1., 0., 0.]])
    for key in ("W1", "b1", "W2"):
        np.testing.assert_array_equal(result["gradients"][key], 0.)
    assert np.linalg.norm(result["gradients"]["b2"]) > 0
    huge = f["mc_initialize"]("mlp", 1, 2, width=1); huge["W1"][:] = 1e308
    with pytest.raises(FloatingPointError):
        f["mc_forward"](huge, [[1e308]])
    linear = dict(kind="linear", W=np.array([[1e200, -1e200]]), b=np.zeros(2))
    assert math.isfinite(f["mc_network_objective"](linear, [[1e-200]], [[1., 0.]], penalty=0.)["loss"])


def test_training_one_update_snapshot_trace_zero_budget_and_scaler_isolation():
    f = functions(); x = np.array([[-.4, .1], [.2, .3], [.7, -.1]])
    classes = [10, 20, 40]; labels = [10, 40, 20]; q = np.eye(3)[[0, 2, 1]]
    initial = f["mc_initialize"]("mlp", 2, 3, seed=88, width=3)
    evaluated = f["mc_network_objective"](initial, x, q, penalty=.07)
    trained = f["mc_train"]("mlp", x, labels, classes, seed=88, width=3, steps=1, learning_rate=.12, penalty=.07)
    assert trained["iterations"] == 1 and trained["status"] == "fixed_budget" and len(trained["loss_history"]) == 2
    for key in evaluated["gradients"]:
        np.testing.assert_allclose(trained["parameters"][key], initial[key] - .12 * evaluated["gradients"][key])
    assert trained["loss_history"][0] == pytest.approx(scalar_network(initial, x, q, [1.] * 3, .07, 1.))
    assert trained["loss_history"][-1] == pytest.approx(scalar_network(trained["parameters"], x, q, [1.] * 3, .07, 1.))
    empty_budget = f["mc_train"]("linear", x, labels, classes, steps=0)
    assert empty_budget["iterations"] == 0 and len(empty_budget["loss_history"]) == 1
    scaler = f["mc_fit_scaler"]([[-2., 7.], [2., 7.]])
    np.testing.assert_array_equal(scaler["mean"], [0., 7.]); np.testing.assert_array_equal(scaler["scale"], [2., 1.])
    before = copy.deepcopy(scaler)
    np.testing.assert_array_equal(f["mc_scale"]([[10., 9.]], scaler), [[5., 2.]])
    for key in scaler:
        np.testing.assert_array_equal(scaler[key], before[key])


@pytest.mark.parametrize("steps", [-1, 3001, 1.5, True])
def test_training_work_budget_contract(steps):
    with pytest.raises(ValueError):
        functions()["mc_train"]("linear", [[0.]], [0], [0, 1], steps=steps)


def test_displayed_scalar_auditor_agrees_with_independent_objective_and_preserves_probes():
    f = functions(); p = f["mc_initialize"]("mlp", 2, 3, width=3)
    x = np.array([[.2, -.3], [.4, .1], [.7, -.6]]); q = np.array([[.1, .8, .1], [.2, .3, .5], [0., 0., 0.]])
    w = np.array([1., 2., 0.]); before = copy.deepcopy(p)
    assert f["mc_scalar_loss"](p, x, q, w, .07, .8) == pytest.approx(scalar_network(p, x, q, w, .07, .8))
    result = f["mc_gradient_audit"](p, x, q, w)
    assert result["passed"] and result["maximum"] < 1e-7
    for key in f["mc_parameter_fields"](p):
        np.testing.assert_array_equal(p[key], before[key])


def test_metrics_nll_brier_counts_empty_bins_and_explicit_bin_boundary():
    f = functions(); probabilities = np.array([[.8, .2], [.6, .4], [.3, .7], [.9, .1]])
    metrics = f["mc_metrics"](np.log(probabilities), [10, 20, 20, 20], [10, 20], bins=2)
    expected_nll = -(math.log(.8) + math.log(.4) + math.log(.7) + math.log(.1)) / 4
    assert metrics["nll"] == pytest.approx(expected_nll)
    assert metrics["accuracy"] == .5 and metrics["brier"] == pytest.approx(.65)
    assert metrics["ece"] == pytest.approx(.25)
    assert metrics["bins"][0]["count"] == 0 and metrics["bins"][0]["accuracy"] is None
    assert sum(row["count"] for row in metrics["bins"]) == 4
    edge = f["mc_metrics"](np.zeros((2, 2)), [10, 20], [10, 20])
    assert edge["bins"][4]["count"] == 2
    rounded = f["mc_metrics"]([[0., 1e-100, 2e-100]], [40], [10, 20, 40], temperature=1e308)
    np.testing.assert_array_equal(rounded["prediction"], [40])


def test_zero_aggregate_ece_does_not_establish_subgroup_calibration():
    f = functions(); z = np.tile([math.log(4), 0.], (5, 1))
    full = f["mc_metrics"](z, [0, 0, 0, 0, 1], [0, 1])
    assert full["ece"] == pytest.approx(0., abs=3e-16)
    assert f["mc_metrics"](z[:4], [0] * 4, [0, 1])["ece"] == pytest.approx(.2)
    assert f["mc_metrics"](z[4:], [1], [0, 1])["ece"] == pytest.approx(.8)


def test_temperature_grid_against_scalar_losses_and_nonstationary_boundary():
    f = functions(); z = np.tile([math.log(4), 0.], (5, 1)); labels = [0, 0, 0, 0, 1]
    result = f["mc_fit_temperature"](z, labels, [0, 1], [.5, 1., 2.])
    q = np.eye(2)[labels]
    expected = [scalar_categorical(z, q, np.ones(5), t) for t in (.5, 1., 2.)]
    np.testing.assert_allclose([row["nll"] for row in result["table"]], expected)
    assert result["temperature"] == 1. and not result["boundary"]
    assert result["table"][1]["log_temperature_gradient"] == pytest.approx(0., abs=3e-16)
    boundary = f["mc_fit_temperature"](z, labels, [0, 1], [.1, .5])
    assert boundary["boundary"] and abs(boundary["table"][boundary["chosen"]]["log_temperature_gradient"]) > .1
    tie = f["mc_fit_temperature"]([[2.], [3.]], [10, 10], [10], [.5, 1., 2.])
    assert tie["chosen"] == 0 and tie["boundary"]


def test_temperature_convexity_depends_on_coordinate_not_just_scalar_dimension():
    beta = .1
    first = -1 / (1 + math.exp(beta))
    second = math.exp(beta) / (1 + math.exp(beta)) ** 2
    log_temperature_second = beta * first + beta * beta * second
    assert second > 0 and log_temperature_second < -.04
    tau = math.log(10); step = 1e-3
    objective = lambda log_t: math.log1p(math.exp(-math.exp(-log_t)))
    finite_second = (objective(tau + step) - 2 * objective(tau) + objective(tau - step)) / step ** 2
    assert finite_second == pytest.approx(log_temperature_second, rel=2e-6, abs=1e-8)
    result = functions()["mc_categorical_objective"]([[0., 1.]], [[0., 1.]], temperature=10.)
    assert result["log_temperature_gradient"] == pytest.approx(-beta * first)


@pytest.mark.parametrize("grid", [[], [0.], [-1.], [np.nan], [np.inf], [[1.]], [1.] * 17])
def test_temperature_grid_contract(grid):
    with pytest.raises(ValueError):
        functions()["mc_fit_temperature"]([[0., 1.]], [0], [0, 1], grid)


def test_cost_actions_scalar_risks_class_order_and_reject_option():
    f = functions(); p = np.array([[.6, .3, .1]])
    costs = np.array([[0., 1., 5.], [1., 0., 5.], [1., 1., 0.], [.6, .6, .6]])
    result = f["mc_cost_actions"](p, costs)
    expected = [[sum(float(p[0, k]) * float(costs[a, k]) for k in range(3)) for a in range(4)]]
    np.testing.assert_allclose(result["expected_cost"], expected)
    np.testing.assert_allclose(expected, [[.8, 1.1, .9, .6]])
    np.testing.assert_array_equal(result["action"], [3])
    permutation = [2, 0, 1]
    np.testing.assert_array_equal(f["mc_cost_actions"](p[:, permutation], costs[:, permutation])["action"], [3])
    assert f["mc_cost_actions"](np.empty((0, 3)), costs)["action"].shape == (0,)


def test_token_and_sequence_means_padding_replication_and_perplexity_unit():
    f = functions(); z = np.array([[0., 0.], [0., math.log(3)], [0., math.log(7)], [1000., -1000.]])
    w = np.array([1., 1., 1., 0.]); q = np.array([[1., 0.]] * 3 + [[0., 0.]])
    token = f["mc_categorical_objective"](z, q, w)
    sequence = f["mc_categorical_objective"](z, q, [.5, .5, 1., 0.])
    assert token["loss"] == pytest.approx(2 * math.log(2))
    assert sequence["loss"] == pytest.approx(2.25 * math.log(2))
    assert math.exp(token["loss"]) == pytest.approx(4.)
    assert math.exp(sequence["loss"]) == pytest.approx(2 ** 2.25)
    assert (2 ** 1.5 + 2 ** 3) / 2 != pytest.approx(4.)
    np.testing.assert_array_equal(token["gradient"][-1], [0., 0.])
    repeated = f["mc_categorical_objective"](np.tile(z, (2, 1)), np.tile(q, (2, 1)), np.tile(w, 2))
    assert repeated["loss"] == pytest.approx(token["loss"])
    np.testing.assert_allclose(repeated["gradient"][:4] * 2, token["gradient"])


def test_mask_removes_direct_loss_contribution_but_other_shared_paths_remain():
    f = functions(); value = .3
    result = f["mc_categorical_objective"]([[value, 0.], [2 * value, 0.]], [[0., 0.], [0., 1.]], weights=[0., 1.])
    np.testing.assert_array_equal(result["gradient"][0], [0., 0.])
    chain = result["gradient"][0, 0] + 2 * result["gradient"][1, 0]
    independent = lambda candidate: math.log1p(math.exp(2 * float(candidate[0])))
    assert chain == pytest.approx(central_difference(independent, [value])[0], rel=1e-8)
    assert chain > 1.


def test_complete_candidate_protocol_ignores_final_features_and_labels(monkeypatch):
    f = functions(); first = f["mc_prepare_experiment"](steps=1)
    original_sample = f["mc_ring_sample"]

    def alter_final(rng, n):
        result = original_sample(rng, n)
        if n == 300:
            result["X"] *= 1e6
            result["labels"] = np.roll(result["labels"], 1)
        return result

    monkeypatch.setitem(f, "mc_ring_sample", alter_final)
    second = f["mc_prepare_experiment"](steps=1)
    assert len(first["records"]) == len(first["fitted"]) == 18
    assert [(r["kind"], r["smoothing"], r["seed"]) for r in first["records"]] == [
        (kind, epsilon, seed) for kind in ("linear", "mlp") for epsilon in (0., .1, .25) for seed in (40401, 40402, 40403)]
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(first["scaler"][key], second["scaler"][key])
    assert first["chosen"] == second["chosen"] and first["temperature"] == second["temperature"]
    for a, b in zip(first["fitted"], second["fitted"]):
        assert len(a["loss_history"]) == 2 and a["status"] == "fixed_budget"
        for key in f["mc_parameter_fields"](a["parameters"]):
            np.testing.assert_array_equal(a["parameters"][key], b["parameters"][key])
    for a, b in zip(first["records"], second["records"]):
        assert a["validation_nll"] == b["validation_nll"]
        assert a["gradient_target_met"] == (a["gradient_norm"] <= 1e-3)
    assert first["chosen"] == int(np.argmin([r["validation_nll"] for r in first["records"]]))


def test_final_cached_panel_probabilities_costs_and_second_call_guard(monkeypatch):
    f = functions(); study = f["mc_prepare_experiment"](steps=2)
    original_forward = f["mc_forward"]; calls = []

    def counted(parameters, x):
        calls.append(len(x)); return original_forward(parameters, x)

    monkeypatch.setitem(f, "mc_forward", counted)
    result = f["mc_final_panel"](study)
    assert calls == [300] * 18 and study["final_used"] and study["final_result"] is result
    assert result["logits"].shape == (18, 300, 3) and len(result["metrics"]) == 18
    np.testing.assert_array_equal(result["raw"]["prediction"], result["calibrated"]["prediction"])
    target = np.eye(3)[[list(study["classes"]).index(label) for label in study["data"]["final"]["labels"]]]
    selected = study["chosen"]; logits = result["logits"][selected]
    for name, temperature in (("raw", 1.), ("calibrated", study["temperature"]["temperature"])):
        expected = [-sum(float(q) * lp for q, lp in zip(row, scalar_log_probabilities(z, temperature))) for z, row in zip(logits, target)]
        np.testing.assert_allclose(result[name]["row_nll"], expected, atol=2e-14)
    assert result["paired"]["mean"] == pytest.approx(float(np.mean(result["calibrated"]["row_nll"] - result["raw"]["row_nll"])))
    assert len(result["paired"]["draws"]) == 400
    with pytest.raises(RuntimeError, match="already used"):
        f["mc_final_panel"](study)
    assert calls == [300] * 18


def test_final_guard_consumes_access_before_an_inference_exception(monkeypatch):
    f = functions(); study = f["mc_prepare_experiment"](steps=0)

    def interrupted(*args):
        raise RuntimeError("injected forward interruption")

    monkeypatch.setitem(f, "mc_forward", interrupted)
    with pytest.raises(RuntimeError, match="injected"):
        f["mc_final_panel"](study)
    assert study["final_used"]
    with pytest.raises(RuntimeError, match="already used"):
        f["mc_final_panel"](study)


def test_paired_nll_sensitivity_matches_explicit_row_resampling():
    values = np.array([-.3, .2, -.1, -.6]); f = functions()
    actual = f["mc_paired_sensitivity"](values, repeats=80, seed=16)
    rng = np.random.default_rng(16)
    expected = [sum(float(values[i]) for i in rng.integers(4, size=4)) / 4 for _ in range(80)]
    np.testing.assert_allclose(actual["draws"], expected, atol=2e-16)
    np.testing.assert_allclose([actual["lower"], actual["upper"]], np.quantile(expected, [.025, .975]))
