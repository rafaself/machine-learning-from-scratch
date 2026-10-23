"""Independent conjugate, distributional and derivative checks; never learner cells."""

import copy
from fractions import Fraction
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("bayesian-and-variational-inference")


def rational_beta(a, b):
    return Fraction(math.factorial(a - 1) * math.factorial(b - 1), math.factorial(a + b - 1))


def test_beta_hand_posterior_evidence_moments_and_interior_mode():
    posterior = functions()["beta_update"](2, 2, 3, 1)
    assert (posterior["a"], posterior["b"]) == (5., 3.)
    assert posterior["mean"] == pytest.approx(5 / 8)
    assert posterior["variance"] == pytest.approx(5 / 192)
    assert posterior["interior_map"] == pytest.approx(2 / 3)
    ordered = rational_beta(5, 3) / rational_beta(2, 2)
    assert ordered == Fraction(2, 35)
    assert posterior["log_ordered_evidence"] == pytest.approx(math.log(float(ordered)))
    assert math.exp(posterior["log_ordered_evidence"]) * math.comb(4, 3) == pytest.approx(8 / 35)
    uniform = functions()["beta_update"](1, 1, 0, 0)
    assert uniform["interior_map"] is None and uniform["log_ordered_evidence"] == 0.
    assert uniform["variance"] == pytest.approx(1 / 12)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 3), (5, 3)])
@pytest.mark.parametrize("m", [0, 1, 4, 12])
def test_predictive_tables_against_independent_rational_integration(a, b, m):
    expected = [math.comb(m, k) * rational_beta(a + k, b + m - k) / rational_beta(a, b)
                for k in range(m + 1)]
    assert sum(expected) == 1
    mean = sum(k * p for k, p in enumerate(expected))
    variance = sum((k - mean) ** 2 * p for k, p in enumerate(expected))
    result = functions()["beta_predictive"](a, b, m)
    np.testing.assert_allclose(result["mass"], [float(p) for p in expected], rtol=2e-13, atol=1e-15)
    assert result["mean"] == pytest.approx(float(mean))
    assert result["variance"] == pytest.approx(float(variance))
    if (a, b, m) == (5, 3, 4):
        assert expected == [Fraction(1, 22), Fraction(5, 33), Fraction(3, 11), Fraction(7, 22), Fraction(7, 33)]
        assert result["variance"] == pytest.approx(1.25)
        assert result["plugin_variance"] == pytest.approx(.9375)


def test_predictive_dependence_and_no_double_counted_single_outcome_variance():
    mean, variance = Fraction(5, 8), Fraction(5, 192)
    two_successes = rational_beta(7, 3) / rational_beta(5, 3)
    assert two_successes == Fraction(5, 12)
    assert two_successes - mean * mean == variance
    one = functions()["beta_predictive"](5, 3, 1)
    assert one["variance"] == pytest.approx(15 / 64)
    assert one["parameter_variance"] == pytest.approx(float(variance))
    assert one["variance"] == one["plugin_variance"]
    four = functions()["beta_predictive"](5, 3, 4)
    assert four["variance"] == pytest.approx(float(4 * mean * (1 - mean) + 12 * variance))


def test_sequential_updates_preserve_batch_posterior_and_ordered_evidence():
    observations = [1, 0, 1, 1]
    a, b, log_evidence = 2., 2., 0.
    for value in observations:
        update = functions()["beta_update"](a, b, value, 1 - value)
        a, b = update["a"], update["b"]
        log_evidence += update["log_ordered_evidence"]
    batch = functions()["beta_update"](2, 2, 3, 1)
    assert (a, b) == (batch["a"], batch["b"])
    assert log_evidence == pytest.approx(batch["log_ordered_evidence"], abs=1e-14)


@pytest.mark.parametrize("a,b,expected", [
    (1., 1., [0., 0.]), (1., 3., [math.log(3), -np.inf]),
    (2., 1., [-np.inf, math.log(2)]), (.5, .5, [np.inf, np.inf]),
])
def test_beta_endpoint_limits_do_not_evaluate_zero_times_infinity(a, b, expected):
    actual = functions()["beta_log_density"]([0., 1.], a, b)
    np.testing.assert_allclose(actual, expected)
    assert not np.isnan(actual).any()


def test_beta_density_normalization_and_moments_against_independent_polynomial():
    grid = np.linspace(0, 1, 10001)
    polynomial = 105 * grid ** 4 * (1 - grid) ** 2
    density = np.exp(functions()["beta_log_density"](grid, 5, 3))
    np.testing.assert_allclose(density, polynomial, atol=2e-13)
    assert np.trapezoid(density, grid) == pytest.approx(1., abs=2e-12)
    mean = np.trapezoid(grid * polynomial, grid)
    variance = np.trapezoid((grid - mean) ** 2 * polynomial, grid)
    actual_mean, actual_variance = functions()["beta_moments"](5, 3)
    assert actual_mean == pytest.approx(mean, abs=2e-12)
    assert actual_variance == pytest.approx(variance, abs=2e-12)


@pytest.mark.parametrize("shapes", [(0, 1), (-1, 2), (np.inf, 1), (np.nan, 1), (True, 1)])
def test_beta_requires_proper_finite_shapes(shapes):
    with pytest.raises(ValueError):
        functions()["log_beta"](*shapes)


@pytest.mark.parametrize("h,t", [(-1, 0), (1.5, 0), (True, 0), (10001, 0)])
def test_beta_count_contract(h, t):
    with pytest.raises(ValueError):
        functions()["beta_update"](2, 2, h, t)


@pytest.mark.parametrize("m", [-1, 201, 1.5, True])
def test_predictive_work_limit_and_integer_count(m):
    with pytest.raises(ValueError):
        functions()["beta_predictive"](2, 2, m)


def test_beta_range_failures_and_large_bounded_table_are_visible():
    with pytest.raises(FloatingPointError):
        functions()["log_beta"](1e308, 1e308)
    with pytest.raises(FloatingPointError):
        functions()["beta_moments"](1e308, 1.)
    for value in ([-.1], [1.1], [np.nan]):
        with pytest.raises(ValueError):
            functions()["beta_log_density"](value, 2, 2)
    table = functions()["beta_predictive"](.5, .5, 200)
    assert sum(table["mass"]) == pytest.approx(1., abs=2e-11)
    assert table["mass"][0] == pytest.approx(table["mass"][-1], abs=1e-14)


def test_map_changes_coordinates_with_the_density_jacobian():
    original = functions()["beta_update"](2, 2, 3, 1)["interior_map"]
    mapped = Fraction(5, 8)
    assert original == pytest.approx(2 / 3) and original != float(mapped)

    def transformed_log_density(candidate):
        theta = 1 / (1 + math.exp(-float(candidate[0])))
        # Independent polynomial posterior times the logit Jacobian.
        return math.log(105) + 5 * math.log(theta) + 3 * math.log1p(-theta)

    mode_logit = np.array([math.log(5 / 3)])
    assert abs(central_difference(transformed_log_density, mode_logit)[0]) < 1e-9


def test_realized_variance_increase_and_prior_predictive_total_variance():
    prior_mean = Fraction(10, 11)
    prior_variance = Fraction(10, 1452)
    success_mean, failure_mean = Fraction(11, 12), Fraction(10, 12)
    success_variance, failure_variance = Fraction(11, 1872), Fraction(20, 1872)
    expected_variance = prior_mean * success_variance + (1 - prior_mean) * failure_variance
    mean_variance = prior_mean * (success_mean - prior_mean) ** 2 + (1 - prior_mean) * (failure_mean - prior_mean) ** 2
    assert failure_variance > prior_variance > expected_variance
    assert expected_variance + mean_variance == prior_variance
    actual = functions()["beta_update"](10, 1, 0, 1)
    assert actual["variance"] == pytest.approx(float(failure_variance))
    assert functions()["beta_update"](10, 1, 1, 0)["variance"] == pytest.approx(float(success_variance))


def test_gaussian_hand_posterior_predictive_and_prior_only_case():
    result = functions()["gaussian_mean_update"]([1., 2.])
    assert result["mean"] == pytest.approx(4 / 3)
    assert result["variance"] == pytest.approx(4 / 9)
    assert result["predictive_variance"] == pytest.approx(13 / 9)
    assert result["mean_loss_l2"] == pytest.approx(1 / 8)
    prior = functions()["gaussian_mean_update"]([], prior_mean=2., prior_variance=3., noise_variance=4.)
    assert prior == dict(mean=2., variance=3., predictive_variance=7., mean_loss_l2=None, n_samples=0)


@pytest.mark.parametrize("rows", [1, 4])
def test_gaussian_square_completion_stationarity_and_loss_normalization(rows):
    y = np.linspace(-.5, 1.7, rows)
    prior_mean, prior_variance, noise = -.2, 2.3, .7
    result = functions()["gaussian_mean_update"](y, prior_mean, prior_variance, noise)

    def negative_log_posterior(candidate):
        theta = float(candidate[0])
        return sum((float(value) - theta) ** 2 for value in y) / (2 * noise) + (theta - prior_mean) ** 2 / (2 * prior_variance)

    assert abs(central_difference(negative_log_posterior, [result["mean"]])[0]) < 1e-9
    expected_precision = 1 / prior_variance + rows / noise
    assert result["variance"] == pytest.approx(1 / expected_precision)
    theta = .3
    posterior_gradient = sum(theta - float(value) for value in y) / noise + (theta - prior_mean) / prior_variance
    mean_loss_gradient = sum(theta - float(value) for value in y) / rows + result["mean_loss_l2"] * (theta - prior_mean)
    assert mean_loss_gradient == pytest.approx(posterior_gradient * noise / rows)
    duplicated = functions()["gaussian_mean_update"](np.repeat(y, 2), prior_mean, prior_variance, noise)
    assert duplicated["mean_loss_l2"] == pytest.approx(result["mean_loss_l2"] / 2)
    assert duplicated["variance"] < result["variance"]


@pytest.mark.parametrize("kwargs", [
    {"prior_variance": 0}, {"prior_variance": True}, {"noise_variance": -1},
    {"noise_variance": np.inf}, {"prior_mean": np.nan},
])
def test_gaussian_conjugate_parameter_contracts(kwargs):
    with pytest.raises(ValueError):
        functions()["gaussian_mean_update"]([1.], **kwargs)


def test_gaussian_conjugate_shapes_work_and_normalization_range():
    for y in ([[1.]], [np.inf], np.zeros(10001)):
        with pytest.raises(ValueError):
            functions()["gaussian_mean_update"](y)
    for prior_variance, noise in ((1e-308, 1e308), (1e308, 1e-308)):
        with pytest.raises(FloatingPointError):
            functions()["gaussian_mean_update"]([1.], prior_variance=prior_variance, noise_variance=noise)


def scalar_gaussian_kl(mean_q, covariance_q, mean_p, covariance_p):
    a, b, c = float(covariance_p[0][0]), float(covariance_p[0][1]), float(covariance_p[1][1])
    q11, q12, q22 = float(covariance_q[0][0]), float(covariance_q[0][1]), float(covariance_q[1][1])
    determinant_p, determinant_q = a * c - b * b, q11 * q22 - q12 * q12
    u, v = float(mean_q[0]) - float(mean_p[0]), float(mean_q[1]) - float(mean_p[1])
    trace = (c * q11 - 2 * b * q12 + a * q22) / determinant_p
    quadratic = (c * u * u - 2 * b * u * v + a * v * v) / determinant_p
    return .5 * (trace + quadratic - 2 + math.log(determinant_p / determinant_q))


def test_general_gaussian_kl_direction_and_constants_against_scalar_formula():
    mean_p, covariance_p = [.5, -.3], [[1.2, .4], [.4, .8]]
    mean_q, covariance_q = [-.2, .7], [[.7, -.1], [-.1, 1.4]]
    target = functions()["gaussian_target"](mean_p, covariance_p)
    expected = scalar_gaussian_kl(mean_q, covariance_q, mean_p, covariance_p)
    assert functions()["gaussian_kl"](mean_q, covariance_q, target) == pytest.approx(expected, abs=1e-14)
    assert functions()["gaussian_kl"](mean_p, covariance_p, target) == pytest.approx(0., abs=1e-14)
    reverse_target = functions()["gaussian_target"](mean_q, covariance_q)
    reverse = functions()["gaussian_kl"](mean_p, covariance_p, reverse_target)
    assert reverse == pytest.approx(scalar_gaussian_kl(mean_p, covariance_p, mean_q, covariance_q))
    assert not np.isclose(reverse, expected)


@pytest.mark.parametrize("rho", [0., .3, .8, -.8])
def test_meanfield_optimum_is_precision_conditionals_not_marginals(rho):
    mean = np.array([.5, -.25]); covariance = np.array([[1., rho], [rho, 1.]])
    target = functions()["gaussian_target"](mean, covariance)
    before = copy.deepcopy(target)
    optimum = functions()["meanfield_closed_form"](target)
    np.testing.assert_array_equal(optimum["mean"], mean)
    np.testing.assert_allclose(optimum["variance"], [1 - rho ** 2] * 2)
    assert optimum["kl"] == pytest.approx(-math.log(1 - rho ** 2) / 2, abs=1e-14)
    assert functions()["gaussian_kl"](mean, covariance, target) == pytest.approx(0., abs=1e-14)
    for key in target:
        np.testing.assert_equal(target[key], before[key])


def test_one_sequential_coordinate_sweep_matches_independent_conditional_updates():
    mean_p, covariance_p = [.5, -.25], [[1., .8], [.8, 1.]]
    target = functions()["gaussian_target"](mean_p, covariance_p)
    initial_mean, initial_variance = np.array([2., -2.]), np.ones(2)
    before = initial_mean.copy(), initial_variance.copy()
    result = functions()["fit_meanfield_gaussian"](target, initial_mean, initial_variance, max_sweeps=1, tol=0.)
    first = mean_p[0] + .8 * (initial_mean[1] - mean_p[1])
    second = mean_p[1] + .8 * (first - mean_p[0])
    np.testing.assert_allclose(result["mean"], [first, second])
    np.testing.assert_allclose(result["variance"], [.36, .36])
    expected_trace = [
        scalar_gaussian_kl(initial_mean, np.eye(2), mean_p, covariance_p),
        scalar_gaussian_kl([first, initial_mean[1]], np.diag([.36, 1.]), mean_p, covariance_p),
        scalar_gaussian_kl([first, second], np.diag([.36, .36]), mean_p, covariance_p),
    ]
    np.testing.assert_allclose(result["coordinate_kl"], expected_trace, atol=1e-14)
    assert result["status"] == "budget" and result["sweeps"] == 1
    np.testing.assert_array_equal(initial_mean, before[0])
    np.testing.assert_array_equal(initial_variance, before[1])


def test_meanfield_stopping_status_trace_and_nonzero_restriction_gap():
    target = functions()["gaussian_target"]([.5, -.25], [[1., .8], [.8, 1.]])
    result = functions()["fit_meanfield_gaussian"](target, [2., -2.], [1., 1.])
    assert result["status"] == "tolerance" and result["sweeps"] <= 100
    assert len(result["coordinate_kl"]) == 1 + 2 * result["sweeps"]
    assert result["mean_history"].shape == (result["sweeps"] + 1, 2)
    assert np.all(np.diff(result["coordinate_kl"]) <= 1e-12)
    np.testing.assert_allclose(result["mean"], target["mean"], atol=1e-4)
    assert result["kl"] == pytest.approx(-math.log(.36) / 2, abs=2e-8)
    assert result["kl"] > .5
    assert result["kl"] == pytest.approx(functions()["gaussian_kl"](result["mean"], np.diag(result["variance"]), target))
    loose = functions()["fit_meanfield_gaussian"](target, [2., -2.], [1., 1.], tol=100.)
    assert loose["status"] == "tolerance" and loose["sweeps"] == 1
    assert loose["mean_gradient_norm"] > 1.


@pytest.mark.parametrize("kwargs", [
    {"max_sweeps": 0}, {"max_sweeps": 201}, {"max_sweeps": True}, {"tol": -1},
    {"tol": np.inf}, {"initial_mean": [0.]}, {"initial_variance": [0., 1.]},
    {"initial_mean": [np.nan, 0.]},
])
def test_coordinate_fit_work_and_initial_state_contracts(kwargs):
    target = functions()["gaussian_target"]([0., 0.], [[1., .2], [.2, 1.]])
    with pytest.raises(ValueError):
        functions()["fit_meanfield_gaussian"](target, **kwargs)


@pytest.mark.parametrize("mean,covariance", [
    ([], []), ([0.], [[0.]]), ([0.], [[-1.]]), ([np.nan], [[1.]]),
    ([0., 0.], [[1., .4], [.1, 1.]]), ([0.], [[np.inf]]),
    (np.zeros(9), np.eye(9)),
])
def test_gaussian_target_dimension_spd_and_finiteness(mean, covariance):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        functions()["gaussian_target"](mean, covariance)


def scalar_sample_kl(parameters, epsilon, mean_p, covariance_p):
    means, logs = parameters[:2], parameters[2:]
    std = [math.exp(float(value)) for value in logs]
    a, b, c = covariance_p[0][0], covariance_p[0][1], covariance_p[1][1]
    determinant = a * c - b * b
    total = 0.
    for noise in epsilon:
        sample = [float(means[j]) + std[j] * float(noise[j]) for j in range(2)]
        u, v = sample[0] - mean_p[0], sample[1] - mean_p[1]
        log_p = -math.log(2 * math.pi) - math.log(determinant) / 2 - (c * u * u - 2 * b * u * v + a * v * v) / (2 * determinant)
        log_q = -math.log(2 * math.pi) - sum(float(value) for value in logs) \
            - sum(((sample[j] - float(means[j])) / std[j]) ** 2 for j in range(2)) / 2
        total += log_q - log_p
    return total / len(epsilon)


@pytest.mark.parametrize("rows", [1, 4, 17])
def test_reparameterized_gradients_against_independent_fixed_noise_scalar_objective(rows):
    mean_p, covariance_p = [.5, -.25], [[1., .8], [.8, 1.]]
    target = functions()["gaussian_target"](mean_p, covariance_p)
    parameters = np.r_[.1, .3, np.log([.9, .7])]
    epsilon = np.random.default_rng(294).normal(size=(rows, 2))
    result = functions()["reparameterized_kl"](parameters[:2], parameters[2:], epsilon, target)
    objective = lambda value: scalar_sample_kl(value, epsilon, mean_p, covariance_p)
    actual = np.r_[result["mean_gradient"], result["log_std_gradient"]]
    assert result["kl_estimate"] == pytest.approx(objective(parameters), abs=1e-14)
    np.testing.assert_allclose(actual, central_difference(objective, parameters), rtol=2e-6, atol=2e-8)
    assert result["iid_standard_error"] is None if rows == 1 else result["iid_standard_error"] > 0
    duplicated = functions()["reparameterized_kl"](parameters[:2], parameters[2:], np.repeat(epsilon, 2, axis=0), target)
    for key in ("kl_estimate", "mean_gradient", "log_std_gradient"):
        np.testing.assert_allclose(duplicated[key], result[key], atol=1e-14)


def test_exact_variational_gradients_and_moment_exact_probe():
    mean_p, covariance_p = [.5, -.25], [[1., .8], [.8, 1.]]
    target = functions()["gaussian_target"](mean_p, covariance_p)
    parameters = np.r_[.1, .3, np.log([.9, .7])]

    def objective(value):
        covariance = np.diag([math.exp(2 * float(entry)) for entry in value[2:]])
        return scalar_gaussian_kl(value[:2], covariance, mean_p, covariance_p)

    exact = functions()["diagonal_kl_gradients"](parameters[:2], parameters[2:], target)
    expected_gradient = central_difference(objective, parameters)
    np.testing.assert_allclose(np.r_[exact["mean_gradient"], exact["log_std_gradient"]], expected_gradient, rtol=2e-6, atol=2e-9)
    assert exact["kl"] == pytest.approx(objective(parameters))
    probe = math.sqrt(2) * np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    np.testing.assert_array_equal(probe.mean(axis=0), [0., 0.])
    np.testing.assert_allclose(probe.T @ probe / len(probe), np.eye(2))
    sampled = functions()["reparameterized_kl"](parameters[:2], parameters[2:], probe, target)
    assert sampled["kl_estimate"] == pytest.approx(exact["kl"], abs=1e-14)
    np.testing.assert_allclose(sampled["mean_gradient"], exact["mean_gradient"], atol=1e-14)
    np.testing.assert_allclose(sampled["log_std_gradient"], exact["log_std_gradient"], atol=1e-14)


def test_sampled_negative_kl_is_not_clipped_into_false_certainty():
    target = functions()["gaussian_target"]([0., 0.], np.eye(2))
    estimate = functions()["reparameterized_kl"]([0., 0.], np.log([2., 2.]), [[0., 0.]], target)
    exact = functions()["diagonal_kl_gradients"]([0., 0.], np.log([2., 2.]), target)
    assert estimate["kl_estimate"] == pytest.approx(-2 * math.log(2))
    assert estimate["kl_estimate"] < 0 < exact["kl"]
    assert estimate["iid_standard_error"] is None


def test_variational_shapes_sample_limits_and_variance_range():
    target = functions()["gaussian_target"]([0., 0.], np.eye(2))
    for mean, log_std in (([0.], [0., 0.]), ([0., 0.], [np.nan, 0.])):
        with pytest.raises(ValueError):
            functions()["diagonal_kl_gradients"](mean, log_std, target)
    for log_std in ([1000., 0.], [-1000., 0.]):
        with pytest.raises(FloatingPointError):
            functions()["diagonal_kl_gradients"]([0., 0.], log_std, target)
    for epsilon in (np.zeros((0, 2)), np.zeros((20001, 2)), [[1.]], [[np.inf, 0.]]):
        with pytest.raises(ValueError):
            functions()["reparameterized_kl"]([0., 0.], [0., 0.], epsilon, target)


def test_two_dimensional_contour_radius_uses_mass_not_marginal_sigma_rule():
    radius_squared = -2 * math.log(.05)
    assert 1 - math.exp(-radius_squared / 2) == pytest.approx(.95)
    assert 1 - math.exp(-2.) < .95  # Radius two gives a different joint mass.
