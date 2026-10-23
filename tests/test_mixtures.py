"""Independent objectives and finite cases for the displayed mixture references."""

import copy
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("gaussian-mixtures-and-em")


def scalar_gaussian_2d(x, mean, covariance):
    a, b, c = float(covariance[0][0]), float(covariance[0][1]), float(covariance[1][1])
    determinant = a * c - b * b
    u, v = float(x[0]) - float(mean[0]), float(x[1]) - float(mean[1])
    quadratic = (c * u * u - 2 * b * u * v + a * v * v) / determinant
    return math.exp(-quadratic / 2) / (2 * math.pi * math.sqrt(determinant))


def test_gaussian_normalizers_quadratics_and_tail_values_independently():
    X = [[-.7, 1.1], [2.3, -.4]]
    means = [[.2, -.3], [-1., 2.]]
    covariances = [[[1.2, .3], [.3, .8]], [[.7, -.2], [-.2, 1.5]]]
    expected = [[math.log(scalar_gaussian_2d(x, mean, covariance))
                 for mean, covariance in zip(means, covariances)] for x in X]
    actual = functions()["gaussian_log_densities"](X, means, covariances)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    tail = functions()["gaussian_log_densities"]([[1000.]], [[0.]], [[[1.]]])[0, 0]
    assert tail == pytest.approx(-500000 - math.log(2 * math.pi) / 2)
    assert math.exp(tail) == 0.


@pytest.mark.parametrize("X,means,covariances", [
    ([], [[0.]], [[[1.]]]), ([[0.]], [], []), ([[0., 1.]], [[0.]], [[[1.]]]),
    ([[np.inf]], [[0.]], [[[1.]]]), ([[0.]], [[np.nan]], [[[1.]]]),
    ([[0.]], [[0.]], [[[np.inf]]]), ([[0.]], [[0.]], [[[0.]]]),
    ([[0., 0.]], [[0., 0.]], [[[1., .4], [.1, 1.]]]),
    (np.zeros((2, 9)), np.zeros((1, 9)), np.eye(9)[None]),
])
def test_gaussian_input_and_positive_definiteness_contract(X, means, covariances):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        functions()["gaussian_log_densities"](X, means, covariances)


def test_extreme_posterior_keeps_log_probabilities_and_finite_kl():
    model = dict(weights=[.5, .5], means=[[0.], [1.]], covariances=[[[1.]], [[1.]]])
    posterior = functions()["gmm_e_step"]([[1000.]], model)
    np.testing.assert_array_equal(posterior["responsibilities"], [[0., 1.]])
    logs = posterior["log_responsibilities"][0]
    assert logs[1] - logs[0] == pytest.approx(999.5)
    assert np.isfinite(logs).all()
    audit = functions()["gmm_elbo_audit"]([[1000.]], model, [[1., 0.]])
    assert audit["KL_gap"] == pytest.approx(999.5)
    assert audit["residual"] == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize("weights", [[0., 1.], [-.1, 1.1], [.3, .3], [1.], [np.nan, .5]])
def test_mixture_weights_are_validated_without_silent_clipping(weights):
    model = dict(weights=weights, means=[[0.], [1.]], covariances=[[[1.]], [[1.]]])
    with pytest.raises(ValueError):
        functions()["gmm_e_step"]([[0.]], model)


def test_observed_density_integrates_to_one_on_independent_quadrature_grid():
    model = dict(weights=[.3, .7], means=[[-1.], [2.]], covariances=[[[.5]], [[1.5]]])
    grid = np.linspace(-15, 18, 10001)
    posterior = functions()["gmm_e_step"](grid[:, None], model)
    assert np.trapezoid(np.exp(posterior["log_density"]), grid) == pytest.approx(1., abs=1e-10)
    np.testing.assert_allclose(posterior["responsibilities"].sum(axis=1), 1., atol=1e-15)


@pytest.mark.parametrize("q", [
    [[.1, .9], [.5, .5], [.8, .2]], [[1., 0.], [0., 1.], [.5, .5]],
])
def test_bound_and_kl_against_independent_scalar_density_entropy(q):
    X = [-1., 0., 1.]
    weights, means, variances = [.3, .7], [-1., 2.], [.5, 1.5]
    model = dict(weights=weights, means=np.array(means)[:, None],
                 covariances=np.array(variances).reshape(2, 1, 1))
    likelihood = bound = gap = 0.
    for x, row in zip(X, q):
        joint = [weight * math.exp(-(x - mean) ** 2 / (2 * variance))
                 / math.sqrt(2 * math.pi * variance)
                 for weight, mean, variance in zip(weights, means, variances)]
        total = sum(joint)
        likelihood += math.log(total) / 3
        for probability, mass in zip(row, joint):
            if probability:
                bound += probability * math.log(mass / probability) / 3
                gap += probability * math.log(probability / (mass / total)) / 3
    result = functions()["gmm_elbo_audit"](np.array(X)[:, None], model, q)
    assert result["mean_log_likelihood"] == pytest.approx(likelihood)
    assert result["bound"] == pytest.approx(bound)
    assert result["KL_gap"] == pytest.approx(gap)
    assert gap >= 0 and abs(result["residual"]) < 1e-14
    posterior = functions()["gmm_e_step"](np.array(X)[:, None], model)
    tight = functions()["gmm_elbo_audit"](np.array(X)[:, None], model, posterior["responsibilities"])
    assert abs(tight["KL_gap"]) < 1e-14 and tight["bound"] == pytest.approx(likelihood)


@pytest.mark.parametrize("q", [[[.2, .2]], [[1.1, -.1]], [[np.nan, .5]], [[1.]]])
def test_auxiliary_rows_reject_invalid_simplex_shapes(q):
    model = dict(weights=[.5, .5], means=[[0.], [1.]], covariances=[[[1.]], [[1.]]])
    with pytest.raises(ValueError):
        functions()["gmm_elbo_audit"]([[0.]], model, q)


def test_hand_posterior_m_step_and_fixed_floor_ascent_chain():
    X = np.array([[-1.], [1.]])
    old = dict(weights=np.array([.5, .5]), means=np.array([[-1.], [1.]]),
               covariances=np.array([[[1.]], [[1.]]]))
    posterior = functions()["gmm_e_step"](X, old)
    r = math.exp(2) / (math.exp(2) + 1)
    np.testing.assert_allclose(posterior["responsibilities"], [[r, 1 - r], [1 - r, r]])
    expected_ll = -math.log(2 * math.pi) / 2 + math.log((1 + math.exp(-2)) / 2)
    assert posterior["mean_log_likelihood"] == pytest.approx(expected_ll)
    q = posterior["responsibilities"]
    for floor in (.01, .5):
        updated = functions()["gmm_m_step"](X, q, floor=floor)
        np.testing.assert_allclose(updated["weights"], [.5, .5])
        np.testing.assert_allclose(updated["means"][:, 0], [1 - 2 * r, 2 * r - 1])
        np.testing.assert_allclose(updated["covariances"][:, 0, 0], max(floor, 1 - (2 * r - 1) ** 2))
        before = functions()["gmm_elbo_audit"](X, old, q)
        after = functions()["gmm_elbo_audit"](X, updated, q)
        assert before["bound"] == pytest.approx(before["mean_log_likelihood"])
        assert after["bound"] >= before["bound"]
        assert after["mean_log_likelihood"] >= after["bound"] - 1e-14


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_weighted_m_step_against_scalar_moments_with_ml_denominator(mode):
    X = np.array([[-2., 1.], [1., 2.], [4., -1.]])
    q = np.array([[.8, .2], [.3, .7], [.1, .9]])
    model = functions()["gmm_m_step"](X, q, covariance=mode, floor=1e-6)
    for c in range(2):
        mass = sum(float(row[c]) for row in q)
        mean = [sum(float(row[c]) * float(x[j]) for row, x in zip(q, X)) / mass for j in range(2)]
        covariance = np.array([
            [sum(float(row[c]) * (float(x[a]) - mean[a]) * (float(x[b]) - mean[b])
                 for row, x in zip(q, X)) / mass for b in range(2)] for a in range(2)
        ])
        if mode == "diagonal":
            covariance = np.diag(np.diag(covariance))
        assert model["weights"][c] == pytest.approx(mass / len(X))
        np.testing.assert_allclose(model["means"][c], mean)
        np.testing.assert_allclose(model["covariances"][c], covariance, atol=1e-12)
    duplicated = functions()["gmm_m_step"](np.repeat(X, 2, axis=0), np.repeat(q, 2, axis=0), covariance=mode, floor=1e-6)
    for key in ("weights", "means", "covariances"):
        np.testing.assert_allclose(duplicated[key], model[key], atol=1e-12)


@pytest.mark.parametrize("scatter", [0., .04, 3.])
def test_scalar_covariance_floor_solves_constrained_objective(scatter):
    result = functions()["gmm_covariance_update"]([[scatter]], floor=.1)[0, 0]
    assert result == pytest.approx(max(scatter, .1))
    objective = lambda variance: math.log(variance) + scatter / variance
    assert all(objective(result) <= objective(candidate) + 1e-12
               for candidate in np.geomspace(.1, 10., 101))


def test_rotated_singular_scatter_clips_only_binding_eigenvalue():
    scatter = np.array([[2., 2.], [2., 2.]])
    full = functions()["gmm_covariance_update"](scatter, "full", .25)
    diagonal = functions()["gmm_covariance_update"](scatter, "diagonal", .25)
    np.testing.assert_allclose(full, [[2.125, 1.875], [1.875, 2.125]], atol=1e-13)
    np.testing.assert_array_equal(diagonal, np.diag([2., 2.]))
    assert not np.allclose(full, scatter + .25 * np.eye(2))
    assert np.linalg.eigvalsh(full).min() == pytest.approx(.25)

    def objective(matrix):
        a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
        determinant = a * c - b * b
        return math.log(determinant) + (2 * c + 2 * a - 4 * b) / determinant

    for angle in np.linspace(0, math.pi, 15):
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        for values in ([.25, .25], [.25, 4.], [.5, 3.], [2., 5.]):
            candidate = rotation @ np.diag(values) @ rotation.T
            assert objective(full) <= objective(candidate) + 1e-10


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_non_psd_scatter_and_zero_component_mass_are_not_silently_repaired(mode):
    with pytest.raises(ValueError):
        functions()["gmm_covariance_update"]([[-1.]], mode)
    with pytest.raises(FloatingPointError, match="zero numerical component mass"):
        functions()["gmm_m_step"]([[0.], [1.]], [[1., 0.], [1., 0.]], covariance=mode)


@pytest.mark.parametrize("rows", [1, 4])
def test_observed_nll_all_derivatives_against_independent_scalar_mixture(rows):
    X = np.linspace(-.8, 1.3, rows * 2).reshape(rows, 2)
    means = np.array([[-.3, .4], [.8, -.2]])
    logits = np.array([-.4, .2])
    covariance_parameters = np.array([[1.2, .15, .8], [.7, -.1, 1.4]])
    theta = np.r_[means.ravel(), logits, covariance_parameters.ravel()]

    def unpack(candidate):
        means = candidate[:4].reshape(2, 2)
        logits = candidate[4:6]
        weights = [math.exp(value - max(logits)) for value in logits]
        weights = np.array(weights) / sum(weights)
        covariances = np.array([[[a, b], [b, c]] for a, b, c in candidate[6:].reshape(2, 3)])
        return dict(weights=weights, means=means, covariances=covariances)

    def objective(candidate):
        model = unpack(candidate)
        return -sum(math.log(sum(float(weight) * scalar_gaussian_2d(x, mean, covariance)
                               for weight, mean, covariance in zip(model["weights"], model["means"], model["covariances"])))
                    for x in X) / rows

    model = unpack(theta)
    gradient = functions()["gmm_nll_gradients"](X, model)
    covariance_gradient = gradient["covariances"]
    packed_covariance = np.array([[matrix[0, 0], matrix[0, 1] + matrix[1, 0], matrix[1, 1]]
                                  for matrix in covariance_gradient])
    actual = np.r_[gradient["means"].ravel(), gradient["logits"], packed_covariance.ravel()]
    assert gradient["loss"] == pytest.approx(objective(theta), abs=1e-13)
    np.testing.assert_allclose(actual, central_difference(objective, theta), rtol=2e-6, atol=2e-9)
    duplicated = functions()["gmm_nll_gradients"](np.repeat(X, 2, axis=0), model)
    for key in ("loss", "means", "logits", "covariances"):
        np.testing.assert_allclose(duplicated[key], gradient[key], atol=1e-13)
    assert sum(gradient["logits"]) == pytest.approx(0., abs=1e-15)


def test_em_returned_snapshot_budget_and_tolerance_do_not_claim_stationarity():
    X = np.array([[-1.], [1.]])
    initial = dict(weights=np.array([.5, .5]), means=np.array([[-1.], [1.]]),
                   covariances=np.array([[[1.]], [[1.]]]))
    before = copy.deepcopy(initial)
    limited = functions()["fit_gmm"](X, 2, initial=initial, max_steps=1, tol=0.)
    assert limited["status"] == "budget" and limited["iterations"] == 1
    consistent = functions()["gmm_e_step"](X, limited)
    np.testing.assert_allclose(limited["responsibilities"], consistent["responsibilities"])
    assert limited["mean_log_likelihood"] == consistent["mean_log_likelihood"]
    assert limited["minimum_mass"] == pytest.approx(consistent["responsibilities"].sum(axis=0).min())
    loose = functions()["fit_gmm"](X, 2, initial=initial, max_steps=100, tol=10.)
    assert loose["status"] == "tolerance" and loose["iterations"] == 1
    assert np.linalg.norm(functions()["gmm_nll_gradients"](X, loose)["means"]) > .01
    complete = functions()["fit_gmm"](X, 2, initial=initial, max_steps=100)
    assert complete["status"] == "tolerance" and np.all(np.diff(complete["history"]) >= -1e-12)
    assert np.linalg.eigvalsh(complete["covariances"]).min() >= .01 - 1e-12
    for key in initial:
        np.testing.assert_array_equal(initial[key], before[key])


def test_single_component_solution_and_label_permutation_invariance():
    X = np.array([[-2., 1.], [1., 2.], [4., -1.], [2., 3.]])
    model = functions()["fit_gmm"](X, 1, floor=1e-6)
    mean = [sum(float(x[j]) for x in X) / len(X) for j in range(2)]
    covariance = [[sum((float(x[a]) - mean[a]) * (float(x[b]) - mean[b]) for x in X) / len(X)
                   for b in range(2)] for a in range(2)]
    np.testing.assert_allclose(model["means"], [mean])
    np.testing.assert_allclose(model["covariances"], [covariance])
    mixture = dict(weights=[.3, .7], means=[[-1.], [2.]], covariances=[[[.5]], [[1.5]]])
    permuted = {key: np.asarray(value)[::-1] for key, value in mixture.items()}
    first = functions()["gmm_e_step"]([[-2.], [0.], [3.]], mixture)
    second = functions()["gmm_e_step"]([[-2.], [0.], [3.]], permuted)
    np.testing.assert_allclose(first["log_density"], second["log_density"])
    np.testing.assert_allclose(first["responsibilities"][:, ::-1], second["responsibilities"])


def test_explicit_initial_model_must_obey_floor_mode_and_component_count():
    initial = dict(weights=[.5, .5], means=[[-1.], [1.]], covariances=[[[1e-8]], [[1.]]])
    with pytest.raises(ValueError, match="floor"):
        functions()["fit_gmm"]([[-1.], [1.]], 2, floor=1e-6, initial=initial)
    initial = dict(weights=[1.], means=[[0., 0.]], covariances=[[[1., .2], [.2, 1.]]])
    with pytest.raises(ValueError, match="diagonal"):
        functions()["fit_gmm"]([[0., 0.], [1., 1.]], 1, covariance="diagonal", initial=initial)
    with pytest.raises(ValueError, match="k components"):
        functions()["fit_gmm"]([[0., 0.], [1., 1.]], 2, initial=initial)


@pytest.mark.parametrize("settings", [
    {"k": 0}, {"k": True}, {"k": 9}, {"max_steps": 0}, {"max_steps": 201},
    {"max_steps": True}, {"tol": -1.}, {"tol": np.inf}, {"floor": 0.},
    {"floor": np.nan}, {"floor": True}, {"covariance": "unknown"},
])
def test_fit_work_and_model_contracts(settings):
    parameters = {"k": 2, **settings}
    with pytest.raises(ValueError):
        functions()["fit_gmm"]([[0.], [1.]], **parameters)


def test_restart_records_keep_numerical_failure_and_budget_fits(monkeypatch):
    namespace = functions()
    original = namespace["fit_gmm"]
    calls = []

    def controlled(X, k, seed, **settings):
        calls.append(seed)
        if len(calls) == 1:
            raise FloatingPointError("deliberate arithmetic failure")
        return original(X, k, seed=seed, **settings)

    monkeypatch.setitem(namespace, "fit_gmm", controlled)
    X = np.linspace(-2, 3, 20).reshape(-1, 1)
    model, records, winner = namespace["gmm_restarts"](X, 2, starts=4, seed=22, max_steps=1, tol=0.)
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(22).spawn(4)]
    assert calls == seeds and [record["seed"] for record in records] == seeds
    assert records[0]["status"] == "numerical_failure" and "error" in records[0]
    assert all(record["status"] == "budget" for record in records[1:])
    assert winner == max(range(1, 4), key=lambda i: records[i]["mean_log_likelihood"])
    assert model["status"] == "budget"


def test_all_failed_starts_and_contract_errors_are_distinct(monkeypatch):
    namespace = functions()

    def failed(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate factorization failure")

    with monkeypatch.context() as scoped:
        scoped.setitem(namespace, "fit_gmm", failed)
        model, records, winner = namespace["gmm_restarts"]([[0.], [1.]], 2, starts=3)
        assert model is None and winner is None and len(records) == 3
        assert all(record["status"] == "numerical_failure" for record in records)
    with pytest.raises(ValueError):
        namespace["gmm_restarts"]([[0.], [1.]], 0)
    # Identical values can initialize indistinguishable components; tie wins first.
    assert namespace["gmm_restarts"]([[0.], [0.], [0.]], 2, starts=4)[2] == 0


def test_arbitrary_auxiliary_update_improves_bound_but_lowers_observed_likelihood():
    X = np.array([[-3.], [3.]])
    old = dict(weights=[.5, .5], means=[[-3.], [3.]], covariances=[[[1.]], [[1.]]])
    q = np.full((2, 2), .5)
    new = functions()["gmm_m_step"](X, q)
    before, after = (functions()["gmm_elbo_audit"](X, model, q) for model in (old, new))
    expected_old = -math.log(2 * math.pi) / 2 + math.log((1 + math.exp(-18)) / 2)
    expected_new = -(math.log(2 * math.pi) + math.log(9) + 1) / 2
    expected_bound = -math.log(2 * math.pi) / 2 - 9
    assert before["mean_log_likelihood"] == pytest.approx(expected_old)
    assert after["mean_log_likelihood"] == pytest.approx(expected_new)
    assert before["bound"] == pytest.approx(expected_bound)
    assert after["bound"] > before["bound"]
    assert after["mean_log_likelihood"] < before["mean_log_likelihood"]
    assert before["KL_gap"] > 0 and abs(after["KL_gap"]) < 1e-14


def test_isotropic_hard_limit_keeps_exact_ties_and_concentrates_unique_minimum():
    model = dict(weights=[1 / 3] * 3, means=[[-1.], [1.], [4.]],
                 covariances=[[[1e-4]]] * 3)
    result = functions()["gmm_e_step"]([[0.], [1.2]], model)
    np.testing.assert_allclose(result["responsibilities"][0], [.5, .5, 0.], atol=1e-12)
    np.testing.assert_array_equal(result["responsibilities"][1], [0., 1., 0.])


def test_experiment_data_have_reproducible_separate_draws():
    train, validation = functions()["gmm_experiment_data"]()
    assert train.shape == (180, 2) and validation.shape == (160, 2)
    repeated = functions()["gmm_experiment_data"]()
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])
    assert not np.any(np.all(train[:, None] == validation[None, :], axis=2))
