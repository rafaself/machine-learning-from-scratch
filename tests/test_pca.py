"""Independent scalar objectives and geometry checks for displayed PCA references."""

import copy
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("pca-and-svd")


def scalar_covariance(X):
    n, d = len(X), len(X[0])
    means = [sum(float(row[j]) for row in X) / n for j in range(d)]
    return np.array([
        [sum((float(row[a]) - means[a]) * (float(row[b]) - means[b]) for row in X) / n
         for b in range(d)] for a in range(d)
    ])


def test_hand_rectangle_spectrum_scores_projection_and_distortion():
    X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    model = functions()["fit_pca"](X, 1)
    np.testing.assert_allclose(model["mean"], [0., 0.])
    np.testing.assert_allclose(model["singular_values"], [4., 2.])
    np.testing.assert_allclose(model["explained_variance"], [4., 1.])
    np.testing.assert_allclose(model["explained_variance_ratio"], [.8, .2])
    np.testing.assert_allclose(model["components"], [[1.], [0.]], atol=1e-15)
    scores = functions()["pca_transform"](X, model)
    np.testing.assert_allclose(scores[:, 0], [-2., -2., 2., 2.])
    reconstructed = functions()["pca_reconstruct"](scores, model)
    np.testing.assert_allclose(reconstructed, [[-2., 0.], [-2., 0.], [2., 0.], [2., 0.]])
    residual = sum((float(x[j]) - float(y[j])) ** 2 for x, y in zip(X, reconstructed) for j in range(2))
    assert residual == pytest.approx(4.)
    audit = functions()["pca_training_audit"](X, model)
    assert audit["raw_distortion"] == audit["predicted_training_distortion"] == pytest.approx(1.)
    assert model["numerical_rank"] == 2


@pytest.mark.parametrize("shape", [(9, 3), (4, 7)])
@pytest.mark.parametrize("standardize", [False, True])
def test_tall_and_wide_spectra_projectors_against_independent_covariance(shape, standardize):
    X = np.random.default_rng(288).normal(size=shape) + np.arange(shape[1])
    before = X.copy()
    n, d = shape
    rank = min(n - 1, d)
    model = functions()["fit_pca"](X, min(2, rank), standardize=standardize)
    means = np.array([sum(float(row[j]) for row in X) / n for j in range(d)])
    covariance = scalar_covariance(X)
    scales = np.sqrt(np.diag(covariance)) if standardize else np.ones(d)
    expected_covariance = covariance / scales[:, None] / scales[None, :]
    values, vectors = np.linalg.eigh(expected_covariance)
    k = model["components"].shape[1]
    expected_projector = vectors[:, -k:] @ vectors[:, -k:].T
    np.testing.assert_allclose(model["mean"], means)
    np.testing.assert_allclose(model["scales"], scales)
    np.testing.assert_allclose(model["explained_variance"], values[::-1], atol=2e-14)
    np.testing.assert_allclose(model["components"] @ model["components"].T, expected_projector, atol=2e-13)
    assert model["singular_values"].shape == (min(shape),)
    assert model["explained_variance"].shape == (d,)
    assert model["numerical_rank"] == rank
    assert model["rank_tolerance"] == pytest.approx(max(shape) * np.finfo(float).eps * model["singular_values"][0])
    for column in model["components"].T:
        assert column[np.argmax(np.abs(column))] >= 0
    audit = functions()["pca_training_audit"](X, model)
    assert audit["orthogonality_error"] < 2e-14
    assert audit["idempotence_error"] < 2e-14
    assert audit["spectrum_max_error"] < 2e-14
    assert audit["standardized_distortion"] == pytest.approx(sum(values[:-k]), abs=2e-14)
    np.testing.assert_array_equal(X, before)
    # A wide fit may retain a null axis; its basis is nonunique, its training error is zero.
    full = functions()["fit_pca"](X, min(shape), standardize=standardize)
    recovered = functions()["pca_reconstruct"](functions()["pca_transform"](X, full), full)
    np.testing.assert_allclose(recovered, X, atol=2e-13)


def test_covariance_divisor_changes_values_but_not_projector_or_ratios():
    X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    population = scalar_covariance(X)
    sample = population * len(X) / (len(X) - 1)
    p_values, p_vectors = np.linalg.eigh(population)
    s_values, s_vectors = np.linalg.eigh(sample)
    np.testing.assert_allclose(s_values, p_values * 4 / 3)
    np.testing.assert_allclose(s_values / sum(s_values), p_values / sum(p_values))
    np.testing.assert_allclose(p_vectors[:, -1:] @ p_vectors[:, -1:].T,
                               s_vectors[:, -1:] @ s_vectors[:, -1:].T)


@pytest.mark.parametrize("standardize", [False, True])
def test_whitening_covariance_inverse_and_frozen_geometry(standardize):
    X = np.random.default_rng(282).normal(size=(12, 4)) * [1., 4., .2, 2.] + [2., -1., 5., 0.]
    ordinary = functions()["fit_pca"](X, 3, standardize=standardize)
    white = functions()["fit_pca"](X, 3, standardize=standardize, whiten=True)
    Z = functions()["pca_transform"](X, white)
    np.testing.assert_allclose(Z.mean(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(Z.T @ Z / len(X), np.eye(3), atol=2e-14)
    np.testing.assert_allclose(Z.T @ Z / (len(X) - 1), np.eye(3) * 12 / 11, atol=2e-14)
    reconstructed = functions()["pca_reconstruct"](Z, white)
    expected = functions()["pca_reconstruct"](functions()["pca_transform"](X, ordinary), ordinary)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-14)
    audit = functions()["pca_training_audit"](X, white)
    np.testing.assert_allclose(audit["score_covariance"], np.eye(3), atol=2e-14)
    assert audit["standardized_distortion"] == pytest.approx(audit["predicted_training_distortion"])


def test_query_transform_never_refits_mean_scales_or_components():
    X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    model = functions()["fit_pca"](X, 1, standardize=True)
    before = copy.deepcopy(model)
    probe = X + [10., 20.]
    actual = functions()["pca_transform"](probe, model)
    expected = ((probe - model["mean"]) / model["scales"]) @ model["components"]
    np.testing.assert_allclose(actual, expected)
    assert abs(float(actual.mean())) > 1.
    for key in model:
        np.testing.assert_equal(model[key], before[key])


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("whiten", [False, True])
def test_zero_component_baseline_and_empty_queries(standardize, whiten):
    X = np.array([[1., 2.], [3., 4.], [5., 8.]])
    model = functions()["fit_pca"](X, 0, standardize=standardize, whiten=whiten)
    scores = functions()["pca_transform"](X, model)
    assert scores.shape == (3, 0)
    np.testing.assert_allclose(functions()["pca_reconstruct"](scores, model), np.tile(X.mean(axis=0), (3, 1)))
    assert functions()["pca_transform"](np.empty((0, 2)), model).shape == (0, 0)
    assert functions()["pca_reconstruct"](np.empty((0, 0)), model).shape == (0, 2)
    audit = functions()["pca_training_audit"](X, model)
    assert audit["score_covariance"].shape == (0, 0)
    assert audit["orthogonality_error"] == audit["idempotence_error"] == 0.
    assert audit["standardized_distortion"] == pytest.approx(sum(model["explained_variance"]))


def test_constant_features_and_unseen_variation_have_explicit_conventions():
    X = np.array([[0., 7.], [1., 7.], [2., 7.]])
    model = functions()["fit_pca"](X, 1, standardize=True)
    np.testing.assert_array_equal(model["constant_features"], [False, True])
    np.testing.assert_allclose(model["scales"], [math.sqrt(2 / 3), 1.])
    predicted = functions()["pca_reconstruct"](functions()["pca_transform"]([[3., 100.]], model), model)
    np.testing.assert_allclose(predicted, [[3., 7.]])
    with pytest.raises(ValueError, match="null"):
        functions()["fit_pca"](X, 2, standardize=True, whiten=True)
    constant = np.full((4, 3), 7.)
    constant_model = functions()["fit_pca"](constant, 1, standardize=True)
    assert constant_model["numerical_rank"] == 0
    np.testing.assert_array_equal(constant_model["explained_variance_ratio"], [0., 0., 0.])
    np.testing.assert_array_equal(functions()["pca_transform"](constant, constant_model), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="null"):
        functions()["fit_pca"](constant, 1, whiten=True)


def test_numeric_rank_and_weak_component_whitening_use_declared_tolerance():
    X = np.array([[1., 1e-17], [-1., 1e-17], [1., -1e-17], [-1., -1e-17]])
    model = functions()["fit_pca"](X, 2)
    assert model["numerical_rank"] == 1
    assert model["singular_values"][1] < model["rank_tolerance"]
    with pytest.raises(ValueError, match="null"):
        functions()["fit_pca"](X, 2, whiten=True)


def test_scaled_column_norm_avoids_spurious_constant_features_and_exposes_raw_range():
    for magnitude in (1e-300, 1e300):
        X = np.array([[-magnitude], [magnitude]])
        model = functions()["fit_pca"](X, 1, standardize=True)
        assert not model["constant_features"][0]
        assert model["scales"][0] == pytest.approx(magnitude, rel=1e-14, abs=0.)
        np.testing.assert_allclose(model["explained_variance"], [1.])
        with pytest.raises(FloatingPointError):
            functions()["fit_pca"](X, 1)


@pytest.mark.parametrize("settings", [
    {"k": -1}, {"k": 3}, {"k": True}, {"k": 1.5},
    {"standardize": "yes"}, {"whiten": 1},
])
def test_fit_dimension_and_mode_contracts(settings):
    with pytest.raises(ValueError):
        functions()["fit_pca"]([[0., 1.], [2., 3.]], **{"k": 1, **settings})


@pytest.mark.parametrize("X", [[], [[0.]], [[np.inf], [1.]], np.zeros((2, 65)), np.zeros((2001, 1))])
def test_fit_dense_bounds_and_finiteness(X):
    with pytest.raises(ValueError):
        functions()["fit_pca"](X, 0)


def test_transform_reconstruction_and_audit_shape_contracts():
    X = np.array([[0., 1.], [1., 2.], [3., 0.]])
    model = functions()["fit_pca"](X, 1)
    with pytest.raises(ValueError):
        functions()["pca_transform"]([[1.]], model)
    for scores in ([1.], [[1., 2.]], [[np.nan]]):
        with pytest.raises(ValueError):
            functions()["pca_reconstruct"](scores, model)
    with pytest.raises(ValueError):
        functions()["pca_training_audit"](X[:2], model)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("width", [0, 1, 2])
def test_projection_half_loss_gradient_against_independent_scalar_products(rows, width):
    A = np.linspace(-.8, 1.3, rows * 3).reshape(rows, 3)
    W = np.linspace(-.4, .7, 3 * width).reshape(3, width)

    def objective(candidate):
        squared = 0.
        for x in A:
            scores = [sum(float(x[j]) * float(candidate[j, c]) for j in range(3)) for c in range(width)]
            for j in range(3):
                residual = float(x[j]) - sum(scores[c] * float(candidate[j, c]) for c in range(width))
                squared += residual * residual
        return squared / (2 * rows)

    loss, gradient = functions()["projection_loss_gradient"](A, W)
    assert loss == pytest.approx(objective(W))
    assert gradient.shape == W.shape
    np.testing.assert_allclose(gradient, central_difference(objective, W), rtol=2e-6, atol=2e-9)
    duplicate_loss, duplicate_gradient = functions()["projection_loss_gradient"](np.repeat(A, 2, axis=0), W)
    assert duplicate_loss == pytest.approx(loss)
    np.testing.assert_allclose(duplicate_gradient, gradient, atol=1e-14)


def test_zero_trial_matrix_is_stationary_without_being_a_pca_optimum():
    A = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    loss, gradient = functions()["projection_loss_gradient"](A, np.zeros((2, 1)))
    better, _ = functions()["projection_loss_gradient"](A, [[1.], [0.]])
    np.testing.assert_array_equal(gradient, np.zeros((2, 1)))
    assert loss == pytest.approx(2.5) and better == pytest.approx(.5)


def test_rayleigh_grid_and_distinct_tied_projectors_have_correct_objectives():
    X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    for angle in np.linspace(0, math.pi, 101):
        vector = np.array([math.cos(angle), math.sin(angle)])
        scalar_variance = sum(sum(float(x[j]) * vector[j] for j in range(2)) ** 2 for x in X) / len(X)
        residual = X - (X @ vector)[:, None] * vector
        assert scalar_variance <= 4. + 1e-14
        assert np.sum(residual * residual) / len(X) == pytest.approx(5. - scalar_variance, abs=1e-13)
    first = np.diag([1., 0.])
    second = np.full((2, 2), .5)
    assert not np.allclose(first, second)
    for projector in (first, second):
        assert np.trace(np.eye(2)) - np.trace(projector) == 1.
        np.testing.assert_allclose(projector @ projector, projector)


def test_raw_svd_energy_and_centered_affine_variation_are_different():
    X = np.array([[10., -1.], [10., 1.]])
    model = functions()["fit_pca"](X, 1)
    np.testing.assert_allclose(model["mean"], [10., 0.])
    np.testing.assert_allclose(model["components"] @ model["components"].T, [[0., 0.], [0., 1.]], atol=1e-14)
    reconstructed = functions()["pca_reconstruct"](functions()["pca_transform"](X, model), model)
    np.testing.assert_allclose(reconstructed, X)
    _, singular, vt = np.linalg.svd(X, full_matrices=False)
    np.testing.assert_allclose(singular ** 2, [200., 2.])
    raw_reconstructed = (X @ vt[:1].T) @ vt[:1]
    assert np.sum((X - raw_reconstructed) ** 2) == pytest.approx(2.)


def test_outlier_leverage_rotates_axis_and_harms_declared_clean_probe():
    clean = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    augmented = np.vstack([clean, [0., 10.]])
    np.testing.assert_allclose(scalar_covariance(clean), [[2.5, 0.], [0., 0.]])
    np.testing.assert_allclose(scalar_covariance(augmented), [[2., 0.], [0., 16.]])
    old, new = (functions()["fit_pca"](sample, 1) for sample in (clean, augmented))
    np.testing.assert_allclose(new["mean"], [0., 2.])
    np.testing.assert_allclose(old["components"] @ old["components"].T, np.diag([1., 0.]), atol=1e-14)
    np.testing.assert_allclose(new["components"] @ new["components"].T, np.diag([0., 1.]), atol=1e-14)
    reconstructed = functions()["pca_reconstruct"](functions()["pca_transform"](clean, new), new)
    assert np.sum((clean - reconstructed) ** 2) / len(clean) == pytest.approx(2.5)


def test_retaining_99_percent_variance_can_erase_all_target_information():
    X = np.array([[-10., -1.], [-10., 1.], [10., -1.], [10., 1.]])
    y = (X[:, 1] > 0).astype(int)
    model = functions()["fit_pca"](X, 1)
    assert model["explained_variance_ratio"][0] == pytest.approx(100 / 101)
    np.testing.assert_allclose(model["components"] @ model["components"].T, np.diag([1., 0.]), atol=1e-14)
    # Count the exact analytic projection, not floating-point side channels.
    correct = sum(max(sum(y[X[:, 0] == value] == c) for c in (0, 1)) for value in (-10., 10.))
    assert correct / len(X) == .5
    np.testing.assert_array_equal((X[:, 1] > 0).astype(int), y)


def test_nested_basis_validation_reconstruction_can_keep_improving_to_zero():
    rng = np.random.default_rng(285)
    train, probe = rng.normal(size=(15, 3)), rng.normal(size=(12, 3))
    errors = []
    for k in range(4):
        model = functions()["fit_pca"](train, k)
        reconstructed = functions()["pca_reconstruct"](functions()["pca_transform"](probe, model), model)
        errors.append(float(np.sum((probe - reconstructed) ** 2) / len(probe)))
    assert np.all(np.diff(errors) <= 1e-12)
    assert errors[-1] < 1e-28


def test_scaled_metric_tail_is_not_a_raw_unit_reconstruction_error():
    X = np.array([[-20., -1.], [-20., 1.], [20., 0.], [20., 1.], [0., 0.]])
    model = functions()["fit_pca"](X, 1, standardize=True)
    audit = functions()["pca_training_audit"](X, model)
    reconstructed = functions()["pca_reconstruct"](functions()["pca_transform"](X, model), model)
    expected = sum(((float(x[j]) - float(y[j])) / model["scales"][j]) ** 2
                   for x, y in zip(X, reconstructed) for j in range(2)) / len(X)
    assert audit["standardized_distortion"] == pytest.approx(expected)
    assert expected == pytest.approx(audit["predicted_training_distortion"])
    assert not np.isclose(audit["raw_distortion"], expected)


def test_condition_number_squares_and_centered_constructed_weak_directions():
    X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    assert np.linalg.cond(X) == pytest.approx(2.)
    assert np.linalg.cond(scalar_covariance(X)) == pytest.approx(4.)
    u = .5 * np.array([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])
    v = np.array([[1., -1.], [1., 1.]]) / math.sqrt(2)
    np.testing.assert_allclose(u.T @ u, np.eye(2))
    np.testing.assert_allclose(v.T @ v, np.eye(2), atol=1e-15)
    for epsilon in (.01, .000001):
        constructed = (u * [1., epsilon]) @ v.T
        np.testing.assert_allclose(constructed.mean(axis=0), 0., atol=1e-16)
        np.testing.assert_allclose(np.linalg.svd(constructed, compute_uv=False), [1., epsilon], rtol=1e-9)


def test_experiment_samples_are_reproducible_and_separately_drawn():
    train, validation = functions()["pca_experiment_data"]()
    assert train.shape == (120, 6) and validation.shape == (160, 6)
    again = functions()["pca_experiment_data"]()
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(validation, again[1])
    assert not np.any(np.all(train[:, None] == validation[None, :], axis=2))
