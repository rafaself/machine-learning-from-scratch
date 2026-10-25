"""Independent density, derivative, rank, provenance and evaluation checks."""

import copy
from fractions import Fraction
import math

import numpy as np
import pytest

from test_numerics import central_difference, reference_functions


def functions():
    return reference_functions("density-estimation-and-anomaly-detection")


def scalar_kde_logpdf(train, query, h):
    """Tiny scalar mixture oracle, with an independent scalar log-sum-exp."""
    d = len(query)
    exponents = [-sum((float(a) - float(b)) ** 2 for a, b in zip(query, row)) / (2 * h * h)
                 for row in train]
    largest = max(exponents)
    return (largest + math.log(sum(math.exp(x - largest) for x in exponents))
            - math.log(len(train)) - d * math.log(h) - d * math.log(2 * math.pi) / 2)


@pytest.mark.parametrize("h,expected_density,expected_gradient", [
    (1., math.exp(-.5) / math.sqrt(2 * math.pi), 0.),
    (.5, 2 * math.exp(-2) / math.sqrt(2 * math.pi), -3.),
])
def test_hand_kde_density_nll_and_log_bandwidth_gradient(h, expected_density, expected_gradient):
    f = functions()
    loss, gradient = f["kde_nll_log_bandwidth_gradient"]([[-1.], [1.]], [[0.]], h)
    assert math.exp(-loss) == pytest.approx(expected_density, rel=2e-14)
    assert gradient == pytest.approx(expected_gradient, abs=1e-14)
    assert loss == pytest.approx(-math.log(expected_density))


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("block", [1, 7, 256])
def test_multidimensional_kde_against_scalar_mixture_and_independent_difference(d, block):
    rng = np.random.default_rng(901 + d)
    train = rng.normal(size=(11, d))
    queries = rng.normal(size=(17, d)) + .3
    h = .73
    before = train.copy(), queries.copy()
    expected = np.array([scalar_kde_logpdf(train, row, h) for row in queries])
    actual = functions()["kde_density_logpdf"](train, queries, h, block)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)

    def objective(log_bandwidth):
        candidate = math.exp(float(log_bandwidth[0]))
        return -sum(scalar_kde_logpdf(train, row, candidate) for row in queries) / len(queries)

    loss, gradient = functions()["kde_nll_log_bandwidth_gradient"](train, queries, h, block)
    assert loss == pytest.approx(objective([math.log(h)]), abs=2e-14)
    assert gradient == pytest.approx(central_difference(objective, [math.log(h)])[0], rel=2e-7, abs=2e-8)
    # Duplicating independent-objective queries must not change mean-loss scaling.
    duplicated = functions()["kde_nll_log_bandwidth_gradient"](train, np.tile(queries, (3, 1)), h, block)
    np.testing.assert_allclose(duplicated, [loss, gradient], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(train, before[0])
    np.testing.assert_array_equal(queries, before[1])


@pytest.mark.parametrize("h", [.2, .7, 1.4])
def test_kde_integrates_to_one_and_recovers_mixture_moments(h):
    train = np.array([[-1.5], [.25], [1.]])
    grid = np.linspace(-14., 14., 20000)
    density = np.exp(functions()["kde_density_logpdf"](train, grid[:, None], h))
    assert np.trapezoid(density, grid) == pytest.approx(1., abs=2e-12)
    mean = np.trapezoid(grid * density, grid)
    variance = np.trapezoid((grid - mean) ** 2 * density, grid)
    expected_mean = sum(float(x[0]) for x in train) / 3
    expected_variance = sum((float(x[0]) - expected_mean) ** 2 for x in train) / 3 + h * h
    assert mean == pytest.approx(expected_mean, abs=2e-12)
    assert variance == pytest.approx(expected_variance, abs=2e-11)


def test_joint_normalization_contains_h_to_dimension_not_univariate_factor():
    axis = np.linspace(-7., 7., 121)
    a, b = np.meshgrid(axis, axis, indexing="ij")
    query = np.column_stack((a.ravel(), b.ravel()))
    h = .4
    density = np.exp(functions()["kde_density_logpdf"]([[-1., .5], [1., -.5]], query, h)).reshape(a.shape)
    assert np.trapezoid(np.trapezoid(density, axis, axis=1), axis) == pytest.approx(1., abs=2e-12)
    origin = np.exp(functions()["kde_density_logpdf"]([[0., 0.]], [[0., 0.]], h))[0]
    assert origin == pytest.approx(1 / (2 * math.pi * h * h))


def test_direct_density_underflows_but_log_density_and_gradient_remain_finite():
    assert math.exp(-1000 ** 2 / 2) == 0.
    logs = functions()["kde_density_logpdf"]([[0.]], [[1000.]], 1.)
    assert logs[0] == pytest.approx(-500000 - math.log(2 * math.pi) / 2)
    loss, gradient = functions()["kde_nll_log_bandwidth_gradient"]([[0.]], [[1000.]], 1.)
    assert math.isfinite(loss) and gradient == pytest.approx(1 - 1000 ** 2)
    with pytest.raises(FloatingPointError):
        functions()["kde_density_logpdf"]([[-1e308]], [[1e308]], 1.)


def test_training_self_likelihood_is_unbounded_and_duplicates_are_not_loo_independence():
    x = np.array([[-1.], [1.]])
    f = functions()["kde_density_logpdf"]
    ordinary = -f(x, x, .1).mean()
    smaller = -f(x, x, 1e-4).mean()
    assert smaller < ordinary - 6
    # Indexed LOO on duplicate rows retains an identical other center.
    duplicate_log = scalar_kde_logpdf([[1.]], [1.], 1e-4)
    distinct_log = scalar_kde_logpdf([[-1.]], [1.], 1e-4)
    assert duplicate_log > 8 and distinct_log < -1e8


@pytest.mark.parametrize("h", [0., -1., np.nan, np.inf, True, [1.]])
def test_invalid_bandwidth_is_not_clipped(h):
    with pytest.raises(ValueError):
        functions()["kde_density_logpdf"]([[0.]], [[0.]], h)


@pytest.mark.parametrize("block", [0, 257, 1.5, True])
def test_distance_scratch_block_contract(block):
    with pytest.raises(ValueError):
        functions()["kde_density_logpdf"]([[0.]], [[0.]], 1., block)


@pytest.mark.parametrize("bad", [np.empty((0, 1)), np.zeros((2001, 1)), np.zeros((2, 17)), [0., 1.], [[np.nan]]])
def test_fitting_shape_finite_and_work_budgets(bad):
    with pytest.raises(ValueError):
        functions()["fit_density_coordinates"](bad)


def test_scoring_empty_dimensions_and_query_budget():
    f = functions()
    assert f["kde_density_logpdf"]([[1., 2.]], np.empty((0, 2)), .5).shape == (0,)
    gaussian = f["fit_density_gaussian"]([[1., 2.]])
    assert f["gaussian_density_logpdf"](np.empty((0, 2)), gaussian).shape == (0,)
    with pytest.raises(ValueError):
        f["kde_nll_log_bandwidth_gradient"]([[1.]], np.empty((0, 1)), .5)
    for query in (np.zeros((20001, 1)), [[0., 1.]], [[np.inf]]):
        with pytest.raises(ValueError):
            f["kde_density_logpdf"]([[1.]], query, 1.)


def test_standardizer_ml_divisor_constants_frozen_fit_and_range_failure():
    f = functions()
    train = np.array([[-2., 7.], [2., 7.]])
    coordinates = f["fit_density_coordinates"](train)
    np.testing.assert_array_equal(coordinates["mean"], [0., 7.])
    np.testing.assert_array_equal(coordinates["scale"], [2., 1.])
    before = copy.deepcopy(coordinates)
    np.testing.assert_array_equal(f["apply_density_coordinates"]([[10., 9.]], coordinates), [[5., 2.]])
    train[:] = -100
    for key in before:
        np.testing.assert_array_equal(coordinates[key], before[key])
    with pytest.raises(FloatingPointError):
        f["fit_density_coordinates"]([[0.], [1e-300]])
    with pytest.raises(FloatingPointError):
        f["fit_density_coordinates"]([[-1e308], [1e308]])


def test_gaussian_baseline_independent_two_dimensional_covariance_algebra():
    train = [[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]]
    model = functions()["fit_density_gaussian"](train, .02)
    np.testing.assert_allclose(model["mean"], [0., 0.])
    np.testing.assert_allclose(model["covariance"], [[4., 0.], [0., 1.]])
    queries = [[0., 0.], [2., 1.], [-1., 3.]]
    expected = [-math.log(2 * math.pi) - math.log(4) / 2 - .5 * (x * x / 4 + y * y)
                for x, y in queries]
    np.testing.assert_allclose(functions()["gaussian_density_logpdf"](queries, model), expected, atol=2e-14)
    hand = functions()["fit_density_gaussian"]([[-1.], [1.]])
    assert math.exp(functions()["gaussian_density_logpdf"]([[0.]], hand)[0]) == pytest.approx(1 / math.sqrt(2 * math.pi))


def test_covariance_floor_clips_only_small_eigenvalues_and_supports_rank_deficiency():
    f = functions()
    model = f["fit_density_gaussian"]([[-1., -1.], [1., 1.]], .1)
    np.testing.assert_allclose(np.linalg.eigvalsh(model["covariance"]), [.1, 2.], atol=1e-14)
    np.testing.assert_allclose(model["chol"] @ model["chol"].T, model["covariance"])
    one = f["fit_density_gaussian"]([[3., 4.]], .2)
    np.testing.assert_allclose(one["covariance"], np.eye(2) * .2)
    expected = -math.log(2 * math.pi) - math.log(.2)
    assert f["gaussian_density_logpdf"]([[3., 4.]], one)[0] == pytest.approx(expected)
    with pytest.raises(FloatingPointError):
        f["gaussian_density_logpdf"]([[1e308, 0.]], model)


def test_bandwidth_selection_against_independent_losses_first_tie_and_copy():
    f = functions()
    train = np.array([[-1.], [0.], [1.]])
    validation = np.array([[-.2], [.7]])
    grid = [.2, .6, 1.]
    losses = [-sum(scalar_kde_logpdf(train, row, h) for row in validation) / 2 for h in grid]
    fit = f["select_density_bandwidth"](train, validation, grid)
    assert fit["selected_index"] == int(np.argmin(losses))
    np.testing.assert_allclose([r["validation_nll"] for r in fit["validation_table"]], losses)
    tie = f["select_density_bandwidth"](train, validation, [.6, .6])
    assert tie["selected_index"] == 0
    train[:] = 999
    np.testing.assert_array_equal(fit["train"], [[-1.], [0.], [1.]])


@pytest.mark.parametrize("grid", [[], [0.], [np.nan], [np.inf], [[.1]], [.1] * 13])
def test_bandwidth_grid_contract(grid):
    with pytest.raises(ValueError):
        functions()["select_density_bandwidth"]([[0.]], [[0.]], grid)


def test_validation_selection_does_not_refit_coordinates_or_retain_query_rows():
    f = functions()
    train = np.array([[-1., 0.], [1., 2.], [0., 1.]])
    first = f["fit_density_models"](train, [[.1, .8], [.5, 1.4]])
    other = f["fit_density_models"](train, [[100., 150.], [110., 160.]])
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(first["coordinates"][key], other["coordinates"][key])
    np.testing.assert_array_equal(first["kde"]["train"], other["kde"]["train"])
    np.testing.assert_array_equal(first["gaussian"]["covariance"], other["gaussian"]["covariance"])
    assert len(first["kde"]["train"]) == len(train)
    frozen = copy.deepcopy(first)
    query = np.array([[.2, .5]])
    expected = f["score_density_models"](first, query)
    query[:] = 1e5
    f["score_density_models"](first, query)
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(first["coordinates"][key], frozen["coordinates"][key])
    np.testing.assert_array_equal(expected["kde"], f["score_density_models"](first, [[.2, .5]])["kde"])


def independent_conformal(calibration, queries, alpha):
    level = Fraction(str(alpha))
    numerator = [1 + sum(float(c) >= float(q) for c in calibration) for q in queries]
    flags = [Fraction(value, len(calibration) + 1) <= level for value in numerator]
    return numerator, flags


def test_hand_rank_example_and_below_resolution_infinite_threshold():
    f = functions()
    result = f["conformal_density_decisions"](np.arange(1., 10.), [7., 7.1, 9., 10.], "0.35")
    assert result["k"] == 7 and result["threshold"] == 7.
    np.testing.assert_array_equal(result["numerator"], [4, 3, 2, 1])
    np.testing.assert_allclose(result["pvalue"], [.4, .3, .2, .1])
    np.testing.assert_array_equal(result["flags"], [False, True, True, True])
    below = f["conformal_density_decisions"](np.arange(1., 10.), [1e20], "0.05")
    assert math.isinf(below["threshold"]) and below["rejectable_ranks"] == 0
    assert not below["flags"].any()
    # The tempting quantile recipe selects the eighth rather than seventh score.
    assert np.quantile(np.arange(1., 10.), 7 / 9, method="higher") == 8.


@pytest.mark.parametrize("scores", [[0., 1., 2., 3., 4.], [0., 0., 1., 1., 2.], [1., 1., 1., 1., 1.]])
@pytest.mark.parametrize("alpha", ["0.1", "0.2", "0.35", "0.5", "0.9"])
def test_exhaustive_exchangeable_held_out_ranks_with_ties(scores, alpha):
    f = functions()
    rejected = 0
    for index, query in enumerate(scores):
        calibration = scores[:index] + scores[index + 1:]
        expected_numerator, expected_flag = independent_conformal(calibration, [query], alpha)
        result = f["conformal_density_decisions"](calibration, [query], alpha)
        np.testing.assert_array_equal(result["numerator"], expected_numerator)
        np.testing.assert_array_equal(result["flags"], expected_flag)
        assert result["flags"][0] == (query > result["threshold"])
        rejected += int(result["flags"][0])
    assert Fraction(rejected, len(scores)) <= Fraction(alpha)
    if len(set(scores)) == len(scores):
        expected = Fraction(alpha) * len(scores)
        assert rejected == expected.numerator // expected.denominator
    if len(set(scores)) == 1:
        assert rejected == 0


def test_decimal_alpha_rank_boundaries_and_float_reporting_never_decide_rejection():
    f = functions()
    for alpha in ("0.3", .3, Fraction(3, 10)):
        result = f["conformal_density_decisions"](np.arange(9.), [6., 6.1], alpha)
        assert result["rejectable_ranks"] == 3 and result["k"] == 7
        np.testing.assert_array_equal(result["flags"], [False, True])
    # The explicitly documented policy preserves the supplied decimal value.
    assert f["density_alpha"](.1 + .2) == Fraction("0.30000000000000004")
    many = f["conformal_density_threshold"](np.arange(999.), "0.001")
    assert many["rejectable_ranks"] == 1 and many["threshold"] == 998.


@pytest.mark.parametrize("alpha", [0., 1., -1., np.nan, np.inf, True, "1/0", ["0.1"]])
def test_invalid_alpha_fails_explicitly(alpha):
    with pytest.raises(ValueError):
        functions()["conformal_density_threshold"]([0., 1.], alpha)


@pytest.mark.parametrize("scores", [[], [np.nan], [np.inf], [[1., 2.]], np.zeros(20001)])
def test_calibration_shape_finiteness_and_budget(scores):
    with pytest.raises(ValueError):
        functions()["conformal_density_threshold"](scores)


def test_rank_scores_monotone_invariance_empty_queries_and_no_mutation():
    f = functions()
    calibration = np.array([2., 0., 1., 1.])
    query = np.array([-1., 1., 1.5, 3.])
    before = calibration.copy()
    first = f["conformal_density_decisions"](calibration, query, "0.4")
    changed = f["conformal_density_decisions"](np.exp(calibration), np.exp(query), "0.4")
    np.testing.assert_array_equal(first["flags"], changed["flags"])
    np.testing.assert_array_equal(first["numerator"], changed["numerator"])
    assert f["conformal_density_decisions"](calibration, [])["flags"].shape == (0,)
    np.testing.assert_array_equal(calibration, before)


@pytest.mark.parametrize("n,b", [(1, 0), (1, 1), (400, 0), (400, 400), (400, 30)])
def test_flag_reports_against_independent_wilson_quadratic_roots(n, b):
    result = functions()["density_flag_report"](np.arange(n) < b)
    assert (result["n"], result["flagged"], result["rate"]) == (n, b, b / n)
    # Solve N*(rhat-p)^2 = z^2*p*(1-p) as a separate quadratic.
    z = 1.959963984540054
    roots = np.sort(np.roots([n + z * z, -2 * b - z * z, b * b / n]))
    np.testing.assert_allclose(result["approximate_95_interval"], roots, atol=2e-14)
    low, high = result["approximate_95_interval"]
    assert 0 <= low <= b / n <= high <= 1


@pytest.mark.parametrize("flags", [[], [0, 1], [[True]], [True] * 20001])
def test_metric_input_contract(flags):
    with pytest.raises(ValueError):
        functions()["density_flag_report"](flags)


def test_unit_jacobian_and_calibrated_decisions_for_both_fitted_models():
    f = functions()
    rng = np.random.default_rng(914)
    train = rng.normal(size=(40, 2)) * [1., .3] + [2., -1.]
    validation = rng.normal(size=(23, 2)) * [1., .3] + [2., -1.]
    calibration = rng.normal(size=(19, 2)) * [1., .3] + [2., -1.]
    query = rng.normal(size=(30, 2)) * [1., .3] + [2., -1.]
    scale, offset = np.array([1000., .01]), np.array([7., -3.])
    original = f["fit_density_models"](train, validation)
    converted = f["fit_density_models"](train * scale + offset, validation * scale + offset)
    a = f["score_density_models"](original, query)
    b = f["score_density_models"](converted, query * scale + offset)
    ac = f["score_density_models"](original, calibration)
    bc = f["score_density_models"](converted, calibration * scale + offset)
    assert original["kde"]["selected_index"] == converted["kde"]["selected_index"]
    for name in ("gaussian", "kde"):
        np.testing.assert_allclose(b[name] - a[name], math.log(10), atol=2e-11)
        np.testing.assert_array_equal(f["conformal_density_decisions"](ac[name], a[name])["flags"],
                                      f["conformal_density_decisions"](bc[name], b[name])["flags"])


def test_typicality_radius_density_point_density_and_noninvertible_projection():
    d = 64
    mode = math.sqrt(d - 1)
    derivative = (d - 1) / mode - mode
    assert derivative == pytest.approx(0., abs=2e-15)
    log_density_ratio = (8 ** 2 - 0 ** 2) / 2
    assert log_density_ratio == 32
    # Same retained coordinate, different discarded coordinate: identical marginal,
    # different joint density. A projected density cannot specify the joint.
    marginal = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    joint0 = marginal(1) * marginal(0)
    joint4 = marginal(1) * marginal(4)
    assert joint0 / joint4 == pytest.approx(math.exp(8))


def test_all_controlled_comparison_records_counts_and_information_boundaries():
    f = functions()
    study = f["density_comparison"]()
    again = f["density_comparison"]()
    assert study["records"] == again["records"]
    assert len(study["records"]) == 6
    assert {(r["condition"], r["method"]) for r in study["records"]} == {
        (condition, method) for condition in ("clean", "training_contamination", "calibration_contamination")
        for method in ("gaussian", "kde")}
    data = study["data"]
    for condition, training, calibration, fit_key in (
        ("clean", data["train"], data["calibration"], "clean"),
        ("training_contamination", data["dirty_train"], data["calibration"], "training_contamination"),
        ("calibration_contamination", data["train"], data["dirty_calibration"], "clean"),
    ):
        fit = study["fits"][fit_key]
        expected_mean = [sum(float(row[j]) for row in training) / len(training) for j in range(2)]
        np.testing.assert_allclose(fit["coordinates"]["mean"], expected_mean, atol=1e-14)
        assert fit["kde"]["train"].shape == (240, 2)
        assert len(fit["kde"]["validation_table"]) == 5
        score_cal = f["score_density_models"](fit, calibration)
        for method in ("gaussian", "kde"):
            record = next(r for r in study["records"] if r["condition"] == condition and r["method"] == method)
            assert record["alpha"] == "0.1" and record["calibration_n"] == 199
            for population in ("nominal", "gap", "shifted_nominal"):
                query_score = f["score_density_models"](fit, data[population])[method]
                _, flags = independent_conformal(score_cal[method], query_score, "0.1")
                report = record["evaluations"][population]
                assert report["n"] == 400 and report["flagged"] == sum(flags)
                assert report["rate"] == sum(flags) / 400
    clean_kde = next(r for r in study["records"] if r["condition"] == "clean" and r["method"] == "kde")
    assert clean_kde["evaluations"]["gap"]["flagged"] == 400
    assert clean_kde["evaluations"]["nominal"]["flagged"] == 29
    assert clean_kde["evaluations"]["shifted_nominal"]["flagged"] == 222
    dirty = next(r for r in study["records"] if r["condition"] == "calibration_contamination" and r["method"] == "kde")
    assert dirty["evaluations"]["gap"]["flagged"] == 0
    # These are fixture-specific counts, not claimed universal statistical guarantees.


def test_uniform_calibration_repeats_are_complete_paired_and_reproducible():
    f = functions()
    records = f["density_calibration_repeats"]()
    assert records == f["density_calibration_repeats"]()
    assert len(records) == 256
    for repeat in range(128):
        pair = [r for r in records if r["repeat"] == repeat]
        assert [r["n"] for r in pair] == [19, 99]
        for row in pair:
            assert row["exact_marginal_rate"] == .1
            assert row["conditional_nominal_rate"] == 1 - row["threshold"]
            assert 0 <= row["conditional_nominal_rate"] <= 1
    # Existence of high conditional rates is consistent with marginal validity.
    assert max(r["conditional_nominal_rate"] for r in records if r["n"] == 19) > .25
